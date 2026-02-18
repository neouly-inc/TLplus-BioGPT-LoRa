"""Orchestrator for TL++ on PubMedQA with BioGPT (+LoRA)."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict

from transformers import get_linear_schedule_with_warmup

from runtime.protocol import (
    OrchestratorCommunicator, SecureOrchestratorCommunicator,
    SecretSharing, SecureSocketCommunicator, SecureMessageType,
)
from core.data import OrchestratorDataLoader, NUM_PUBMEDQA_CLASSES
from core.models import (
    create_orchestrator_model, create_node_model,
    DEFAULT_LORA_R, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_TARGET_MODULES,
)
from runtime.utils import (
    BatchScheduler, GradientAggregator, ModelEvaluator, DataMerger,
    SecureDataMerger, SecureEvaluator,
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8080
DEFAULT_HELPER_PORT = 8082
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_TEST_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 500
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_MAX_LENGTH = 512
NOTIFICATION_SLEEP_TIME = 1

KST = timezone(timedelta(hours=9))


# ==============================================================================
# LOGGING
# ==============================================================================


class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=KST)
        return dt.strftime(datefmt) if datefmt else dt.isoformat()


def setup_logging(log_dir: str = 'logs', prefix: str = 'TL++_biogpt_base') -> Path:
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'{prefix}_{timestamp}.log'
    formatter = KSTFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S KST',
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.info("=" * 80)
    logging.info("TL++ Orchestrator  [BioGPT / PubMedQA]")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")
    return log_file


# ==============================================================================
# SECURE TRAINING PHASES  (adapted for BioGPT & attention_mask)
# ==============================================================================


class SecureTrainingPhases:
    """Encapsulates the four phases of secure MPC training."""

    def __init__(self, orchestrator: 'SecureOrchestrator'):
        self.orch = orchestrator
        self.comm_helper = SecureSocketCommunicator()

    # ---- Phase 1 ----

    def phase1_collect_shares(self, sample_indices: List) -> Optional[Tuple]:
        node_state = self.orch.model.get_node_state()
        self.orch.comm.broadcast_batch_assignment(node_state, sample_indices)

        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {
                'phase': 'forward',
                'model_state': self.orch.model.state_dict(),
                'cut_layer': self.orch.config.get('cut_layer', 1),
                'num_classes': NUM_PUBMEDQA_CLASSES,
                'n_nodes': len(self.orch.comm.node_handlers),
            },
        )

        forward_results_orch = self.orch.comm.collect_forward_results()

        msg_type, helper_forward = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type == SecureMessageType.SHUTDOWN:
            self._send_empty_signals(len(forward_results_orch))
            return None

        merged_data = SecureDataMerger.merge_shares(forward_results_orch, self.orch.device)
        if merged_data is None:
            self.comm_helper._send_message(
                self.orch.comm.helper_socket, SecureMessageType.SHUTDOWN, {}
            )
            self._send_empty_signals(len(forward_results_orch))
            return None

        return merged_data  # (merged_share_0, merged_attn, merged_labels, sizes, valid)

    # ---- Phase 2 ----

    def phase2_forward_computation(self, merged_share_0, merged_attn) -> Tuple:
        self.orch.model.train()
        merged_share_0_input = merged_share_0.requires_grad_(True)
        outputs_share_0 = self.orch.model.forward_from_cut(merged_share_0_input, merged_attn)

        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {'phase': 'compute', 'action': 'forward'},
        )

        msg_type, helper_output = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type != SecureMessageType.HELPER_READY:
            return None, None, None

        outputs_share_1 = helper_output['output_share'].to(self.orch.device)
        return outputs_share_0, outputs_share_1, merged_share_0_input

    # ---- Phase 3 ----

    def phase3_loss_and_backward(
        self,
        outputs_share_0,
        outputs_share_1,
        merged_share_0_input,
        merged_labels,
    ) -> Tuple:
        outputs_for_loss = outputs_share_0.detach() + outputs_share_1.detach()
        outputs_for_loss.requires_grad_(True)

        loss = self.orch.criterion(outputs_for_loss, merged_labels)
        loss.backward()
        output_gradient = outputs_for_loss.grad.clone()

        self.orch.optimizer.zero_grad()
        outputs_share_0.backward(output_gradient)
        cut_grad_share_0 = merged_share_0_input.grad

        if cut_grad_share_0 is None or output_gradient is None:
            self.comm_helper._send_message(
                self.orch.comm.helper_socket, SecureMessageType.SHUTDOWN, {}
            )
            return None, None

        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {
                'phase': 'compute',
                'action': 'backward',
                'output_gradient': output_gradient.detach().cpu(),
            },
        )

        msg_type, helper_grad = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type != SecureMessageType.HELPER_READY:
            return None, loss.item()

        cut_grad_share_1 = helper_grad['cut_gradient'].to(self.orch.device)
        return cut_grad_share_0 + cut_grad_share_1, loss.item()

    # ---- Phase 4 ----

    def phase4_distribute_gradients(self, cut_gradient, split_sizes, valid_nodes) -> bool:
        share_0_new, share_1_new = SecretSharing.share_tensor(
            cut_gradient, SecretSharing._gradient_noise_scale
        )
        grad_splits_0 = torch.split(share_0_new, split_sizes, dim=0)
        grad_splits_1 = torch.split(share_1_new, split_sizes, dim=0)

        grad_shares_orch, grad_shares_helper = [], []
        g = 0
        for is_valid in valid_nodes:
            if is_valid:
                grad_shares_orch.append(grad_splits_0[g].detach().cpu())
                grad_shares_helper.append(grad_splits_1[g].detach().cpu())
                g += 1
            else:
                grad_shares_orch.append(None)
                grad_shares_helper.append(None)

        self.orch.comm.broadcast_backward_signal(grad_shares_orch)
        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {'phase': 'backward', 'gradient_shares': grad_shares_helper},
        )

        grad_results_orch = self.orch.comm.collect_gradient_results()
        msg_type, helper_param_grads = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type == SecureMessageType.SHUTDOWN:
            return False

        grad_results_helper = helper_param_grads.get('gradient_shares', [])
        self._apply_reconstructed_gradients(grad_results_orch, grad_results_helper, valid_nodes)
        return True

    def _apply_reconstructed_gradients(self, grads_orch, grads_helper, valid_nodes):
        node_param_names = self.orch.model.get_node_param_names()
        reconstructed = []
        for g_orch, g_helper, is_valid in zip(grads_orch, grads_helper, valid_nodes):
            if not is_valid or not g_orch or not g_helper:
                reconstructed.append({})
                continue
            rec = {}
            for name in node_param_names:
                if name in g_orch and name in g_helper:
                    rec[name] = SecretSharing.reconstruct_tensor(g_orch[name], g_helper[name])
            reconstructed.append(rec)

        aggregated = GradientAggregator.aggregate_gradients(
            reconstructed, node_param_names, valid_nodes
        )
        for name, param in self.orch.model.named_parameters():
            if name in aggregated:
                if param.grad is None:
                    param.grad = aggregated[name].to(self.orch.device)
                else:
                    param.grad += aggregated[name].to(self.orch.device)

        if self.orch.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.orch.model.parameters(), self.orch.grad_clip_norm)
        self.orch.optimizer.step()
        self.orch.scheduler.step()

    def _send_empty_signals(self, n_nodes: int) -> None:
        self.orch.comm.broadcast_backward_signal([None] * n_nodes)
        self.orch.comm.collect_gradient_results()


# ==============================================================================
# MAIN ORCHESTRATOR CLASS
# ==============================================================================


class SecureOrchestrator:
    """Orchestrator for distributed BioGPT fine-tuning on PubMedQA."""

    def __init__(self, config: dict):
        self.config = config
        self.secure_mode = config.get('secure', False)
        self._setup_device()

        logging.info(f"Computation device: {self.device}")
        logging.info(f"Secure mode: {'✓ ENABLED' if self.secure_mode else 'DISABLED'}")
        logging.info("")

        self._setup_communication()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.test_loader = None
        self.batch_scheduler = None
        self.evaluator = None
        self.grad_clip_norm = None

        if self.secure_mode:
            self.secure_phases = SecureTrainingPhases(self)

    def _setup_device(self) -> None:
        if self.config.get('no_accel', False):
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def _setup_communication(self) -> None:
        if self.secure_mode:
            self.comm = SecureOrchestratorCommunicator(
                host=self.config.get('host', DEFAULT_HOST),
                port=self.config.get('port', DEFAULT_PORT),
                helper_host=self.config.get('helper_host', DEFAULT_HOST),
                helper_port=self.config.get('helper_port', DEFAULT_HELPER_PORT),
            )
        else:
            self.comm = OrchestratorCommunicator(
                host=self.config.get('host', DEFAULT_HOST),
                port=self.config.get('port', DEFAULT_PORT),
            )

    # ---- Setup ----

    def setup(self):
        logging.info("-" * 80)
        logging.info("NETWORK SETUP")
        logging.info("-" * 80)

        self.comm.start()
        logging.info(f"✓ Orchestrator listening on {self.config.get('host')}:{self.config.get('port')}")

        if self.secure_mode:
            self._setup_helper()

        self._accept_nodes()
        self._setup_model()
        self._setup_data()
        self._setup_optimization()

        logging.info("=" * 80)
        logging.info("Setup complete — Ready for training")
        logging.info("=" * 80)
        logging.info("")

    def _setup_helper(self) -> None:
        logging.info("Waiting for helper node …")
        self.comm.wait_for_helper()
        logging.info("✓ Helper node connected")
        act_noise, grad_noise = SecretSharing.get_noise_scaling()
        self.comm.send_noise_config_to_helper(act_noise, grad_noise)
        logging.info(f"✓ Noise config sent (activation: {act_noise:.1%}, gradient: {grad_noise:.1%})")
        logging.info("")

    def _accept_nodes(self) -> None:
        n_nodes = self.config.get('n_nodes', 1)
        logging.info(f"Waiting for {n_nodes} node(s) …")
        for i in range(n_nodes):
            self.comm.accept_node()
            logging.info(f"  ✓ Node {i+1}/{n_nodes} connected")
        logging.info(f"✓ All {n_nodes} node(s) connected")
        logging.info("")

    def _setup_model(self) -> None:
        logging.info("-" * 80)
        logging.info("MODEL INITIALISATION  [BioGPT + LoRA on orch-side layers]")
        logging.info("-" * 80)

        cut_layer = self.config.get('cut_layer', 1)
        n_nodes = self.config.get('n_nodes', 1)
        use_lora = self.config.get('use_lora', False)
        max_length = self.config.get('max_length', DEFAULT_MAX_LENGTH)

        # Compute per-class node counts for fair data partitioning
        nodes_per_class = [0, 0, 0]  # yes / no / maybe
        for node_idx in range(n_nodes):
            pubmedqa_class = node_idx % 3
            nodes_per_class[pubmedqa_class] += 1

        # Node model template (pretrained, used to initialise each node)
        node_model_template = create_node_model(cut_layer=cut_layer)
        node_params = sum(p.numel() for p in node_model_template.parameters())
        logging.info(f"Node-side parameters:      {node_params:,}")

        # Broadcast config to each node
        logging.info("")
        logging.info("Broadcasting configuration to nodes …")

        for handler in self.comm.node_handlers:
            node_idx = handler.node_id - 1           # 0-indexed
            pubmedqa_class = node_idx % 3
            rank_in_class = node_idx // 3
            n_in_class = nodes_per_class[pubmedqa_class]

            config_dict = {
                'node_id': handler.node_id,
                'node_model': node_model_template,
                'max_length': max_length,
                'n_nodes_in_class': n_in_class,
                'node_rank_in_class': rank_in_class,
            }
            if self.secure_mode:
                act_noise, grad_noise = SecretSharing.get_noise_scaling()
                config_dict['activation_noise'] = act_noise
                config_dict['gradient_noise'] = grad_noise

            handler.send_init(config_dict)

        logging.info("✓ Configuration broadcast to all nodes")

        # Full orchestrator model
        self.model = create_orchestrator_model(
            num_classes=NUM_PUBMEDQA_CLASSES,
            cut_layer=cut_layer,
            use_lora=use_lora,
            lora_r=self.config.get('lora_r', DEFAULT_LORA_R),
            lora_alpha=self.config.get('lora_alpha', DEFAULT_LORA_ALPHA),
            lora_dropout=self.config.get('lora_dropout', DEFAULT_LORA_DROPOUT),
            lora_target_modules=self.config.get('lora_target_modules'),
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total parameters:          {total_params:,}")
        logging.info(f"Trainable parameters:      {trainable_params:,}")
        if use_lora:
            logging.info(f"  (LoRA on orch-side layers, full fine-tune on node-side layers)")

    def _setup_data(self) -> None:
        logging.info("")
        logging.info("-" * 80)
        logging.info("DATA LOADING")
        logging.info("-" * 80)

        samples_per_node = self.comm.collect_dataset_sizes()
        total = sum(samples_per_node)
        logging.info("✓ Dataset sizes received:")
        for i, s in enumerate(samples_per_node):
            logging.info(f"    Node {i+1}: {s:,} samples")
        logging.info(f"  Total training: {total:,}")
        logging.info("")

        self.test_loader = OrchestratorDataLoader(
            batch_size=self.config.get('test_batch_size', DEFAULT_TEST_BATCH_SIZE),
            data_dir=self.config.get('data_dir', './data'),
            max_length=self.config.get('max_length', DEFAULT_MAX_LENGTH),
        )
        logging.info(f"✓ Test set: {len(self.test_loader):,} samples")
        logging.info("")

        self.batch_scheduler = BatchScheduler(
            samples_per_node=samples_per_node,
            batch_size=self.config.get('train_batch_size', DEFAULT_TRAIN_BATCH_SIZE),
            shuffle=True,
        )
        logging.info(f"✓ Batch scheduler: {len(self.batch_scheduler)} batches/epoch")
        logging.info("")

    def _setup_optimization(self) -> None:
        logging.info("-" * 80)
        logging.info("OPTIMISATION SETUP")
        logging.info("-" * 80)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.get('lr', DEFAULT_LR),
            weight_decay=self.config.get('weight_decay', DEFAULT_WEIGHT_DECAY),
            eps=1e-8,
        )

        epochs = self.config.get('epochs', DEFAULT_EPOCHS)
        total_steps = len(self.batch_scheduler) * epochs
        warmup_steps = self.config.get('warmup_steps', DEFAULT_WARMUP_STEPS)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.grad_clip_norm = self.config.get('grad_clip_norm', DEFAULT_GRAD_CLIP_NORM)

        if self.secure_mode:
            self.evaluator = SecureEvaluator()
        else:
            self.evaluator = ModelEvaluator(self.criterion)

        logging.info(f"✓ Optimizer:  AdamW (lr={self.config.get('lr', DEFAULT_LR):.2e})")
        logging.info(f"✓ Scheduler:  Linear warmup ({warmup_steps} steps / {total_steps} total)")
        logging.info(f"✓ Grad clip:  {self.grad_clip_norm}")
        logging.info("")

    # ---- Training ----

    def train(self):
        logging.info("-" * 80)
        logging.info("TRAINING")
        logging.info("-" * 80)

        epochs = self.config.get('epochs', DEFAULT_EPOCHS)
        patience = self.config.get('patience', DEFAULT_PATIENCE)

        best_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info("")
            logging.info(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6e}")

            train_loss = self.train_epoch(epoch, epochs)
            test_loss, test_acc, info = self.evaluate()

            logging.info(
                f"  Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}%"
            )

            # Per-class breakdown
            report = info.get('classification_report', {})
            if report:
                for cls in ['yes', 'no', 'maybe']:
                    if cls in report:
                        f1 = report[cls].get('f1-score', 0) * 100
                        logging.info(f"    {cls:6s}: F1={f1:5.2f}%")

            if info.get('unique_predictions', 0) <= 1:
                logging.warning("  ⚠️  Model predicting only 1 class — possible collapse")

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                patience_counter = 0
                logging.info(f"  🏆 New best accuracy: {best_acc:.2f}%")
            else:
                patience_counter += 1
                logging.info(f"  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                logging.info(f"\nEarly stopping after {patience} epochs without improvement")
                break

        logging.info("")
        logging.info("=" * 80)
        logging.info("TRAINING COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Best accuracy:  {best_acc:.2f}% (Epoch {best_epoch})")
        logging.info("=" * 80)

        logging.info("")
        logging.info("Notifying nodes of completion …")
        self._notify_training_complete()

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        batches = self.batch_scheduler.create_epoch_batches()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(batches, desc=f'Epoch {epoch+1}/{total_epochs}', ncols=100, leave=False)
        for sample_indices in pbar:
            if self.secure_mode:
                loss = self._train_batch_secure(sample_indices)
            else:
                loss = self._train_batch_standard(sample_indices)

            if loss is not None:
                epoch_loss += loss
                n_batches += 1
                pbar.set_postfix({'loss': f'{loss:.4f}'})

        return epoch_loss / max(n_batches, 1)

    def _train_batch_standard(self, sample_indices) -> Optional[float]:
        """Standard mode: direct split-learning training step."""
        # --- send batch assignment ---
        node_state = self.model.get_node_state()
        self.comm.broadcast_batch_assignment(node_state, sample_indices)

        # --- collect forward results ---
        forward_results = self.comm.collect_forward_results()
        merged_data = DataMerger.merge(forward_results, self.device)

        if merged_data is None:
            self.comm.broadcast_backward_signal([None] * len(forward_results))
            self.comm.collect_gradient_results()
            return None

        merged_act, merged_mask, merged_labels, split_sizes, valid_nodes = merged_data

        # --- forward from cut ---
        self.model.train()
        merged_act_input = merged_act.requires_grad_(True)
        logits = self.model.forward_from_cut(merged_act_input, merged_mask)
        loss = self.criterion(logits, merged_labels)

        # --- backward ---
        self.optimizer.zero_grad()
        loss.backward()
        cut_gradients = merged_act_input.grad

        if cut_gradients is None:
            self.comm.broadcast_backward_signal([None] * len(valid_nodes))
            self.comm.collect_gradient_results()
            return None

        # --- send gradients to nodes ---
        gradient_splits = DataMerger.split_gradients(cut_gradients, split_sizes, valid_nodes)
        self.comm.broadcast_backward_signal(gradient_splits)

        # --- collect & aggregate node gradients ---
        gradient_results = self.comm.collect_gradient_results()
        node_param_names = self.model.get_node_param_names()
        aggregated = GradientAggregator.aggregate_gradients(
            gradient_results, node_param_names, valid_nodes
        )

        for name, param in self.model.named_parameters():
            if name in aggregated:
                if param.grad is None:
                    param.grad = aggregated[name].to(self.device)
                else:
                    param.grad += aggregated[name].to(self.device)

        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def _train_batch_secure(self, sample_indices) -> Optional[float]:
        merged_data = self.secure_phases.phase1_collect_shares(sample_indices)
        if merged_data is None:
            return None

        merged_share_0, merged_attn, merged_labels, split_sizes, valid_nodes = merged_data

        result = self.secure_phases.phase2_forward_computation(merged_share_0, merged_attn)
        if result[0] is None:
            self.secure_phases._send_empty_signals(len(valid_nodes))
            return None
        outputs_share_0, outputs_share_1, merged_share_0_input = result

        result = self.secure_phases.phase3_loss_and_backward(
            outputs_share_0, outputs_share_1, merged_share_0_input, merged_labels
        )
        if result[0] is None:
            self.secure_phases._send_empty_signals(len(valid_nodes))
            return None
        cut_gradient, loss_value = result

        success = self.secure_phases.phase4_distribute_gradients(
            cut_gradient, split_sizes, valid_nodes
        )
        return loss_value if success else None

    # ---- Evaluation ----

    def evaluate(self) -> Tuple[float, float, Dict]:
        if self.secure_mode:
            return self.evaluator.evaluate(
                self.model, self.test_loader, self.criterion, self.device
            )
        return self.evaluator.evaluate(self.model, self.test_loader, self.device)

    # ---- Cleanup ----

    def _notify_training_complete(self):
        try:
            empty = [[] for _ in self.comm.node_handlers]
            self.comm.broadcast_batch_assignment(None, empty)
            time.sleep(NOTIFICATION_SLEEP_TIME)
            logging.info("✓ Nodes notified")
        except Exception as e:
            logging.debug(f"Could not notify nodes: {e}")

    def cleanup(self):
        logging.info("")
        logging.info("Closing connections …")
        self.comm.close()
        logging.info("✓ Cleanup complete")


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="TL++ Orchestrator — BioGPT + LoRA on PubMedQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Network
    parser.add_argument('--host', type=str, default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--secure', action='store_true')
    parser.add_argument('--helper_host', type=str, default=DEFAULT_HOST)
    parser.add_argument('--helper_port', type=int, default=DEFAULT_HELPER_PORT)

    # Secure noise
    parser.add_argument('--activation_noise', type=float, default=0.02)
    parser.add_argument('--gradient_noise', type=float, default=0.10)

    # Model
    parser.add_argument('--cut_layer', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--n_nodes', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--use_lora', action='store_true',
                        help='Apply LoRA to orchestrator-side BioGPT layers')
    parser.add_argument('--lora_r', type=int, default=DEFAULT_LORA_R)
    parser.add_argument('--lora_alpha', type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument('--lora_dropout', type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None,
                        help='BioGPT sub-module names to apply LoRA to')

    # Training
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH)

    # Optimisation
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument('--grad_clip_norm', type=float, default=DEFAULT_GRAD_CLIP_NORM)

    # Hardware / paths
    parser.add_argument('--no_accel', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./data')

    args = parser.parse_args()
    if not 1 <= args.n_nodes <= 9:
        parser.error('n_nodes must be between 1 and 9 for PubMedQA (3 classes × up to 3 nodes each)')
    return vars(args)


def main():
    config = parse_args()

    prefix = 'TL++_biogpt_secure' if config.get('secure') else 'TL++_biogpt_base'
    if config.get('use_lora'):
        prefix += '_lora'
    log_file = setup_logging(prefix=prefix)
    logging.info("")

    if config.get('secure'):
        SecretSharing.configure_noise_scaling(
            activation_noise=config.get('activation_noise'),
            gradient_noise=config.get('gradient_noise'),
        )

    logging.info("-" * 80)
    logging.info("CONFIGURATION")
    logging.info("-" * 80)
    logging.info(f"Dataset:            PubMedQA (yes / no / maybe)")
    logging.info(f"Model:              BioGPT (microsoft/biogpt)")
    logging.info(f"Secure mode:        {config.get('secure')}")
    logging.info(f"LoRA:               {config.get('use_lora')}")
    if config.get('use_lora'):
        logging.info(f"  r={config['lora_r']}, alpha={config['lora_alpha']}, "
                     f"dropout={config['lora_dropout']}")
    logging.info(f"Nodes:              {config.get('n_nodes')}")
    logging.info(f"Cut layer:          {config.get('cut_layer')}")
    logging.info(f"Batch size:         {config.get('train_batch_size')}")
    logging.info(f"Learning rate:      {config.get('lr'):.2e}")
    logging.info(f"Max seq length:     {config.get('max_length')}")
    logging.info("")

    orchestrator = SecureOrchestrator(config)
    try:
        orchestrator.setup()
        orchestrator.train()
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.info("Training interrupted by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error(f"Training failed: {e}")
        logging.error("=" * 80)
        import traceback
        logging.error(traceback.format_exc())
    finally:
        orchestrator.cleanup()
        logging.info(f"\nFull log: {log_file}")
        logging.info("=" * 80)


if __name__ == '__main__':
    main()
