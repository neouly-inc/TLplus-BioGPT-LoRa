"""Training utilities for TL++ on PubMedQA with BioGPT."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, classification_report


# Answer labels (for reporting)
TARGET_NAMES = ['yes', 'no', 'maybe']


# ==============================================================================
# BATCH SCHEDULING  (unchanged from original TL++)
# ==============================================================================


class BatchScheduler:
    """Manages virtual batch creation for distributed training."""

    def __init__(
        self,
        samples_per_node: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        if not samples_per_node:
            raise ValueError("samples_per_node cannot be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.samples_per_node = samples_per_node
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_nodes = len(samples_per_node)
        self.total_samples = sum(samples_per_node)

        if seed is not None:
            np.random.seed(seed)

        self.global_samples = [
            (node_id, local_idx)
            for node_id in range(self.n_nodes)
            for local_idx in range(samples_per_node[node_id])
        ]
        self.n_batches = (self.total_samples + batch_size - 1) // batch_size

    def create_epoch_batches(self) -> List[List[np.ndarray]]:
        if self.shuffle:
            np.random.shuffle(self.global_samples)

        batches = []
        for batch_idx in range(self.n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.total_samples)
            batch_samples = self.global_samples[start:end]

            indices_per_node = [[] for _ in range(self.n_nodes)]
            for node_id, local_idx in batch_samples:
                indices_per_node[node_id].append(local_idx)

            indices_per_node = [
                np.array(idxs, dtype=np.int64) for idxs in indices_per_node
            ]
            batches.append(indices_per_node)

        return batches

    def __len__(self) -> int:
        return self.n_batches

    def __repr__(self) -> str:
        return (
            f"BatchScheduler(n_nodes={self.n_nodes}, "
            f"batch_size={self.batch_size}, n_batches={self.n_batches})"
        )


# ==============================================================================
# GRADIENT AGGREGATION  (unchanged from original TL++)
# ==============================================================================


class GradientAggregator:
    """Aggregates gradients from distributed nodes via averaging."""

    @staticmethod
    def aggregate_gradients(
        gradient_dicts: List[Dict[str, torch.Tensor]],
        param_names: List[str],
        valid_nodes: List[bool],
    ) -> Dict[str, torch.Tensor]:
        if len(gradient_dicts) != len(valid_nodes):
            raise ValueError(
                f"Mismatch: {len(gradient_dicts)} grad dicts "
                f"but {len(valid_nodes)} valid_nodes flags"
            )

        collections: Dict[str, List[torch.Tensor]] = {n: [] for n in param_names}

        for grads, is_valid in zip(gradient_dicts, valid_nodes):
            if not is_valid or not grads:
                continue
            for name in param_names:
                if name in grads:
                    collections[name].append(grads[name])

        return {
            name: torch.stack(tensors).mean(dim=0)
            for name, tensors in collections.items()
            if tensors
        }


# ==============================================================================
# NODE-SIDE GRADIENT COMPUTATION  (unchanged from original TL++)
# ==============================================================================


class NodeGradientComputer:
    """Computes parameter gradients on the node side via autograd."""

    @staticmethod
    def compute_gradients(
        model: nn.Module,
        activations: torch.Tensor,
        cut_gradients: torch.Tensor,
        param_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        params = [p for n, p in model.named_parameters() if n in param_names]
        if not params:
            return {}

        grads = torch.autograd.grad(
            outputs=activations,
            inputs=params,
            grad_outputs=cut_gradients,
            retain_graph=False,
            only_inputs=True,
        )

        result: Dict[str, torch.Tensor] = {}
        grad_idx = 0
        for name, _ in model.named_parameters():
            if name in param_names:
                result[name] = grads[grad_idx].detach().cpu()
                grad_idx += 1

        return result


# ==============================================================================
# DATA MERGING — STANDARD MODE
# Adapted for PubMedQA: forward results now include 'attention_mask'
# ==============================================================================


class DataMerger:
    """Merges forward results (activations + attention_mask + labels) from nodes."""

    @staticmethod
    def merge(
        forward_results: List[Dict],
        device: torch.device,
    ) -> Optional[Tuple]:
        """Merge forward results from all nodes.

        Each result dict contains:
            'activations'    : [B, L, H] hidden states at cut point
            'attention_mask' : [B, L] 2-D padding mask
            'labels'         : [B] integer class labels

        Returns
        -------
        (merged_activations, merged_attn_mask, merged_labels, split_sizes, valid_nodes)
        or None if all batches are empty.
        """
        act_list = []
        mask_list = []
        label_list = []
        split_sizes = []
        valid_nodes = []

        for result in forward_results:
            if result.get('activations') is not None:
                act_list.append(result['activations'])
                mask_list.append(result['attention_mask'])
                label_list.append(result['labels'])
                split_sizes.append(result['activations'].shape[0])
                valid_nodes.append(True)
            else:
                valid_nodes.append(False)

        if not act_list:
            return None

        merged_act = torch.cat(act_list, dim=0).to(device)
        merged_mask = torch.cat(mask_list, dim=0).to(device)
        merged_labels = torch.cat(label_list, dim=0).to(device)

        return merged_act, merged_mask, merged_labels, split_sizes, valid_nodes

    @staticmethod
    def split_gradients(
        gradients: torch.Tensor,
        split_sizes: List[int],
        valid_nodes: List[bool],
    ) -> List[Optional[torch.Tensor]]:
        """Split merged cut-point gradients back to per-node tensors."""
        grad_list = torch.split(gradients, split_sizes, dim=0)
        result = []
        grad_idx = 0
        for is_valid in valid_nodes:
            if is_valid:
                result.append(grad_list[grad_idx].detach().cpu())
                grad_idx += 1
            else:
                result.append(None)
        return result


# ==============================================================================
# DATA MERGING — SECURE MODE
# ==============================================================================


class SecureDataMerger:
    """Merges secret-shared activations + attention_masks from nodes."""

    @staticmethod
    def merge_shares(
        forward_results: List[Dict],
        device: torch.device,
    ) -> Optional[Tuple]:
        """Merge activation shares from multiple nodes.

        Each result dict contains:
            'activations'    : share_0 of hidden states [B, L, H]
            'attention_mask' : [B, L] (sent plaintext, not secret-shared)
            'labels'         : [B] (sent plaintext)

        Returns
        -------
        (merged_shares, merged_attn_mask, merged_labels, split_sizes, valid_nodes)
        or None if all batches empty.
        """
        share_list = []
        mask_list = []
        label_list = []
        split_sizes = []
        valid_nodes = []

        for result in forward_results:
            if result.get('activations') is not None:
                share_list.append(result['activations'])
                mask_list.append(result['attention_mask'])
                label_list.append(result['labels'])
                split_sizes.append(result['activations'].shape[0])
                valid_nodes.append(True)
            else:
                valid_nodes.append(False)

        if not share_list:
            return None

        merged_shares = torch.cat(share_list, dim=0).to(device)
        merged_mask = torch.cat(mask_list, dim=0).to(device)
        merged_labels = torch.cat(label_list, dim=0).to(device)

        return merged_shares, merged_mask, merged_labels, split_sizes, valid_nodes

    @staticmethod
    def split_gradient_shares(
        gradient_shares: torch.Tensor,
        split_sizes: List[int],
        valid_nodes: List[bool],
    ) -> List[Optional[torch.Tensor]]:
        """Split merged gradient shares back to per-node format."""
        grad_list = torch.split(gradient_shares, split_sizes, dim=0)
        result = []
        grad_idx = 0
        for is_valid in valid_nodes:
            if is_valid:
                result.append(grad_list[grad_idx].detach().cpu())
                grad_idx += 1
            else:
                result.append(None)
        return result


# ==============================================================================
# MODEL EVALUATION  (adapted for PubMedQA yes/no/maybe classification)
# ==============================================================================


class ModelEvaluator:
    """Evaluates BioGPT orchestrator on the PubMedQA test set (standard mode)."""

    def __init__(self, criterion: nn.Module):
        self.criterion = criterion

    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
    ) -> Tuple[float, float, Dict]:
        """Evaluate model on test set.

        Returns
        -------
        (average_loss, accuracy_percentage, info_dict)
        """
        model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model.forward_full(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(len(dataloader), 1)
        accuracy = 100.0 * accuracy_score(all_labels, all_preds)
        unique_preds = len(set(all_preds))

        try:
            report = classification_report(
                all_labels, all_preds,
                target_names=TARGET_NAMES,
                output_dict=True,
                zero_division=0,
            )
        except Exception:
            report = {}

        info = {
            'unique_predictions': unique_preds,
            'total_samples': len(all_labels),
            'correct': int(accuracy_score(all_labels, all_preds, normalize=False)),
            'classification_report': report,
        }

        return avg_loss, accuracy, info


class SecureEvaluator:
    """Evaluates BioGPT orchestrator on PubMedQA test set (secure mode).

    Evaluation runs non-secretly (all test data is at orchestrator; no
    privacy concern) for exact accuracy measurement.
    """

    @staticmethod
    def evaluate(
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float, Dict]:
        """Evaluate model on test set (same logic as standard evaluator)."""
        return ModelEvaluator(criterion).evaluate(model, dataloader, device)