"""Node for TL++ on PubMedQA with BioGPT."""

import argparse
import torch
import logging
from datetime import datetime, timezone, timedelta
from typing import Tuple

from protocol import NodeCommunicator, SecureNodeCommunicator, SecretSharing
from data import NodeDataLoader
from utils import NodeGradientComputer


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8080
DEFAULT_HELPER_PORT = 8081


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

KST = timezone(timedelta(hours=9))


class KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=KST)
        return dt.strftime(datefmt) if datefmt else dt.isoformat()


def setup_logging():
    formatter = KSTFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S KST',
    )
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)


# ==============================================================================
# MAIN NODE CLASS
# ==============================================================================


class SecureNode:
    """Training node for TL++ on PubMedQA.

    Responsibilities:
    - Load class-specific PubMedQA training data (yes / no / maybe)
    - Execute forward pass through node-side BioGPT layers
    - Compute gradients for node-side parameters
    - Communicate with orchestrator (and helper in secure mode)
    """

    def __init__(self, config: dict):
        self.config = config
        self.secure_mode = config.get('secure', False)
        self._setup_device()

        logging.info(f"Device: {self.device}")
        logging.info(f"Secure mode: {'✓ ENABLED' if self.secure_mode else 'DISABLED'}")

        self._setup_communication()

        self.node_id = None
        self.model = None
        self.dataset = None
        self.param_names = None

    # ------------------------------------------------------------------
    # DEVICE & COMMUNICATION SETUP
    # ------------------------------------------------------------------

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
            self.comm = SecureNodeCommunicator(
                orch_host=self.config.get('orch_host', '127.0.0.1'),
                orch_port=self.config.get('orch_port', DEFAULT_PORT),
                helper_host=self.config.get('helper_host', '127.0.0.1'),
                helper_port=self.config.get('helper_port', DEFAULT_HELPER_PORT),
            )
        else:
            self.comm = NodeCommunicator(
                host=self.config.get('host', '127.0.0.1'),
                port=self.config.get('port', DEFAULT_PORT),
            )

    # ------------------------------------------------------------------
    # SETUP PHASE
    # ------------------------------------------------------------------

    def setup(self):
        """Connect, receive configuration, load data, initialise model."""
        self.comm.connect()
        if self.secure_mode:
            logging.info("✓ Connected to orchestrator and helper")
        else:
            logging.info(
                f"✓ Connected to orchestrator at "
                f"{self.config.get('host')}:{self.config.get('port')}"
            )

        # Receive init config (node_id + node_model)
        init_config = self.comm.receive_init()
        self.node_id = init_config['node_id']
        logging.info(f"✓ Assigned node ID: {self.node_id}")

        if self.secure_mode:
            act_noise, grad_noise = SecretSharing.get_noise_scaling()
            logging.info(f"✓ Noise config (activation: {act_noise:.1%}, gradient: {grad_noise:.1%})")

        # Load BioGPT node-side model
        node_model = init_config.get('node_model')
        if node_model is None:
            raise ValueError("Model not provided in initialisation")
        self.model = node_model.to(self.device)

        param_names, _ = self.model.get_state_names()
        self.param_names = param_names

        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"✓ Model parameters: {total_params:,}")

        # Load PubMedQA class-specific dataset
        n_nodes_in_class = init_config.get('n_nodes_in_class', 1)
        node_rank_in_class = init_config.get('node_rank_in_class', 0)

        self.dataset = NodeDataLoader(
            class_id=self.node_id,
            augment=False,
            data_dir=self.config.get('data_dir', './data'),
            max_length=init_config.get('max_length', 512),
            n_nodes_in_class=n_nodes_in_class,
            node_rank_in_class=node_rank_in_class,
        )
        logging.info(f"✓ Loaded {len(self.dataset)} PubMedQA training samples")

        self.comm.send_dataset_size(len(self.dataset))
        logging.info("✓ Ready for training")
        logging.info("")

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------

    def run(self):
        batch_count = 0
        while True:
            if not self.process_batch():
                break
            batch_count += 1
            if batch_count % 50 == 0:
                logging.info(f"Processed {batch_count} batches")
        logging.info(f"✓ Training complete.  Total batches: {batch_count}")

    def process_batch(self) -> bool:
        """Process one training batch.

        Returns True to continue, False when training is done.
        """
        try:
            # ---- Receive batch assignment ----
            assignment = self.comm.receive_batch_assignment()
            model_state = assignment['model_state']
            sample_indices = assignment['sample_indices']

            if model_state is None:
                logging.info("Training complete — received shutdown signal")
                return False

            # ---- Update node-side model weights ----
            device_state = {k: v.to(self.device) for k, v in model_state.items()}
            self.model.load_state_dict(device_state, strict=False)
            self.model.train()

            # ---- Handle empty batch ----
            if len(sample_indices) == 0:
                self._process_empty_batch()
                return True

            # ---- Load text batch ----
            input_ids, attention_mask, labels = self._load_batch(sample_indices)

            # ---- Forward pass through node-side BioGPT layers ----
            # activations: [B, L, H],  attn_4d: [B,1,L,L] (not transmitted)
            hidden_states, _attn_4d = self.model(input_ids, attention_mask)

            # ---- Send forward result ----
            if self.secure_mode:
                self.comm.send_forward_result_secure(
                    hidden_states, attention_mask, labels
                )
            else:
                self.comm.send_forward_result(
                    activations=hidden_states.detach().cpu(),
                    attention_mask=attention_mask.cpu(),
                    labels=labels.cpu(),
                )

            # ---- Receive cut-point gradients ----
            if self.secure_mode:
                cut_gradients = self.comm.receive_backward_signal_secure()
            else:
                cut_gradients = self.comm.receive_backward_signal()

            if cut_gradients is None:
                self._send_empty_gradients()
                return True

            # ---- Backpropagate through node-side layers ----
            cut_gradients = cut_gradients.to(self.device)
            node_gradients = NodeGradientComputer.compute_gradients(
                model=self.model,
                activations=hidden_states,
                cut_gradients=cut_gradients,
                param_names=self.param_names,
            )

            # ---- Send parameter gradients ----
            if self.secure_mode:
                self.comm.send_gradient_result_secure(node_gradients)
            else:
                self.comm.send_gradient_result(node_gradients)

            return True

        except ConnectionError as e:
            if "closed by peer" in str(e).lower():
                logging.info("Training complete — connection closed")
            else:
                logging.error(f"Connection error: {e}")
            return False
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            return False
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _process_empty_batch(self):
        if self.secure_mode:
            self.comm.send_forward_result_secure(None, None, None)
            self.comm.receive_backward_signal_secure()
            self.comm.send_gradient_result_secure({})
        else:
            self.comm.send_forward_result(None, None, None)
            self.comm.receive_backward_signal()
            self.comm.send_gradient_result({})

    def _send_empty_gradients(self):
        if self.secure_mode:
            self.comm.send_gradient_result_secure({})
        else:
            self.comm.send_gradient_result({})

    def _load_batch(
        self, indices
    ) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """Load a batch of PubMedQA samples from the local dataset.

        Returns (input_ids, attention_mask, labels).
        """
        ids_list, masks_list, labels_list = [], [], []
        for idx in indices:
            inp_ids, attn_mask, label = self.dataset[int(idx)]
            ids_list.append(inp_ids)
            masks_list.append(attn_mask)
            labels_list.append(label)

        input_ids = torch.stack(ids_list).to(self.device)
        attention_mask = torch.stack(masks_list).to(self.device)
        labels = torch.tensor(labels_list, dtype=torch.long).to(self.device)
        return input_ids, attention_mask, labels

    def cleanup(self):
        if self.comm:
            self.comm.close()
        logging.info("✓ Cleanup complete")


# We need the Tuple import at the top of this module
from typing import Tuple  # noqa: E402  (placed here to avoid circular-import issues)


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="TL++ Training Node (BioGPT / PubMedQA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--host', type=str, default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--secure', action='store_true')
    parser.add_argument('--orch_host', type=str, default=DEFAULT_HOST)
    parser.add_argument('--orch_port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--helper_host', type=str, default=DEFAULT_HOST)
    parser.add_argument('--helper_port', type=int, default=DEFAULT_HELPER_PORT)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--no_accel', action='store_true')
    return vars(parser.parse_args())


def main():
    config = parse_args()
    setup_logging()

    logging.info("=" * 80)
    logging.info("TL++ Training Node  [BioGPT / PubMedQA]")
    logging.info("=" * 80)
    logging.info("")

    node = SecureNode(config)
    try:
        node.setup()
        node.run()
    except KeyboardInterrupt:
        logging.info("")
        logging.info("=" * 80)
        logging.info("Node interrupted by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error(f"Node failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.cleanup()


if __name__ == '__main__':
    main()