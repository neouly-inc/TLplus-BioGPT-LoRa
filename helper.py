"""Helper server for TL++ secure mode (BioGPT / PubMedQA)."""

import argparse
import socket
import logging
import torch
from datetime import datetime, timezone, timedelta

from runtime.protocol import SecureSocketCommunicator, SecureMessageType, SecretSharing
from core.models import create_orchestrator_model, NUM_PUBMEDQA_CLASSES


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_HOST = '127.0.0.1'
DEFAULT_NODE_PORT = 8081
DEFAULT_ORCH_PORT = 8082

KST = timezone(timedelta(hours=9))


# ==============================================================================
# LOGGING
# ==============================================================================


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
# HELPER NODE HANDLER
# ==============================================================================


class HelperNodeHandler(SecureSocketCommunicator):
    """Handler for a single training-node connection on the helper side."""

    def __init__(self, sock: socket.socket, address: tuple, node_id: int):
        self.socket = sock
        self.address = address
        self.node_id = node_id
        self.activation_share = None
        self.attention_mask = None  # received plaintext alongside share_1

    def receive_forward_share(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.SHARE_FORWARD
        self.activation_share = payload.get('activations')
        self.attention_mask = payload.get('attention_mask')
        return payload

    def send_backward_share(self, gradient_share) -> None:
        self._send_message(self.socket, SecureMessageType.SHARE_BACKWARD, gradient_share)

    def receive_gradient_share(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.SHARE_GRADIENT
        return payload

    def close(self) -> None:
        self.socket.close()


# ==============================================================================
# HELPER SERVER
# ==============================================================================


class HelperNode:
    """Helper server for two-server MPC.

    Privacy properties
    ------------------
    ✓ Intermediate activations (hidden states): never reconstructed here
    ✓ Attention masks: received plaintext (reveals only padding, not content)
    ✓ Cut-layer gradients: computed on share_1 separately
    ✓ Output shares: sent to orchestrator (reconstructed there only)
    """

    def __init__(self, config: dict):
        self.config = config
        self.node_host = config.get('host', DEFAULT_HOST)
        self.node_port = config.get('port', DEFAULT_NODE_PORT)
        self.orch_host = config.get('orch_host', DEFAULT_HOST)
        self.orch_port = config.get('orch_port', DEFAULT_ORCH_PORT)

        self.model = None
        self.device = torch.device('cpu')   # helper uses CPU

        self.node_server_socket = None
        self.orch_socket = None
        self.node_handlers = []

        self.merged_share_1 = None
        self.merged_attn_mask = None
        self.merged_share_1_input = None
        self.outputs_share_1 = None
        self.batch_count = 0

        logging.info("Helper server initialised")
        logging.info(f"  Node server: {self.node_host}:{self.node_port}")
        logging.info(f"  Orchestrator: {self.orch_host}:{self.orch_port}")
        logging.info(f"  Device: {self.device}")
        logging.info("")

    # ---- Setup ----

    def start_node_server(self) -> None:
        self.node_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.node_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.node_server_socket.bind((self.node_host, self.node_port))
        self.node_server_socket.listen(10)
        logging.info(f"✓ Helper listening on {self.node_host}:{self.node_port}")

    def connect_to_orchestrator(self) -> None:
        self.orch_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.orch_socket.connect((self.orch_host, self.orch_port))

        comm = SecureSocketCommunicator()
        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {'status': 'connected'})

        msg_type, noise_config = comm._receive_message(self.orch_socket)
        if msg_type == SecureMessageType.HELPER_INIT and noise_config.get('phase') == 'noise_config':
            SecretSharing.configure_noise_scaling(
                activation_noise=noise_config['activation_noise'],
                gradient_noise=noise_config['gradient_noise'],
            )
            act, grad = SecretSharing.get_noise_scaling()
            logging.info(f"✓ Noise config received (activation: {act:.1%}, gradient: {grad:.1%})")

        logging.info(f"✓ Connected to orchestrator at {self.orch_host}:{self.orch_port}")

    def accept_node(self) -> HelperNodeHandler:
        sock, addr = self.node_server_socket.accept()
        handler = HelperNodeHandler(sock, addr, len(self.node_handlers))
        self.node_handlers.append(handler)
        return handler

    def wait_for_nodes(self, n_nodes: int) -> None:
        logging.info(f"Waiting for {n_nodes} node(s) …")
        for i in range(n_nodes):
            self.accept_node()
            logging.info(f"  ✓ Node {i+1}/{n_nodes} connected")
        logging.info(f"✓ All {n_nodes} node(s) connected")
        logging.info("")

    # ---- Main loop ----

    def run(self) -> None:
        logging.info("=" * 80)
        logging.info("Helper server ready for secure MPC  [BioGPT / PubMedQA]")
        logging.info("=" * 80)
        logging.info("")
        while True:
            if not self.process_batch():
                break
        logging.info("")
        logging.info(f"✓ Processing complete.  Total batches: {self.batch_count}")

    def process_batch(self) -> bool:
        try:
            comm = SecureSocketCommunicator()
            msg_type, coord_msg = comm._receive_message(self.orch_socket)

            if msg_type == SecureMessageType.SHUTDOWN:
                return False

            phase = coord_msg.get('phase')

            if phase == 'forward':
                return self._phase_forward(comm, coord_msg)
            elif phase == 'compute' and coord_msg.get('action') == 'forward':
                return self._phase_compute_forward(comm)
            elif phase == 'compute' and coord_msg.get('action') == 'backward':
                return self._phase_compute_backward(comm, coord_msg)
            elif phase == 'backward':
                return self._phase_backward(comm, coord_msg)
            else:
                logging.warning(f"Unknown phase: {phase}")
                return True

        except ConnectionError:
            logging.info("Connection closed — training complete")
            return False
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ---- Phases ----

    def _phase_forward(self, comm, coord_msg) -> bool:
        """Receive model + collect share_1 (hidden-state shares) from nodes."""
        model_state = coord_msg.get('model_state')
        cut_layer = coord_msg.get('cut_layer', 1)
        num_classes = coord_msg.get('num_classes', NUM_PUBMEDQA_CLASSES)

        if self.model is None:
            self.model = create_orchestrator_model(
                num_classes=num_classes,
                cut_layer=cut_layer,
            ).to(self.device)

        if model_state is not None:
            self.model.load_state_dict(model_state)
            self.model.eval()

        # Collect share_1 from all nodes
        forward_shares = []
        for handler in self.node_handlers:
            share_data = handler.receive_forward_share()
            forward_shares.append(share_data)

        share_list = []
        attn_list = []
        for result in forward_shares:
            if result.get('activations') is not None:
                share_list.append(result['activations'])
                attn_list.append(result['attention_mask'])

        if not share_list:
            comm._send_message(self.orch_socket, SecureMessageType.SHUTDOWN, {})
            return True

        self.merged_share_1 = torch.cat(share_list, dim=0).to(self.device)
        # Attention masks are plaintext; keep them for use in forward pass
        self.merged_attn_mask = torch.cat(attn_list, dim=0).to(self.device)

        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'forward_shares': forward_shares,
            'status': 'forward_collected',
        })
        return True

    def _phase_compute_forward(self, comm) -> bool:
        """Compute forward pass on share_1 using helper's copy of orch-side model."""
        self.merged_share_1_input = self.merged_share_1.requires_grad_(True)
        # Use the plaintext attention_mask for BioGPT transformer layers
        self.outputs_share_1 = self.model.forward_from_cut(
            self.merged_share_1_input, self.merged_attn_mask
        )

        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'output_share': self.outputs_share_1.detach().cpu(),
            'status': 'forward_computed',
        })
        return True

    def _phase_compute_backward(self, comm, coord_msg) -> bool:
        """Backpropagate through share_1 and return cut-point gradient share."""
        output_gradient = coord_msg.get('output_gradient')

        if (output_gradient is None or
                self.outputs_share_1 is None or
                self.merged_share_1_input is None):
            cut_grad_share_1 = None
        else:
            output_gradient = output_gradient.to(self.device)
            self.outputs_share_1.backward(output_gradient)
            cut_grad_share_1 = self.merged_share_1_input.grad

        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'cut_gradient': cut_grad_share_1.detach().cpu() if cut_grad_share_1 is not None else None,
            'status': 'backward_computed',
        })
        return True

    def _phase_backward(self, comm, coord_msg) -> bool:
        """Distribute gradient shares (share_1) to nodes and collect param grads."""
        gradient_shares = coord_msg.get('gradient_shares', [])

        for handler, grad_share in zip(self.node_handlers, gradient_shares):
            handler.send_backward_share(grad_share)

        node_grad_shares = []
        for handler in self.node_handlers:
            grad_share = handler.receive_gradient_share()
            node_grad_shares.append(grad_share)

        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'gradient_shares': node_grad_shares,
            'status': 'backward_complete',
        })

        self.batch_count += 1
        if self.batch_count % 50 == 0:
            logging.info(f"Processed {self.batch_count} batches")
        return True

    # ---- Cleanup ----

    def cleanup(self) -> None:
        logging.info("")
        logging.info("Closing connections …")
        for handler in self.node_handlers:
            handler.close()
        if self.orch_socket:
            self.orch_socket.close()
        if self.node_server_socket:
            self.node_server_socket.close()
        logging.info("✓ Cleanup complete")


# ==============================================================================
# CLI
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Helper Server for TL++ Secure Mode (BioGPT / PubMedQA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                        help='Host for node connections')
    parser.add_argument('--port', type=int, default=DEFAULT_NODE_PORT,
                        help='Port for node connections')
    parser.add_argument('--orch_host', type=str, default=DEFAULT_HOST,
                        help='Orchestrator host for coordination')
    parser.add_argument('--orch_port', type=int, default=DEFAULT_ORCH_PORT,
                        help='Orchestrator coordination port')
    parser.add_argument('--n_nodes', type=int, default=3, choices=[1, 2, 3],
                        help='Expected number of training nodes')
    return vars(parser.parse_args())


def main():
    config = parse_args()
    setup_logging()

    logging.info("=" * 80)
    logging.info("TL++ Helper Server  [Secure Mode / BioGPT / PubMedQA]")
    logging.info("=" * 80)
    logging.info("")

    helper = HelperNode(config)
    try:
        helper.start_node_server()
        helper.connect_to_orchestrator()
        helper.wait_for_nodes(config['n_nodes'])
        helper.run()
    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.info("Helper interrupted by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error(f"Helper failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        helper.cleanup()


if __name__ == '__main__':
    main()
