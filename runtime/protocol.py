"""Communication protocols for TL++ on PubMedQA with BioGPT.

Changes from the CIFAR-10 version
----------------------------------
* ``send_forward_result`` / ``send_forward_result_secure`` now accept and
  transmit an ``attention_mask`` tensor alongside the hidden-state
  activations.  The orchestrator needs this mask to continue the forward
  pass through the remaining BioGPT transformer layers.
* ``attention_mask`` is always sent **plaintext** (not secret-shared) because
  it only reveals padding positions, not the actual content of the sample.
"""

import socket
import pickle
import struct
import torch
from typing import Any, Tuple, List, Dict, Optional
from enum import IntEnum


# ==============================================================================
# CONSTANTS
# ==============================================================================

HEADER_SIZE = 8
HEADER_FORMAT = '!II'
DEFAULT_SOCKET_TIMEOUT = 30.0


# ==============================================================================
# MESSAGE TYPES  (identical to original TL++)
# ==============================================================================


class MessageType(IntEnum):
    INIT = 1
    DATASET_SIZE = 2
    BATCH_ASSIGNMENT = 3
    FORWARD_RESULT = 4
    BACKWARD_SIGNAL = 5
    GRADIENT_RESULT = 6
    SHUTDOWN = 99


class SecureMessageType(IntEnum):
    INIT = 1
    DATASET_SIZE = 2
    BATCH_ASSIGNMENT = 3
    FORWARD_RESULT = 4
    BACKWARD_SIGNAL = 5
    GRADIENT_RESULT = 6
    SHUTDOWN = 99
    HELPER_INIT = 10
    HELPER_READY = 11
    SHARE_FORWARD = 12
    SHARE_BACKWARD = 13
    SHARE_GRADIENT = 14


# ==============================================================================
# SECRET SHARING  (identical to original TL++)
# ==============================================================================


class SecretSharing:
    """Additive secret sharing for tensor privacy with configurable noise."""

    _activation_noise_scale = 0.02
    _gradient_noise_scale = 0.10

    @classmethod
    def configure_noise_scaling(
        cls,
        activation_noise: Optional[float] = None,
        gradient_noise: Optional[float] = None,
    ) -> None:
        if activation_noise is not None:
            if not 0.0 <= activation_noise <= 1.0:
                raise ValueError(f"activation_noise must be in [0,1], got {activation_noise}")
            cls._activation_noise_scale = activation_noise
        if gradient_noise is not None:
            if not 0.0 <= gradient_noise <= 1.0:
                raise ValueError(f"gradient_noise must be in [0,1], got {gradient_noise}")
            cls._gradient_noise_scale = gradient_noise

    @classmethod
    def get_noise_scaling(cls) -> Tuple[float, float]:
        return cls._activation_noise_scale, cls._gradient_noise_scale

    @staticmethod
    def share_tensor(
        tensor: torch.Tensor,
        noise_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor is None:
            return None, None
        tensor_std = tensor.std().item() if tensor.numel() > 1 else 1.0
        if noise_scale is None:
            noise_scale = 0.0
        actual_noise_std = tensor_std * noise_scale
        share_0 = torch.randn_like(tensor) * actual_noise_std + tensor * 0.5
        share_1 = tensor - share_0
        return share_0, share_1

    @staticmethod
    def reconstruct_tensor(
        share_0: torch.Tensor,
        share_1: torch.Tensor,
    ) -> torch.Tensor:
        if share_0 is None or share_1 is None:
            return None
        return share_0 + share_1

    @staticmethod
    def share_dict(
        tensor_dict: Dict[str, torch.Tensor],
        noise_scale: Optional[float] = None,
    ) -> Tuple[Dict, Dict]:
        if not tensor_dict:
            return {}, {}
        share_0_dict, share_1_dict = {}, {}
        for name, tensor in tensor_dict.items():
            s0, s1 = SecretSharing.share_tensor(tensor, noise_scale)
            share_0_dict[name] = s0
            share_1_dict[name] = s1
        return share_0_dict, share_1_dict

    @staticmethod
    def reconstruct_dict(
        share_0_dict: Dict[str, torch.Tensor],
        share_1_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not share_0_dict or not share_1_dict:
            return {}
        return {
            name: SecretSharing.reconstruct_tensor(share_0_dict[name], share_1_dict[name])
            for name in share_0_dict
        }


# ==============================================================================
# BASE SOCKET COMMUNICATORS
# ==============================================================================


class SocketCommunicator:
    """Base class for socket-based typed messaging (standard mode)."""

    def _send_message(self, sock: socket.socket, msg_type: MessageType, payload: Any) -> None:
        try:
            serialized = pickle.dumps(payload)
            header = struct.pack(HEADER_FORMAT, int(msg_type), len(serialized))
            sock.sendall(header + serialized)
        except (BrokenPipeError, ConnectionResetError) as e:
            raise ConnectionError(f"Failed to send message type {msg_type}: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error sending message: {e}")

    def _receive_message(self, sock: socket.socket) -> Tuple[MessageType, Any]:
        try:
            header = self._receive_exact(sock, HEADER_SIZE)
            msg_type_int, payload_length = struct.unpack(HEADER_FORMAT, header)
            msg_type = MessageType(msg_type_int)
            serialized = self._receive_exact(sock, payload_length)
            return msg_type, pickle.loads(serialized)
        except (ConnectionResetError, struct.error) as e:
            raise ConnectionError(f"Failed to receive message: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error receiving message: {e}")

    def _receive_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by peer")
            data += chunk
        return data


class SecureSocketCommunicator:
    """Base class for socket-based typed messaging (secure mode)."""

    def _send_message(self, sock: socket.socket, msg_type: SecureMessageType, payload: Any) -> None:
        try:
            serialized = pickle.dumps(payload)
            header = struct.pack(HEADER_FORMAT, int(msg_type), len(serialized))
            sock.sendall(header + serialized)
        except (BrokenPipeError, ConnectionResetError) as e:
            raise ConnectionError(f"Failed to send secure message type {msg_type}: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error sending secure message: {e}")

    def _receive_message(self, sock: socket.socket) -> Tuple[SecureMessageType, Any]:
        try:
            header = self._receive_exact(sock, HEADER_SIZE)
            msg_type_int, payload_length = struct.unpack(HEADER_FORMAT, header)
            msg_type = SecureMessageType(msg_type_int)
            serialized = self._receive_exact(sock, payload_length)
            return msg_type, pickle.loads(serialized)
        except (ConnectionResetError, struct.error) as e:
            raise ConnectionError(f"Failed to receive secure message: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error receiving secure message: {e}")

    def _receive_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by peer")
            data += chunk
        return data


# ==============================================================================
# NODE COMMUNICATORS
# ==============================================================================


class NodeCommunicator(SocketCommunicator):
    """Node-side communication handler (standard mode)."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None
        self.node_id = None

    def connect(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def receive_init(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.INIT
        self.node_id = payload['node_id']
        return payload

    def send_dataset_size(self, n_samples: int) -> None:
        self._send_message(self.socket, MessageType.DATASET_SIZE, n_samples)

    def receive_batch_assignment(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.BATCH_ASSIGNMENT
        return payload

    def send_forward_result(
        self,
        activations: Any,
        attention_mask: Any,
        labels: Any,
    ) -> None:
        """Send forward pass results to orchestrator.

        Args:
            activations    : [B, L, H] hidden states at cut point (or None)
            attention_mask : [B, L] 2-D padding mask (or None)
            labels         : [B] integer class labels (or None)
        """
        payload = {
            'activations': activations,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        self._send_message(self.socket, MessageType.FORWARD_RESULT, payload)

    def receive_backward_signal(self) -> Any:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.BACKWARD_SIGNAL
        return payload

    def send_gradient_result(self, gradients: dict) -> None:
        self._send_message(self.socket, MessageType.GRADIENT_RESULT, gradients)

    def close(self) -> None:
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass


class SecureNodeCommunicator(SecureSocketCommunicator):
    """Node-side communication handler (secure mode)."""

    def __init__(
        self,
        orch_host: str,
        orch_port: int,
        helper_host: str,
        helper_port: int,
    ):
        self.orch_host = orch_host
        self.orch_port = orch_port
        self.helper_host = helper_host
        self.helper_port = helper_port
        self.orch_socket = None
        self.helper_socket = None
        self.node_id = None

    def connect(self) -> None:
        self.orch_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.orch_socket.connect((self.orch_host, self.orch_port))
        self.helper_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.helper_socket.connect((self.helper_host, self.helper_port))

    def receive_init(self) -> dict:
        msg_type, payload = self._receive_message(self.orch_socket)
        assert msg_type == SecureMessageType.INIT
        self.node_id = payload['node_id']
        if 'activation_noise' in payload and 'gradient_noise' in payload:
            SecretSharing.configure_noise_scaling(
                activation_noise=payload['activation_noise'],
                gradient_noise=payload['gradient_noise'],
            )
        return payload

    def send_dataset_size(self, n_samples: int) -> None:
        self._send_message(self.orch_socket, SecureMessageType.DATASET_SIZE, n_samples)

    def receive_batch_assignment(self) -> dict:
        msg_type, payload = self._receive_message(self.orch_socket)
        assert msg_type == SecureMessageType.BATCH_ASSIGNMENT
        return payload

    def send_forward_result_secure(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Secret-share activations; send attention_mask + labels plaintext.

        Privacy protocol:
        - activations : split into shares → share_0 to orch, share_1 to helper
        - attention_mask : plaintext to both (reveals only padding positions)
        - labels : plaintext to orchestrator only (needed for loss)
        """
        if activations is None:
            self._send_message(self.orch_socket, SecureMessageType.FORWARD_RESULT, {
                'activations': None, 'attention_mask': None, 'labels': None,
            })
            self._send_message(self.helper_socket, SecureMessageType.SHARE_FORWARD, {
                'activations': None, 'attention_mask': None,
            })
            return

        share_0, share_1 = SecretSharing.share_tensor(
            activations, noise_scale=SecretSharing._activation_noise_scale
        )

        self._send_message(self.orch_socket, SecureMessageType.FORWARD_RESULT, {
            'activations': share_0.detach().cpu(),
            'attention_mask': attention_mask.cpu(),
            'labels': labels.cpu(),
        })
        self._send_message(self.helper_socket, SecureMessageType.SHARE_FORWARD, {
            'activations': share_1.detach().cpu(),
            'attention_mask': attention_mask.cpu(),
            'node_id': self.node_id,
        })

    def receive_backward_signal_secure(self) -> torch.Tensor:
        msg_type, share_0 = self._receive_message(self.orch_socket)
        assert msg_type == SecureMessageType.BACKWARD_SIGNAL
        msg_type, share_1 = self._receive_message(self.helper_socket)
        assert msg_type == SecureMessageType.SHARE_BACKWARD
        if share_0 is None or share_1 is None:
            return None
        return SecretSharing.reconstruct_tensor(share_0, share_1)

    def send_gradient_result_secure(self, gradients: Dict[str, torch.Tensor]) -> None:
        if not gradients:
            self._send_message(self.orch_socket, SecureMessageType.GRADIENT_RESULT, {})
            self._send_message(self.helper_socket, SecureMessageType.SHARE_GRADIENT, {})
            return
        share_0_dict, share_1_dict = SecretSharing.share_dict(
            gradients, noise_scale=SecretSharing._gradient_noise_scale
        )
        self._send_message(self.orch_socket, SecureMessageType.GRADIENT_RESULT, share_0_dict)
        self._send_message(self.helper_socket, SecureMessageType.SHARE_GRADIENT, share_1_dict)

    def close(self) -> None:
        for sock in (self.orch_socket, self.helper_socket):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass


# ==============================================================================
# ORCHESTRATOR COMMUNICATORS
# ==============================================================================


class OrchestratorCommunicator(SocketCommunicator):
    """Orchestrator-side communication manager (standard mode)."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = None
        self.node_handlers = []

    def start(self) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)

    def accept_node(self) -> 'NodeHandler':
        sock, addr = self.server_socket.accept()
        handler = NodeHandler(sock, addr, len(self.node_handlers))
        self.node_handlers.append(handler)
        return handler

    def broadcast_init(self, **kwargs) -> None:
        for handler in self.node_handlers:
            config = {'node_id': handler.node_id, **kwargs}
            handler.send_init(config)

    def collect_dataset_sizes(self) -> List[int]:
        return [h.receive_dataset_size() for h in self.node_handlers]

    def broadcast_batch_assignment(self, model_state, sample_indices_per_node) -> None:
        for handler, indices in zip(self.node_handlers, sample_indices_per_node):
            handler.send_batch_assignment(model_state, indices)

    def collect_forward_results(self) -> List[dict]:
        return [h.receive_forward_result() for h in self.node_handlers]

    def broadcast_backward_signal(self, gradients_per_node) -> None:
        for handler, grads in zip(self.node_handlers, gradients_per_node):
            handler.send_backward_signal(grads)

    def collect_gradient_results(self) -> List[dict]:
        return [h.receive_gradient_result() for h in self.node_handlers]

    def close(self) -> None:
        for h in self.node_handlers:
            h.close()
        if self.server_socket:
            self.server_socket.close()


class SecureOrchestratorCommunicator(SecureSocketCommunicator):
    """Orchestrator-side communication manager (secure mode)."""

    def __init__(self, host: str, port: int, helper_host: str, helper_port: int):
        self.host = host
        self.port = port
        self.helper_host = helper_host
        self.helper_port = helper_port
        self.server_socket = None
        self.helper_server_socket = None
        self.helper_socket = None
        self.node_handlers = []

    def start(self) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)

        self.helper_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.helper_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.helper_server_socket.bind((self.helper_host, self.helper_port))
        self.helper_server_socket.listen(1)

    def wait_for_helper(self) -> None:
        self.helper_socket, _ = self.helper_server_socket.accept()
        msg_type, _ = self._receive_message(self.helper_socket)
        assert msg_type == SecureMessageType.HELPER_READY

    def send_noise_config_to_helper(self, activation_noise: float, gradient_noise: float) -> None:
        self._send_message(self.helper_socket, SecureMessageType.HELPER_INIT, {
            'phase': 'noise_config',
            'activation_noise': activation_noise,
            'gradient_noise': gradient_noise,
        })

    def accept_node(self) -> 'SecureNodeHandler':
        sock, addr = self.server_socket.accept()
        handler = SecureNodeHandler(sock, addr, len(self.node_handlers))
        self.node_handlers.append(handler)
        return handler

    def broadcast_init(self, activation_noise=None, gradient_noise=None, **kwargs) -> None:
        for handler in self.node_handlers:
            config = {'node_id': handler.node_id, **kwargs}
            if activation_noise is not None:
                config['activation_noise'] = activation_noise
                config['gradient_noise'] = gradient_noise
            handler.send_init(config)

    def collect_dataset_sizes(self) -> List[int]:
        return [h.receive_dataset_size() for h in self.node_handlers]

    def broadcast_batch_assignment(self, model_state, sample_indices_per_node) -> None:
        for handler, indices in zip(self.node_handlers, sample_indices_per_node):
            handler.send_batch_assignment(model_state, indices)

    def collect_forward_results(self) -> List[dict]:
        return [h.receive_forward_result() for h in self.node_handlers]

    def broadcast_backward_signal(self, gradients_per_node) -> None:
        for handler, grads in zip(self.node_handlers, gradients_per_node):
            handler.send_backward_signal(grads)

    def collect_gradient_results(self) -> List[dict]:
        return [h.receive_gradient_result() for h in self.node_handlers]

    def close(self) -> None:
        for h in self.node_handlers:
            h.close()
        for sock in (self.helper_socket, self.helper_server_socket, self.server_socket):
            if sock:
                sock.close()


# ==============================================================================
# NODE HANDLERS
# ==============================================================================


class NodeHandler(SocketCommunicator):
    """Handler for one node connection (standard mode)."""

    def __init__(self, sock, address, node_id):
        self.socket = sock
        self.address = address
        self.node_id = node_id + 1  # 1-indexed

    def send_init(self, config: dict) -> None:
        self._send_message(self.socket, MessageType.INIT, config)

    def receive_dataset_size(self) -> int:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.DATASET_SIZE
        return payload

    def send_batch_assignment(self, model_state, sample_indices) -> None:
        self._send_message(self.socket, MessageType.BATCH_ASSIGNMENT, {
            'model_state': model_state,
            'sample_indices': sample_indices,
        })

    def receive_forward_result(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.FORWARD_RESULT
        return payload

    def send_backward_signal(self, gradients) -> None:
        self._send_message(self.socket, MessageType.BACKWARD_SIGNAL, gradients)

    def receive_gradient_result(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.GRADIENT_RESULT
        return payload

    def close(self) -> None:
        self.socket.close()


class SecureNodeHandler(SecureSocketCommunicator):
    """Handler for one node connection (secure mode)."""

    def __init__(self, sock, address, node_id):
        self.socket = sock
        self.address = address
        self.node_id = node_id + 1

    def send_init(self, config: dict) -> None:
        self._send_message(self.socket, SecureMessageType.INIT, config)

    def receive_dataset_size(self) -> int:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.DATASET_SIZE
        return payload

    def send_batch_assignment(self, model_state, sample_indices) -> None:
        self._send_message(self.socket, SecureMessageType.BATCH_ASSIGNMENT, {
            'model_state': model_state,
            'sample_indices': sample_indices,
        })

    def receive_forward_result(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.FORWARD_RESULT
        return payload

    def send_backward_signal(self, gradients) -> None:
        self._send_message(self.socket, SecureMessageType.BACKWARD_SIGNAL, gradients)

    def receive_gradient_result(self) -> dict:
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.GRADIENT_RESULT
        return payload

    def close(self) -> None:
        self.socket.close()