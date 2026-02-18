"""BioGPT + LoRA split model for TL++ on PubMedQA.

Architecture overview
---------------------
The full BioGPT model (6 transformer layers) is split at a *cut layer*:

    Node side  : embed_tokens + embed_positions + layers[0 : cut_n]
    Orch side  : layers[cut_n : 6] + layer_norm + mean-pool + classifier

LoRA is applied **only to the orchestrator-side transformer layers** so that
node-side parameters are fully fine-tuned (they are few and sit on the edge
device) while the larger orchestrator-side stack benefits from parameter-
efficient adaptation.

Cut-layer mapping (BioGPT base has 6 transformer layers)
    cut_layer=1  →  node runs 2 layers, orchestrator runs 4 layers
    cut_layer=2  →  node runs 4 layers, orchestrator runs 2 layers
    cut_layer=3  →  node runs 5 layers, orchestrator runs 1 layer
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BioGptForCausalLM, BioGptConfig


# ==============================================================================
# CONSTANTS
# ==============================================================================

BIOGPT_MODEL_NAME = 'microsoft/biogpt'
BIOGPT_NUM_LAYERS = 6        # BioGPT base
NUM_PUBMEDQA_CLASSES = 3     # yes / no / maybe

# How many transformer layers run on the NODE for each cut_layer value
CUT_LAYER_TO_N_NODE_LAYERS: Dict[int, int] = {
    1: 2,
    2: 4,
    3: 5,
}

# Default LoRA hyper-parameters
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGET_MODULES = ['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2']


# ==============================================================================
# CUSTOM LoRA LINEAR
# ==============================================================================


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with Low-Rank Adaptation (LoRA).

    The base weight is frozen.  Only lora_A and lora_B are trainable.

        output = x @ W.T  +  (x @ A.T @ B.T) * (alpha / r)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        # Frozen base weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        ) if bias else None

        nn.init.kaiming_uniform_(self.weight)

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scale = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """Create a LoRALinear that shares the frozen base weights of *linear*."""
        has_bias = linear.bias is not None
        lora = cls(
            linear.in_features, linear.out_features, r, alpha, dropout, bias=has_bias
        )
        # Share (not copy) the frozen weight tensor
        lora.weight = linear.weight
        lora.weight.requires_grad = False
        if has_bias:
            lora.bias_param = linear.bias
            lora.bias_param.requires_grad = False
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias_param)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale
        return base + lora

    def extra_repr(self) -> str:
        r = self.lora_A.size(0)
        return (
            f"in={self.weight.size(1)}, out={self.weight.size(0)}, "
            f"r={r}, scale={self.scale:.3f}"
        )


def _apply_lora_to_layers(
    layers: nn.ModuleList,
    target_module_names: List[str],
    r: int,
    alpha: float,
    dropout: float,
) -> None:
    """Replace target nn.Linear sub-modules inside *layers* with LoRALinear.

    Modifies *layers* in-place.
    """
    for layer in layers:
        for attr_name in target_module_names:
            # Walk nested attributes, e.g. 'self_attn.q_proj'
            parts = attr_name.split('.')
            try:
                parent = layer
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                module = getattr(parent, parts[-1])
                if isinstance(module, nn.Linear):
                    setattr(parent, parts[-1], LoRALinear.from_linear(module, r, alpha, dropout))
            except AttributeError:
                pass  # Module doesn't exist in this layer — skip silently


# ==============================================================================
# ATTENTION MASK PREPARATION
# ==============================================================================


def _prepare_causal_attn_mask(
    attention_mask: torch.Tensor,  # [B, L]
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build combined causal + padding attention mask for BioGPT layers.

    Returns a [B, 1, seq_len, seq_len] mask where -inf blocks attention.
    """
    batch_size = attention_mask.size(0)

    # Causal mask: upper-triangular = -inf
    causal = torch.full(
        (seq_len, seq_len), float('-inf'), dtype=dtype, device=device
    )
    causal = torch.triu(causal, diagonal=1)           # upper-tri only
    causal = causal[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

    # Padding mask: positions with pad_token → -inf
    pad = (1.0 - attention_mask.float()).masked_fill(
        (1.0 - attention_mask.float()).bool(), float('-inf')
    )
    pad = pad[:, None, None, :].expand(batch_size, 1, seq_len, seq_len)

    return causal + pad


# ==============================================================================
# NODE-SIDE MODEL
# ==============================================================================


class _NodeBioGPTInner(nn.Module):
    """Inner module that mirrors BioGptModel naming for state-dict compatibility."""

    def __init__(self, biogpt_model, cut_n_layers: int):
        super().__init__()
        self.embed_tokens = biogpt_model.embed_tokens
        self.embed_positions = biogpt_model.embed_positions
        self.layers = nn.ModuleList(list(biogpt_model.layers)[:cut_n_layers])

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, L]
        attention_mask: torch.Tensor,     # [B, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run embedding + first cut_n transformer layers.

        Returns
        -------
        hidden_states : [B, L, H]
        combined_attn_mask : [B, 1, L, L]  — prepared for remaining layers
        """
        # Embedding
        # NOTE: BioGPTLearnedPositionalEmbedding.forward expects attention_mask
        # (uses its shape to derive seq_len, then generates arange positions).
        # Passing input_ids causes a CUDA device-side assert because some
        # transformers versions process the values, and token IDs (0–42383) far
        # exceed the position-embedding table size (1026 entries).
        positions = self.embed_positions(attention_mask)
        hidden = self.embed_tokens(input_ids) + positions

        # Prepare 4-D mask once (reused across all node-side layers)
        combined = _prepare_causal_attn_mask(
            attention_mask, hidden.size(1), hidden.dtype, hidden.device
        )

        for layer in self.layers:
            layer_out = layer(hidden, attention_mask=combined)
            hidden = layer_out[0]

        return hidden, combined


class BioGPTNodeModel(nn.Module):
    """Node-side portion of BioGPT.

    Runs: embed_tokens + embed_positions + layers[0 : cut_n_layers]

    Parameters
    ----------
    cut_layer : int
        Split point (1, 2, or 3).  See module docstring for layer counts.
    """

    def __init__(self, cut_layer: int = 1):
        super().__init__()

        if not isinstance(cut_layer, int):
            raise TypeError(f"cut_layer must be int, got {type(cut_layer).__name__}")
        if cut_layer not in CUT_LAYER_TO_N_NODE_LAYERS:
            raise ValueError(f"cut_layer must be 1–3, got {cut_layer}")

        self.cut_layer = cut_layer
        self.cut_n_layers = CUT_LAYER_TO_N_NODE_LAYERS[cut_layer]

        # Load pretrained BioGPT to get architecture + weights
        base = BioGptForCausalLM.from_pretrained(BIOGPT_MODEL_NAME)

        # Wrap node-side sub-modules with BioGptModel-compatible naming
        self.biogpt = _NodeBioGPTInner(base.biogpt, self.cut_n_layers)

        del base  # Free memory

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through node-side layers.

        Returns
        -------
        hidden_states       : [B, L, H]
        combined_attn_mask  : [B, 1, L, L]
        """
        return self.biogpt(input_ids, attention_mask)

    def get_state_names(self) -> Tuple[List[str], List[str]]:
        param_names = [n for n, _ in self.named_parameters()]
        buffer_names = [n for n, _ in self.named_buffers()]
        return param_names, buffer_names

    def __repr__(self) -> str:
        return (
            f"BioGPTNodeModel(cut_layer={self.cut_layer}, "
            f"cut_n_layers={self.cut_n_layers})"
        )


# ==============================================================================
# ORCHESTRATOR-SIDE MODEL
# ==============================================================================


class BioGPTOrchestratorModel(nn.Module):
    """Full BioGPT model managed by the orchestrator.

    Responsible for:
    * Holding the complete set of parameters (node-side + orch-side + head)
    * Providing a split forward pass (`forward_from_cut`)
    * Providing a full forward pass (`forward_full`) for evaluation
    * Extracting node-side state for transmission to nodes
    * Optionally applying LoRA to orchestrator-side transformer layers

    Parameters
    ----------
    num_classes : int
        Classification output size (always 3 for PubMedQA: yes/no/maybe).
    cut_layer : int
        Split point (1, 2, or 3).
    use_lora : bool
        Whether to apply LoRA to orchestrator-side transformer layers.
    lora_r, lora_alpha, lora_dropout : LoRA hyper-parameters.
    lora_target_modules : list of module names inside each layer to adapt.
    """

    def __init__(
        self,
        num_classes: int = NUM_PUBMEDQA_CLASSES,
        cut_layer: int = 1,
        use_lora: bool = False,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: float = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        # Validation
        if cut_layer not in CUT_LAYER_TO_N_NODE_LAYERS:
            raise ValueError(f"cut_layer must be 1–3, got {cut_layer}")
        if not 1 <= num_classes <= 10:
            raise ValueError(f"num_classes must be 1–10, got {num_classes}")

        self.cut_layer = cut_layer
        self.cut_n_layers = CUT_LAYER_TO_N_NODE_LAYERS[cut_layer]
        self.num_classes = num_classes
        self.use_lora = use_lora

        # ---- Load full pretrained BioGPT ----
        logging.info(f"Loading {BIOGPT_MODEL_NAME} …")
        full = BioGptForCausalLM.from_pretrained(BIOGPT_MODEL_NAME)
        self.biogpt = full.biogpt     # BioGptModel (contains all layers)
        hidden_size = full.config.hidden_size

        # Remove LM head (we use a classification head instead)
        del full

        # ---- Classification head ----
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # ---- Optional LoRA on orchestrator-side layers ----
        if use_lora:
            targets = lora_target_modules or DEFAULT_LORA_TARGET_MODULES
            orch_layers = nn.ModuleList(
                list(self.biogpt.layers)[self.cut_n_layers:]
            )
            _apply_lora_to_layers(orch_layers, targets, lora_r, lora_alpha, lora_dropout)
            # Write modified layers back
            for i, layer in enumerate(orch_layers):
                self.biogpt.layers[self.cut_n_layers + i] = layer

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            logging.info(
                f"LoRA applied to orch-side layers [{self.cut_n_layers}, {BIOGPT_NUM_LAYERS}): "
                f"trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
            )

    # ------------------------------------------------------------------
    # FORWARD PASSES
    # ------------------------------------------------------------------

    def forward_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Complete forward pass for standalone evaluation.

        Returns logits [B, num_classes].
        """
        seq_len = input_ids.size(1)

        # Pass attention_mask (not input_ids) — see note in _NodeBioGPTInner.forward
        positions = self.biogpt.embed_positions(attention_mask)
        hidden = self.biogpt.embed_tokens(input_ids) + positions

        combined = _prepare_causal_attn_mask(
            attention_mask, seq_len, hidden.dtype, hidden.device
        )

        for layer in self.biogpt.layers:
            hidden = layer(hidden, attention_mask=combined)[0]

        hidden = self.biogpt.layer_norm(hidden)
        pooled = self._pool(hidden, attention_mask)
        return self.classifier(pooled)

    def forward_from_cut(
        self,
        hidden_states: torch.Tensor,       # [B, L, H]
        attention_mask: torch.Tensor,      # [B, L] raw 2-D padding mask
    ) -> torch.Tensor:
        """Forward pass starting from the cut point.

        Used during distributed training when receiving activations from nodes.

        Returns logits [B, num_classes].
        """
        seq_len = hidden_states.size(1)
        combined = _prepare_causal_attn_mask(
            attention_mask, seq_len, hidden_states.dtype, hidden_states.device
        )

        hidden = hidden_states
        for layer in self.biogpt.layers[self.cut_n_layers:]:
            hidden = layer(hidden, attention_mask=combined)[0]

        hidden = self.biogpt.layer_norm(hidden)
        pooled = self._pool(hidden, attention_mask)
        return self.classifier(pooled)

    # ------------------------------------------------------------------
    # NODE-STATE UTILITIES
    # ------------------------------------------------------------------

    def _node_keywords(self) -> List[str]:
        """Return name keywords that identify node-side parameters."""
        keywords = ['biogpt.embed_tokens', 'biogpt.embed_positions']
        for i in range(self.cut_n_layers):
            keywords.append(f'biogpt.layers.{i}.')
        return keywords

    def get_node_state(self) -> Dict[str, torch.Tensor]:
        """Extract the state dict for the node-side layers.

        Only includes parameters that belong to layers executed on the node
        (embedding + first cut_n transformer layers).
        """
        kws = self._node_keywords()
        return {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if any(kw in name for kw in kws)
        }

    def get_node_param_names(self) -> List[str]:
        """List parameter names for node-side layers."""
        kws = self._node_keywords()
        return [
            name for name, _ in self.named_parameters()
            if any(kw in name for kw in kws)
        ]

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _pool(
        self,
        hidden: torch.Tensor,       # [B, L, H]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """Masked mean-pooling over non-pad positions.  Returns [B, H]."""
        mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        summed = (hidden * mask).sum(dim=1)                  # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1e-9)            # [B, 1]
        return summed / lengths

    def __repr__(self) -> str:
        return (
            f"BioGPTOrchestratorModel(cut_layer={self.cut_layer}, "
            f"num_classes={self.num_classes}, use_lora={self.use_lora})"
        )


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def create_node_model(cut_layer: int) -> BioGPTNodeModel:
    """Create a node-side BioGPT model.

    Args:
        cut_layer : int in {1, 2, 3}

    Returns:
        Initialised BioGPTNodeModel with pretrained weights.
    """
    return BioGPTNodeModel(cut_layer=cut_layer)


def create_orchestrator_model(
    num_classes: int = NUM_PUBMEDQA_CLASSES,
    cut_layer: int = 1,
    use_lora: bool = False,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: float = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: Optional[List[str]] = None,
) -> BioGPTOrchestratorModel:
    """Create a full orchestrator-side BioGPT model.

    Args:
        num_classes         : Output classes (3 for PubMedQA).
        cut_layer           : Split point.
        use_lora            : Whether to apply LoRA to orch-side layers.
        lora_r, lora_alpha, lora_dropout : LoRA hyper-parameters.
        lora_target_modules : Specific sub-modules to adapt with LoRA.

    Returns:
        Initialised BioGPTOrchestratorModel.
    """
    return BioGPTOrchestratorModel(
        num_classes=num_classes,
        cut_layer=cut_layer,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )