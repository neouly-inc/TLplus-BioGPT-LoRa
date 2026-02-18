"""BioGPT Fine-tuning on PubMedQA with LoRA support.

This script fine-tunes the pre-trained BioGPT model on the PubMedQA dataset
for biomedical question answering. Supports both full fine-tuning and LoRA
(Low-Rank Adaptation) for parameter-efficient training.

Aligned with TL++ for fair comparison:
  - 3-class linear classifier head (not causal LM generation)
  - Mean-pool over non-padding positions
  - CrossEntropyLoss directly on logits
  - Custom LoRALinear (same implementation as TL++ models.py)
  - LoRA applied to all 6 transformer layers, embeddings fully trainable
  - Early stopping (patience configurable, default 5)
"""

import sys
import json
import argparse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from transformers import BioGptForCausalLM, BioGptTokenizer

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    from transformers.optimization import get_linear_schedule_with_warmup

import logging


# ==============================================================================
# CONSTANTS
# ==============================================================================

# BioGPT model
MODEL_NAME = 'BioGPT (microsoft/biogpt)'
MODEL_PATH = 'microsoft/biogpt'

# Answer mapping for PubMedQA
ANSWER_MAP = {'yes': 0, 'no': 1, 'maybe': 2}
INVERSE_ANSWER_MAP = {0: 'yes', 1: 'no', 2: 'maybe'}
TARGET_NAMES = ['yes', 'no', 'maybe']
NUM_CLASSES = 3

# Default hyperparameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 500
DEFAULT_MAX_LENGTH = 512
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_PATIENCE = 10

# LoRA hyperparameters
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

# Dataset URLs
PUBMEDQA_BASE_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"
PUBMEDQA_FILES = {
    'ori_pqal.json': 'Labeled subset (1k samples)',
    'ori_pqaa.json': 'Artificial subset (61k samples)',
}


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

KST = timezone(timedelta(hours=9))


class KSTFormatter(logging.Formatter):
    """Custom log formatter using Korean Standard Time."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=KST)
        return dt.strftime(datefmt) if datefmt else dt.isoformat()


def setup_logging(log_dir: str = 'logs', use_lora: bool = False) -> Path:
    """Configure logging to file and console.

    Args:
        log_dir: Directory for log files

    Returns:
        Path to created log file
    """
    Path(log_dir).mkdir(exist_ok=True)

    timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
    prefix = 'centralized_biogpt_lora' if use_lora else 'centralized_biogpt'
    log_file = Path(log_dir) / f'{prefix}_{timestamp}.log'

    formatter = KSTFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S KST'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("=" * 80)
    logging.info("BioGPT Fine-tuning on PubMedQA")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")

    return log_file


# ==============================================================================
# DATA DOWNLOADING
# ==============================================================================

def download_pubmedqa(data_dir: str = 'data') -> bool:
    """Download PubMedQA dataset from GitHub.

    Args:
        data_dir: Directory to save downloaded files

    Returns:
        True if at least one file was successfully downloaded
    """
    logging.info("-" * 80)
    logging.info("DATASET DOWNLOAD")
    logging.info("-" * 80)

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    downloaded = False

    for filename, description in PUBMEDQA_FILES.items():
        filepath = data_path / filename

        if filepath.exists():
            logging.info(f"✓ {filename} already exists")
            downloaded = True
            continue

        try:
            logging.info(f"Downloading {filename} ({description})...")
            url = PUBMEDQA_BASE_URL + filename
            urllib.request.urlretrieve(url, filepath)

            with open(filepath, 'r') as f:
                data = json.load(f)
                logging.info(f"✓ Downloaded {filename}: {len(data):,} samples")
                downloaded = True

        except Exception as e:
            logging.warning(f"✗ Failed to download {filename}: {e}")

    if not downloaded:
        logging.error("Could not download any dataset files")
        logging.error("Please download manually from: https://github.com/pubmedqa/pubmedqa")

    logging.info("")
    return downloaded


# ==============================================================================
# DATA LOADING
# ==============================================================================

@lru_cache(maxsize=1)
def load_pubmedqa_data(data_dir: str = 'data') -> Tuple[Optional[List], Optional[List]]:
    """Load PubMedQA dataset from multiple sources (cached).

    Attempts to load in the following order:
    1. Hugging Face datasets library
    2. Local JSON files
    3. Download from GitHub

    Args:
        data_dir: Directory containing data files

    Returns:
        (train_data, test_data) tuple, or (None, None) if loading fails
    """
    logging.info("-" * 80)
    logging.info("DATA LOADING")
    logging.info("-" * 80)

    # Try Hugging Face datasets first
    try:
        from datasets import load_dataset
        logging.info("Attempting to load from Hugging Face...")

        dataset = load_dataset("pubmed_qa", "pqa_labeled")
        available_splits = list(dataset.keys())
        logging.info(f"Available splits: {available_splits}")

        split_name = 'train' if 'train' in available_splits else available_splits[0]
        if split_name != 'train':
            logging.info(f"Using split: {split_name}")

        all_data = [dict(item) for item in dataset[split_name]]

        # Split into train/test (70/30)
        split_idx = int(len(all_data) * 0.7)
        train_data, test_data = all_data[:split_idx], all_data[split_idx:]

        logging.info(f"✓ Loaded from Hugging Face")
        logging.info(f"  Total samples:   {len(all_data):,}")
        logging.info(f"  Training:        {len(train_data):,}")
        logging.info(f"  Test:            {len(test_data):,}")
        logging.info("")

        return train_data, test_data

    except Exception as e:
        logging.info(f"Could not load from Hugging Face: {e}")

    # Try local files
    logging.info("Attempting to load from local files...")

    local_paths = [
        Path(data_dir) / 'ori_pqal.json',
        Path('ori_pqal.json'),
        Path('../data/ori_pqal.json'),
        Path('./pubmedqa/data/ori_pqal.json')
    ]

    for path in local_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                logging.info(f"✓ Found local file: {path}")

                items = [
                    {
                        'QUESTION': item.get('QUESTION', ''),
                        'CONTEXTS': item.get('CONTEXTS', []),
                        'LONG_ANSWER': item.get('LONG_ANSWER', ''),
                        'final_decision': item.get('final_decision', 'maybe')
                    }
                    for item in data.values()
                ]

                # Split into train/test (80/20)
                split_idx = int(len(items) * 0.8)
                train_data, test_data = items[:split_idx], items[split_idx:]

                logging.info(f"  Total samples:   {len(items):,}")
                logging.info(f"  Training:        {len(train_data):,}")
                logging.info(f"  Test:            {len(test_data):,}")
                logging.info("")

                return train_data, test_data

            except Exception as e:
                logging.warning(f"Error loading from {path}: {e}")

    # If no local file found, try to download
    logging.info("No local files found. Attempting to download...")

    if download_pubmedqa(data_dir):
        load_pubmedqa_data.cache_clear()
        return load_pubmedqa_data(data_dir)

    logging.error("Failed to load PubMedQA dataset")
    logging.error("Please try:")
    logging.error("  1. Install datasets: pip install datasets")
    logging.error("  2. Or download manually from: https://github.com/pubmedqa/pubmedqa")
    logging.error("")

    return None, None


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class PubMedQADataset(Dataset):
    """PubMedQA dataset for BioGPT classification fine-tuning.

    Formats data as: "Question: {Q}\\nContext: {C}\\nAnswer:"
    (no answer token appended — the classifier head predicts the class directly)
    """

    def __init__(self,
                 data: List[Dict],
                 tokenizer,
                 max_length: int = DEFAULT_MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._preprocess_data()

    def _preprocess_data(self):
        """Pre-process all data items for faster access."""
        self.processed_items = []

        for item in self.data:
            question = item.get('QUESTION', item.get('question', ''))

            context = (
                item.get('LONG_ANSWER') if item.get('LONG_ANSWER')
                else ' '.join(item.get('CONTEXTS', []))
                if item.get('CONTEXTS')
                else item.get('context', item.get('long_answer', ''))
            )

            answer = item.get('final_decision', item.get('answer', 'maybe'))

            # Prompt ends at "Answer:" — model does not generate; classifier predicts
            text = f"Question: {question}\nContext: {context}\nAnswer:"

            self.processed_items.append({
                'text': text,
                'answer': ANSWER_MAP.get(str(answer).strip().lower(), 2)
            })

    def __len__(self) -> int:
        return len(self.processed_items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return (input_ids, attention_mask, answer_label)."""
        item = self.processed_items[idx]

        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'answer':         item['answer'],
        }


# ==============================================================================
# CUSTOM LoRA LINEAR  (identical to TL++ models.py)
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
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        ) if bias else None

        nn.init.kaiming_uniform_(self.weight)

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
        """Create a LoRALinear sharing the frozen base weights of *linear*."""
        has_bias = linear.bias is not None
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r, alpha=alpha, dropout=dropout, bias=has_bias,
        )
        lora.weight = linear.weight          # share frozen base weight
        if has_bias:
            lora.bias_param = linear.bias    # share frozen bias
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight,
                        self.bias_param if self.bias_param is not None else None)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scale


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
            parts = attr_name.split('.')
            try:
                parent = layer
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                module = getattr(parent, parts[-1])
                if isinstance(module, nn.Linear):
                    setattr(parent, parts[-1],
                            LoRALinear.from_linear(module, r, alpha, dropout))
            except AttributeError:
                pass


# ==============================================================================
# BIOGPT CLASSIFIER  (aligned with TL++ BioGPTOrchestratorModel)
# ==============================================================================

class BioGPTClassifier(nn.Module):
    """BioGPT with a mean-pool + linear classification head.

    Architecture identical to TL++'s orchestrator model (full-model variant):
      - BioGptModel (embed_tokens + embed_positions + 6 transformer layers +
        layer_norm) — all parameters fully trainable by default
      - Optional LoRA on all 6 transformer layers (same custom LoRALinear as
        TL++); embeddings and layer_norm remain fully trainable
      - Masked mean-pooling over non-padding positions → [B, H]
      - nn.Linear(H, num_classes) classification head

    This replaces the original causal-LM generation approach so that the
    loss function, output head, and evaluation are identical to TL++.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        use_lora: bool = False,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: float = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()

        logging.info(f"Loading {MODEL_PATH} …")
        full = BioGptForCausalLM.from_pretrained(MODEL_PATH)
        self.biogpt = full.biogpt          # BioGptModel
        hidden_size = full.config.hidden_size
        del full

        # Classification head (freshly initialised, same as TL++)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.use_lora = use_lora

        if use_lora:
            targets = lora_target_modules or DEFAULT_LORA_TARGET_MODULES
            # Apply to ALL 6 transformer layers (centralized has no cut point)
            _apply_lora_to_layers(
                self.biogpt.layers, targets, lora_r, lora_alpha, lora_dropout
            )
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in self.parameters())
            logging.info(
                f"LoRA applied to all 6 layers: "
                f"trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
            )
        else:
            total = sum(p.numel() for p in self.parameters())
            logging.info(f"Full fine-tuning: {total:,} parameters")

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """Returns logits [B, num_classes]."""
        outputs = self.biogpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state          # [B, L, H]
        pooled = self._pool(hidden, attention_mask) # [B, H]
        return self.classifier(pooled)

    def _pool(
        self,
        hidden: torch.Tensor,          # [B, L, H]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """Masked mean-pooling over non-padding positions (same as TL++)."""
        mask    = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        summed  = (hidden * mask).sum(dim=1)                    # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1e-9)               # [B, 1]
        return summed / lengths


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model_and_tokenizer(
    device: torch.device,
    use_lora: bool = False,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: List[str] = None,
) -> Tuple[Optional[nn.Module], Optional[object]]:
    """Load BioGPTClassifier and tokenizer.

    Args:
        device: Computation device
        use_lora: Whether to apply LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout probability
        lora_target_modules: List of module names to apply LoRA to

    Returns:
        (model, tokenizer) tuple, or (None, None) if loading fails
    """
    logging.info("-" * 80)
    logging.info("MODEL LOADING")
    logging.info("-" * 80)

    try:
        tokenizer = BioGptTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = BioGPTClassifier(
            num_classes=NUM_CLASSES,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=float(lora_alpha),
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        ).to(device)

        logging.info(f"✓ BioGPTClassifier ready on {device}")
        logging.info("")
        return model, tokenizer

    except Exception as e:
        logging.error(f"✗ Error loading model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None


# ==============================================================================
# TRAINING
# ==============================================================================

criterion = nn.CrossEntropyLoss()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip_norm: float,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch.

    No gradient accumulation — optimizer steps every batch, identical to TL++.
    Uses CrossEntropyLoss on classifier logits (same as TL++).

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}',
                ncols=100, leave=False)

    for batch in pbar:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['answer'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm
            )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, Dict]:
    """Evaluate model on test set using direct classification.

    No text generation — takes argmax of logits, same as TL++ ModelEvaluator.

    Returns:
        (average_loss, accuracy_percentage, info_dict)
    """
    model.eval()

    total_loss     = 0.0
    all_preds      = []
    all_labels     = []
    n_batches      = 0

    pbar = tqdm(dataloader, desc='Evaluating', ncols=100, leave=False)

    for batch in pbar:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['answer'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1

        preds = logits.argmax(dim=-1)
        all_preds .extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds) * 100.0

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
        'unique_predictions': len(set(all_preds)),
        'total_samples':      len(all_labels),
        'correct':            sum(p == l for p, l in zip(all_preds, all_labels)),
        'report':             report,
    }

    return avg_loss, accuracy, info


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    grad_clip_norm: float,
    patience: int = DEFAULT_PATIENCE,
) -> float:
    """Main training loop with early stopping.

    Early-stopping logic mirrors TL++ orchestrator:
      - track best test accuracy
      - increment patience counter when no improvement
      - stop when patience_counter >= patience

    Args:
        patience: Epochs without improvement before stopping (default 5)

    Returns:
        Best test accuracy (%) achieved
    """
    logging.info("-" * 80)
    logging.info("TRAINING")
    logging.info("-" * 80)

    best_acc         = 0.0
    best_epoch       = 0
    patience_counter = 0
    final_acc        = 0.0

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']

        logging.info("")
        logging.info(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6e}")

        # ---- Train ----
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, grad_clip_norm, epoch, epochs
        )

        # ---- Evaluate ----
        test_loss, test_acc, info = evaluate(model, test_loader, device)
        final_acc = test_acc

        logging.info(
            f"  Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

        # Per-class F1
        report = info.get('report', {})
        if report:
            logging.info("  Per-class F1 scores:")
            for cls in TARGET_NAMES:
                if cls in report:
                    f1 = report[cls].get('f1-score', 0) * 100
                    logging.info(f"    {cls.capitalize():6s}: {f1:5.2f}%")

        # Warn on model collapse (same check as TL++)
        if info['unique_predictions'] == 1:
            logging.warning("  ⚠️  Model predicting only 1 class — possible collapse")

        # ---- Best model ----
        if test_acc > best_acc:
            best_acc   = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            logging.info(f"  🏆 New best accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            logging.info(f"  No improvement ({patience_counter}/{patience})")

        # ---- Early stopping ----
        if patience_counter >= patience:
            logging.info("")
            logging.info(f"Early stopping after {patience} epochs without improvement")
            break

    logging.info("")
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Best accuracy:  {best_acc:.2f}% (Epoch {best_epoch})")
    logging.info(f"Final accuracy: {final_acc:.2f}%")
    logging.info("=" * 80)

    return best_acc


# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================

def get_device(no_accel: bool) -> torch.device:
    if no_accel:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    num_workers = 2 if device.type == 'cuda' else 0
    pin_memory  = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, test_loader


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BioGPT Classification Fine-tuning on PubMedQA (TL++-aligned)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                        help='Early stopping patience (epochs without improvement)')

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_WARMUP_STEPS)

    # Regularisation
    parser.add_argument('--grad_clip_norm', type=float, default=DEFAULT_GRAD_CLIP_NORM)

    # LoRA
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=DEFAULT_LORA_R)
    parser.add_argument('--lora_alpha', type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument('--lora_dropout', type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None,
                        help='Target modules for LoRA (e.g. q_proj v_proj). '
                             'Default: q_proj v_proj k_proj out_proj fc1 fc2')

    # Data
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument('--data_dir', type=str, default='./data')

    # Hardware / paths
    parser.add_argument('--no_accel', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./logs')

    return vars(parser.parse_args())


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point."""
    config = parse_args()

    log_file = setup_logging(config['log_dir'], config['use_lora'])
    logging.info("")

    device = get_device(config['no_accel'])
    logging.info(f"Device: {device}")
    if device.type == 'cpu':
        logging.warning("⚠️  Training on CPU will be very slow!")
    logging.info("")

    # Log configuration
    logging.info("-" * 80)
    logging.info("CONFIGURATION")
    logging.info("-" * 80)
    logging.info(f"Model:              {MODEL_NAME}")
    logging.info(f"Training mode:      {'LoRA' if config['use_lora'] else 'Full fine-tuning'}")
    logging.info(f"Epochs:             {config['epochs']}")
    logging.info(f"Patience:           {config['patience']}")
    logging.info(f"Batch size:         {config['batch_size']}")
    logging.info(f"Learning rate:      {config['learning_rate']:.6f}")
    logging.info(f"Max length:         {config['max_length']}")
    logging.info(f"Gradient clip:      {config['grad_clip_norm']}")

    if config['use_lora']:
        targets = config['lora_target_modules'] or DEFAULT_LORA_TARGET_MODULES
        logging.info(f"LoRA rank:          {config['lora_r']}")
        logging.info(f"LoRA alpha:         {config['lora_alpha']}")
        logging.info(f"LoRA dropout:       {config['lora_dropout']}")
        logging.info(f"LoRA targets:       {', '.join(targets)}")
    logging.info("")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        device=device,
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        lora_target_modules=config['lora_target_modules'],
    )

    if model is None or tokenizer is None:
        logging.error("Failed to load model. Exiting.")
        return

    # Load data
    train_data, test_data = load_pubmedqa_data(config['data_dir'])

    if train_data is None or test_data is None or not train_data or not test_data:
        logging.error("Failed to load data. Exiting.")
        return

    # Create datasets and loaders
    logging.info("-" * 80)
    logging.info("DATASET PREPARATION")
    logging.info("-" * 80)

    train_dataset = PubMedQADataset(train_data, tokenizer, config['max_length'])
    test_dataset  = PubMedQADataset(test_data,  tokenizer, config['max_length'])

    logging.info(f"Training samples:   {len(train_dataset):,}")
    logging.info(f"Test samples:       {len(test_dataset):,}")
    logging.info("")

    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, config['batch_size'], device
    )

    logging.info(f"Training batches:   {len(train_loader)}")
    logging.info(f"Test batches:       {len(test_loader)}")
    logging.info("")

    # Optimizer & scheduler
    logging.info("-" * 80)
    logging.info("OPTIMIZER & SCHEDULER")
    logging.info("-" * 80)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8,
    )

    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps,
    )

    logging.info(f"Optimizer:          AdamW (lr={config['learning_rate']:.6f})")
    logging.info(f"Scheduler:          Linear warmup ({config['warmup_steps']} steps / {total_steps} total)")
    logging.info(f"Warmup steps:       {config['warmup_steps']}")
    logging.info(f"Total steps:        {total_steps}")
    logging.info("")

    # Train
    try:
        best_accuracy = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=config['epochs'],
            grad_clip_norm=config['grad_clip_norm'],
            patience=config['patience'],
        )

        logging.info("")
        logging.info(f"Best test accuracy: {best_accuracy:.2f}%")

    except KeyboardInterrupt:
        logging.info("")
        logging.info("=" * 80)
        logging.info("Training interrupted by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error("")
        logging.error("=" * 80)
        logging.error(f"Training failed: {e}")
        logging.error("=" * 80)
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("")
        logging.info(f"Full log: {log_file}")
        logging.info("=" * 80)


if __name__ == '__main__':
    main()