"""Data loaders for TL++ on PubMedQA with BioGPT."""

import json
import logging
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptTokenizer


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Answer class mapping for PubMedQA
ANSWER_MAP = {'yes': 0, 'no': 1, 'maybe': 2}
INVERSE_ANSWER_MAP = {0: 'yes', 1: 'no', 2: 'maybe'}
NUM_PUBMEDQA_CLASSES = 3          # yes / no / maybe

# BioGPT tokenizer
TOKENIZER_NAME = 'microsoft/biogpt'
DEFAULT_MAX_LENGTH = 512

# Dataset download
PUBMEDQA_BASE_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"
PUBMEDQA_FILES = {
    'ori_pqal.json': 'Labeled subset (1 k samples)',
}


# ==============================================================================
# DATASET DOWNLOAD & LOADING HELPERS
# ==============================================================================

def _download_pubmedqa(data_dir: str = './data') -> bool:
    """Download PubMedQA labeled JSON from GitHub if missing."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    for filename in PUBMEDQA_FILES:
        filepath = data_path / filename
        if filepath.exists():
            return True
        try:
            url = PUBMEDQA_BASE_URL + filename
            logging.info(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, filepath)
            with open(filepath, 'r') as f:
                json.load(f)  # validate JSON
            logging.info(f"✓ Downloaded {filename}")
            return True
        except Exception as e:
            logging.warning(f"Download failed: {e}")
    return False


@lru_cache(maxsize=1)
def _load_pubmedqa_raw(data_dir: str = './data') -> Tuple[Optional[List], Optional[List]]:
    """Load PubMedQA, returning (train_list, test_list).  Cached after first call."""

    # 1. Try Hugging Face datasets
    try:
        from datasets import load_dataset
        logging.info("Loading PubMedQA from Hugging Face …")
        ds = load_dataset("pubmed_qa", "pqa_labeled")
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        all_data = [dict(item) for item in ds[split]]
        cut = int(len(all_data) * 0.7)
        logging.info(f"✓ HF PubMedQA: {len(all_data)} total, {cut} train / {len(all_data)-cut} test")
        return all_data[:cut], all_data[cut:]
    except Exception as e:
        logging.info(f"HF load failed ({e}); falling back to local/download …")

    # 2. Try local files or download
    paths = [
        Path(data_dir) / 'ori_pqal.json',
        Path('ori_pqal.json'),
        Path('./data/ori_pqal.json'),
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            items = [
                {
                    'QUESTION': v.get('QUESTION', ''),
                    'CONTEXTS': v.get('CONTEXTS', []),
                    'LONG_ANSWER': v.get('LONG_ANSWER', ''),
                    'final_decision': v.get('final_decision', 'maybe'),
                }
                for v in raw.values()
            ]
            cut = int(len(items) * 0.8)
            logging.info(f"✓ Local PubMedQA: {len(items)} total, {cut} train / {len(items)-cut} test")
            return items[:cut], items[cut:]
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")

    # 3. Auto-download
    if _download_pubmedqa(data_dir):
        _load_pubmedqa_raw.cache_clear()
        return _load_pubmedqa_raw(data_dir)

    logging.error("Could not load PubMedQA. Run: pip install datasets")
    return None, None


def _item_to_text_and_label(item: Dict) -> Tuple[str, int]:
    """Convert a raw PubMedQA dict to (formatted_text, class_label)."""
    question = item.get('QUESTION', item.get('question', ''))
    context = (
        item.get('LONG_ANSWER')
        if item.get('LONG_ANSWER')
        else ' '.join(item.get('CONTEXTS', []))
        if item.get('CONTEXTS')
        else item.get('context', item.get('long_answer', ''))
    )
    answer = item.get('final_decision', item.get('answer', 'maybe'))
    text = f"Question: {question}\nContext: {context}\nAnswer:"
    label = ANSWER_MAP.get(str(answer).strip().lower(), 2)
    return text, label


# ==============================================================================
# NODE DATA LOADER
# ==============================================================================


class NodeDataLoader:
    """Data loader for a node with class-specific PubMedQA training data.

    Each node is assigned one of the three PubMedQA answer classes
    (yes / no / maybe).  Node IDs > 3 cycle back through classes so that
    up to 9 nodes can be used (3 per class, each receiving a different
    partition of that class's training samples).

    Parameters
    ----------
    class_id : int
        Node class ID in range 1–10 (compatible with original TL++ API).
        Maps to PubMedQA class: (class_id - 1) % 3  →  0=yes, 1=no, 2=maybe.
    augment : bool
        Ignored (kept for API compatibility with original CIFAR-10 loader).
    data_dir : str
        Directory for PubMedQA data files.
    max_length : int
        BioGPT tokeniser maximum sequence length.
    n_nodes_in_class : int
        How many nodes share this class (used to partition data fairly).
    node_rank_in_class : int
        This node's rank among nodes sharing the same class (0-indexed).
    """

    def __init__(
        self,
        class_id: int,
        augment: bool = True,          # kept for API compat
        data_dir: str = './data',
        max_length: int = DEFAULT_MAX_LENGTH,
        n_nodes_in_class: int = 1,
        node_rank_in_class: int = 0,
    ):
        # ---- validation ----
        if not isinstance(class_id, int):
            raise TypeError(f"class_id must be int, got {type(class_id).__name__}")
        if not 1 <= class_id <= 10:
            raise ValueError(f"class_id must be 1–10, got {class_id}")

        self.class_id = class_id
        self.pubmedqa_class = (class_id - 1) % NUM_PUBMEDQA_CLASSES  # 0/1/2
        self.augment = augment
        self.data_dir = data_dir
        self.max_length = max_length

        # ---- load tokeniser ----
        self.tokenizer = BioGptTokenizer.from_pretrained(TOKENIZER_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        # ---- load & filter dataset ----
        train_data, _ = _load_pubmedqa_raw(data_dir)
        if train_data is None:
            raise RuntimeError("Failed to load PubMedQA training data")

        class_samples = [
            item for item in train_data
            if ANSWER_MAP.get(
                str(item.get('final_decision', item.get('answer', 'maybe'))).strip().lower(), 2
            ) == self.pubmedqa_class
        ]

        # Partition among nodes sharing this class
        if n_nodes_in_class > 1:
            part_size = max(1, len(class_samples) // n_nodes_in_class)
            start = node_rank_in_class * part_size
            end = start + part_size if node_rank_in_class < n_nodes_in_class - 1 else len(class_samples)
            class_samples = class_samples[start:end]

        if not class_samples:
            raise RuntimeError(
                f"No training samples for PubMedQA class '{INVERSE_ANSWER_MAP[self.pubmedqa_class]}' "
                f"(class_id={class_id})"
            )

        # ---- pre-tokenise for speed ----
        self._input_ids: List[torch.Tensor] = []
        self._attention_masks: List[torch.Tensor] = []
        self._labels: List[int] = []

        for item in class_samples:
            text, label = _item_to_text_and_label(item)
            enc = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            self._input_ids.append(enc['input_ids'].squeeze(0))
            self._attention_masks.append(enc['attention_mask'].squeeze(0))
            self._labels.append(label)

        self.n_samples = len(self._input_ids)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (input_ids, attention_mask, label)."""
        return self._input_ids[idx], self._attention_masks[idx], self._labels[idx]

    def __repr__(self) -> str:
        cls_name = INVERSE_ANSWER_MAP[self.pubmedqa_class]
        return (
            f"NodeDataLoader(class_id={self.class_id}, "
            f"pubmedqa_class='{cls_name}', n_samples={self.n_samples})"
        )


# ==============================================================================
# ORCHESTRATOR DATA LOADER
# ==============================================================================


class PubMedQATestDataset(Dataset):
    """Torch Dataset for the PubMedQA test split."""

    def __init__(self, test_data: List[Dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._input_ids: List[torch.Tensor] = []
        self._attention_masks: List[torch.Tensor] = []
        self._labels: List[int] = []

        for item in test_data:
            text, label = _item_to_text_and_label(item)
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            self._input_ids.append(enc['input_ids'].squeeze(0))
            self._attention_masks.append(enc['attention_mask'].squeeze(0))
            self._labels.append(label)

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self._input_ids[idx],
            'attention_mask': self._attention_masks[idx],
            'label': torch.tensor(self._labels[idx], dtype=torch.long),
        }


class OrchestratorDataLoader:
    """Data loader for the orchestrator, wrapping the PubMedQA test set.

    Parameters
    ----------
    batch_size : int
        Evaluation batch size.
    num_classes : int
        Ignored (always 3 for PubMedQA); kept for API compatibility.
    data_dir : str
        Directory for PubMedQA data files.
    max_length : int
        BioGPT tokeniser maximum sequence length.
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_classes: int = NUM_PUBMEDQA_CLASSES,
        data_dir: str = './data',
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size}")

        self.batch_size = batch_size
        self.num_classes = NUM_PUBMEDQA_CLASSES   # always 3
        self.data_dir = data_dir
        self.max_length = max_length

        # Tokeniser
        tokenizer = BioGptTokenizer.from_pretrained(TOKENIZER_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        # Load test data
        _, test_data = _load_pubmedqa_raw(data_dir)
        if test_data is None:
            raise RuntimeError("Failed to load PubMedQA test data")

        dataset = PubMedQATestDataset(test_data, tokenizer, max_length)
        self.dataset = dataset
        self.n_samples = len(dataset)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,      # tokenised tensors are in-memory; workers not needed
            pin_memory=True,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):
        return iter(self.loader)

    def __repr__(self) -> str:
        return (
            f"OrchestratorDataLoader(batch_size={self.batch_size}, "
            f"n_samples={self.n_samples}, num_classes={self.num_classes})"
        )