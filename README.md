# TL++ | BioGPT with LoRA on PubMedQA

This implementation enables collaborative training of the [BioGPT](https://github.com/microsoft/BioGPT) language model on the [PubMedQA](https://pubmedqa.github.io/) biomedical question-answering benchmark across multiple edge devices (nodes), while preserving data privacy. It extends the core [TL++](https://github.com/neouly-inc/TLplus) framework with a **transformer-based split model**, **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning, and the same proven secure multiparty computation (MPC) protocol used in the [CNN model](https://github.com/neouly-inc/TLplus/blob/main/core/models.py).

---

### 🔑 Key Advantages

**Privacy-Preserving by Design**

* ✅ Raw biomedical text never leaves edge devices (nodes)
* ✅ Intermediate hidden-state activations and gradients protected using additive secret sharing
* ✅ Semi-honest security model with non-colluding orchestrator and helper node
* ✅ Configurable privacy-utility trade-off via noise parameters

**Distributed & Efficient**

* ⚡ Splits BioGPT's 6 transformer layers between node and orchestrator at a configurable cut point
* ⚡ LoRA applied exclusively to orchestrator-side layers — dramatically reduces trainable parameters without sacrificing accuracy
* ⚡ Masked mean-pooling + linear classification head for direct 3-class prediction (yes / no / maybe)
* ⚡ Supports heterogeneous devices (GPU / CPU / MPS)

**Flexible Deployment**

* 🔄 Base mode for trusted environments (faster training, no MPC overhead)
* 🔒 Secure mode for privacy-critical clinical / research applications (MPC-based)
* 🎯 Configurable model split points across BioGPT's 6 transformer layers
* 🌐 Single-machine or multi-machine deployment

**Production-Ready**

* 📊 Comprehensive KST-timestamped logging to file and console
* 🛡️ Gradient clipping and early stopping
* 🔧 Linear warmup + linear decay scheduler (AdamW)
* ⚙️ Extensive hyperparameter control via CLI

### Use Cases

* **Clinical NLP**: Collaborative training of biomedical QA models across hospital sites without sharing patient notes
* **Pharmaceutical Research**: Multi-institution drug-literature mining with privacy guarantees
* **Medical Education**: Federated fine-tuning of biomedical language models on institution-specific corpora
* **IoT / Edge Health Devices**: On-device language model training with bandwidth-constrained communication

---

### Architecture Components

**Nodes (Edge Devices)**

* Store and process local private biomedical QA data (one PubMedQA answer class per node: *yes*, *no*, or *maybe*)
* Execute the bottom portion of BioGPT: `embed_tokens + embed_positions + layers[0 : cut_n]`
* Send encrypted hidden-state shares (secure mode) or raw activations (base mode) to the orchestrator
* Compute parameter gradients for local node-side model layers

**Orchestrator (Central Server)**

* Coordinates distributed training across all nodes
* Holds the complete BioGPT model; executes `layers[cut_n : 6] + layer_norm + mean-pool + classifier`
* Applies LoRA to orchestrator-side transformer layers (optional, `--use_lora`)
* Aggregates node gradients and updates the global model parameters via AdamW

**Helper (Secure Mode Only)**

* Independent, non-colluding server for the two-server MPC protocol
* Receives `share_1` of each node's hidden-state activations (plaintext attention masks only)
* Runs the orchestrator-side model on `share_1`; returns output shares and cut-point gradient shares
* Critical for MPC privacy: neither server alone can reconstruct raw activations

---

## 🧠 BioGPT Model

TL++ uses `microsoft/biogpt` ([BioGptForCausalLM](https://huggingface.co/microsoft/BioGPT)), a domain-specific GPT-2-style language model pre-trained on 15M+ PubMed abstracts.

### Full Architecture (BioGPT Base)

* **Embedding**: `embed_tokens` (42,384 vocab × 1,024 hidden) + `embed_positions` (learned, 1,026 positions)
* **Transformer**: 6 decoder layers, each with 16-head self-attention (`d_model=1024`, `d_ff=4096`)
* **Classification Head**: Masked mean-pool over non-padding positions → `Linear(1024 → 3)`
* **Loss**: `CrossEntropyLoss` on logits (3-class: yes / no / maybe)

### Model Split by Cut Layer

| `cut_layer` | Node-side layers | Orchestrator-side layers | Communication tensor shape |
|:-----------:|:---------------:|:------------------------:|:---------------------------:|
| 1 (default) | 2 | 4 | `[B, L, 1024]` |
| 2 | 4 | 2 | `[B, L, 1024]` |
| 3 | 5 | 1 | `[B, L, 1024]` |

### LoRA (Low-Rank Adaptation)

When `--use_lora` is passed, LoRA adapters are applied **only to orchestrator-side transformer layers**, leaving node-side parameters fully fine-tuned. This significantly reduces the number of trainable parameters on the server while preserving model quality.

* **Default target modules**: `q_proj`, `v_proj`, `k_proj`, `out_proj`, `fc1`, `fc2`
* **Default rank / alpha**: `r=8`, `α=16` → scale factor `α/r = 2.0`
* **LoRA formula**: `output = x·Wᵀ + (x·Aᵀ·Bᵀ) × (α/r)`
* Base weights (`W`) are **frozen**; only `A` and `B` matrices are trained

---

## 🚀 Installation

### Prerequisites

* Python 3.8+
* PyTorch 2.0+
* Transformers 4.30+
* scikit-learn
* tqdm
* datasets (optional, for Hugging Face auto-download)

### Setup

```bash
# Clone the repository
git clone https://github.com/neouly-inc/TLplus.git
cd TLplus

# Install dependencies
pip install torch transformers scikit-learn tqdm

# Optional: Hugging Face datasets for automatic PubMedQA download
pip install datasets

# Verify installation
python -c "from transformers import BioGptForCausalLM; print('BioGPT ready')"
```

### GPU Support (Optional)

```bash
# For NVIDIA GPUs (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (MPS) — PyTorch 2.0+ includes MPS support
# No additional installation needed
```

### Dataset

PubMedQA is loaded automatically in order of priority:

1. **Hugging Face** (`pip install datasets`) — downloads `pubmed_qa / pqa_labeled` (1,000 labeled samples); 70/30 train/test split
2. **Local file** — place `ori_pqal.json` in `./data/`; 70/30 train/test split
3. **Auto-download** — fetches from the [PubMedQA GitHub](https://github.com/pubmedqa/pubmedqa) if no local file is found

---

## ⚡ Quick Start

### 1. Base Mode (Single Machine, 3 Nodes)

```bash
# Terminal 1 — Orchestrator
python orchestrator.py --n_nodes 3 --cut_layer 1

# Terminal 2 — Node 1  (yes class)
python node.py

# Terminal 3 — Node 2  (no class)
python node.py

# Terminal 4 — Node 3  (maybe class)
python node.py
```

### 2. Base Mode with LoRA

```bash
# Terminal 1 — Orchestrator with LoRA on orchestrator-side layers
python orchestrator.py --n_nodes 3 --cut_layer 1 --use_lora --lora_r 8 --lora_alpha 16

# Terminals 2-4 — Nodes (unchanged)
python node.py
```

### 3. Secure Mode (Single Machine)

```bash
# Terminal 1 — Orchestrator
python orchestrator.py --n_nodes 3 --cut_layer 1 --secure

# Terminal 2 — Helper
python helper.py --n_nodes 3

# Terminals 3-5 — Nodes
python node.py --secure
python node.py --secure
python node.py --secure
```

### 4. Centralized Baseline

```bash
# Full fine-tuning
python centralized.py --epochs 100 --batch_size 8 --learning_rate 2e-5

# LoRA fine-tuning
python centralized.py --use_lora --lora_r 8 --lora_alpha 16 --epochs 100
```

---

## 📘 Base Mode

Base mode provides direct communication between the orchestrator and nodes without any cryptographic privacy protection. Suitable for trusted environments or ablation studies.

### Orchestrator

```bash
python orchestrator.py \
  --n_nodes 3 \
  --cut_layer 1 \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --epochs 100 \
  --train_batch_size 8 \
  --lr 2e-5 \
  --warmup_steps 500 \
  --patience 10
```

### Nodes

```bash
# Each node auto-connects and is assigned a PubMedQA class (yes/no/maybe)
python node.py --host 127.0.0.1 --port 8080
```

### Distributed Setup (Multiple Machines)

#### On the Server (Orchestrator)

```bash
python orchestrator.py \
  --host 0.0.0.0 \
  --port 8080 \
  --n_nodes 3 \
  --use_lora
```

#### On Edge Devices (Nodes)

```bash
# Replace SERVER_IP with the orchestrator's IP address
python node.py --host SERVER_IP --port 8080
```

### Cut Layer Selection

| Cut Layer | Node runs | Orchestrator runs | Best for |
|:---------:|:---------:|:-----------------:|:---------|
| 1 *(default)* | 2 layers | 4 layers | Bandwidth-constrained edge devices |
| 2 | 4 layers | 2 layers | Balanced computation split |
| 3 | 5 layers | 1 layer | Maximum privacy (most computation on node) |

```bash
python orchestrator.py --cut_layer 2 --n_nodes 3
```

### Node Assignment

Each node is assigned one PubMedQA answer class based on its connection order. With 3 nodes, the mapping is:

| Node ID | PubMedQA Class | Training Data |
|:-------:|:--------------:|:--------------|
| 1 | `yes` | All labeled *yes* samples |
| 2 | `no` | All labeled *no* samples |
| 3 | `maybe` | All labeled *maybe* samples |

With more than 3 nodes (up to 9), classes cycle and data is partitioned evenly among nodes sharing the same class (e.g., nodes 1 and 4 both train on *yes* samples, each receiving a non-overlapping partition).

---

## 🔒 Secure Mode

Secure mode uses two-server multi-party computation (MPC) with additive secret sharing to protect hidden-state activations between the edge devices and the servers.

### Security Model

* **Threat Model**: Semi-honest (honest-but-curious) adversaries
* **Assumption**: Orchestrator and helper server do not collude
* **Protection**: Hidden-state activations and parameter gradients are secret-shared; neither server alone can reconstruct them
* **Attention masks**: Transmitted in plaintext (reveals only padding positions, not content)

### Setup

#### 1. Start Orchestrator

```bash
python orchestrator.py \
  --secure \
  --n_nodes 3 \
  --host 0.0.0.0 \
  --port 8080 \
  --helper_host 0.0.0.0 \
  --helper_port 8082 \
  --use_lora \
  --activation_noise 0.02 \
  --gradient_noise 0.10
```

#### 2. Start Helper Server

```bash
python helper.py \
  --n_nodes 3 \
  --host 0.0.0.0 \
  --port 8081 \
  --orch_host ORCHESTRATOR_IP \
  --orch_port 8082
```

#### 3. Start Nodes

```bash
python node.py \
  --secure \
  --orch_host ORCHESTRATOR_IP \
  --orch_port 8080 \
  --helper_host HELPER_IP \
  --helper_port 8081
```

---

## 📊 Centralized Learning Baseline

`centralized.py` provides a standard centralized fine-tuning baseline for fair comparison against TL++. It fine-tunes the identical BioGPT classification model on the complete PubMedQA labeled dataset using a single machine with full data access — no distribution, no split learning, and no MPC overhead.

The baseline uses the same architectural choices as TL++ (mean-pool classification head, CrossEntropyLoss, AdamW optimizer, linear warmup scheduler, gradient clipping, early stopping) to ensure that any accuracy gap between centralized and distributed results reflects the cost of privacy and distribution rather than differences in training setup.

```bash
# Full fine-tuning baseline
python centralized.py \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --patience 10

# LoRA fine-tuning baseline (all 6 layers)
python centralized.py \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --epochs 100 \
  --batch_size 8
```

---

## ⚙️ Configuration

### Orchestrator Options

```
python orchestrator.py --help
```

**Network:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--host` | `127.0.0.1` | Bind address for node connections |
| `--port` | `8080` | Port for node connections |
| `--secure` | `False` | Enable secure MPC mode |
| `--helper_host` | `127.0.0.1` | Helper server host (secure mode) |
| `--helper_port` | `8082` | Helper coordination port (secure mode) |

**Secure Mode Privacy:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--activation_noise` | `0.02` | Gaussian noise scale for activation shares |
| `--gradient_noise` | `0.10` | Gaussian noise scale for gradient shares |

**Model:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--cut_layer` | `1` | BioGPT split point: 1, 2, or 3 |
| `--n_nodes` | `3` | Number of edge nodes (1–9) |
| `--use_lora` | `False` | Apply LoRA to orchestrator-side transformer layers |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha (scale = alpha / r) |
| `--lora_dropout` | `0.1` | Dropout applied to LoRA input |
| `--lora_target_modules` | all 6 | Sub-module names to adapt (e.g. `q_proj v_proj`) |

**Training:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--epochs` | `100` | Maximum training epochs |
| `--patience` | `10` | Early stopping patience |
| `--train_batch_size` | `8` | Training batch size |
| `--test_batch_size` | `16` | Evaluation batch size |
| `--max_length` | `512` | BioGPT tokenizer maximum sequence length |

**Optimisation:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--lr` | `2e-5` | AdamW learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--warmup_steps` | `500` | Linear warmup steps |
| `--grad_clip_norm` | `1.0` | Gradient clipping norm (0 = disabled) |

**Hardware / Paths:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--no_accel` | `False` | Force CPU (disable CUDA / MPS) |
| `--data_dir` | `./data` | Directory for PubMedQA data files |

---

### Node Options

```
python node.py --help
```

**Base Mode:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--host` | `127.0.0.1` | Orchestrator host |
| `--port` | `8080` | Orchestrator port |

**Secure Mode:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--secure` | `False` | Enable secure MPC mode |
| `--orch_host` | `127.0.0.1` | Orchestrator host |
| `--orch_port` | `8080` | Orchestrator port |
| `--helper_host` | `127.0.0.1` | Helper server host |
| `--helper_port` | `8081` | Helper server port |

**Hardware / Paths:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--no_accel` | `False` | Force CPU |
| `--data_dir` | `./data` | Directory for PubMedQA data files |

---

### Helper Options

```
python helper.py --help
```

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--host` | `127.0.0.1` | Bind address for node connections |
| `--port` | `8081` | Port for node connections |
| `--orch_host` | `127.0.0.1` | Orchestrator host |
| `--orch_port` | `8082` | Orchestrator coordination port |
| `--n_nodes` | `3` | Expected number of training nodes |

---

### Centralized Baseline Options

```
python centralized.py --help
```

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--epochs` | `100` | Maximum training epochs |
| `--batch_size` | `8` | Training / evaluation batch size |
| `--patience` | `10` | Early stopping patience |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--warmup_steps` | `500` | Linear warmup steps |
| `--max_length` | `512` | Tokenizer maximum sequence length |
| `--grad_clip_norm` | `1.0` | Gradient clipping norm |
| `--use_lora` | `False` | Enable LoRA on all 6 transformer layers |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--lora_target_modules` | all 6 | Target sub-modules for LoRA |
| `--data_dir` | `./data` | PubMedQA data directory |
| `--no_accel` | `False` | Force CPU |
| `--log_dir` | `./logs` | Directory for log files |

---

## 📝 License

This project is licensed under the MIT License.
