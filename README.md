# SHIELD-Health: Blockchain-Enabled Federated Learning for IoT Healthcare

This repository contains the official reference implementation of **SHIELD-Health**, a secure and privacy-preserving framework for Federated Learning (FL) in Internet of Medical Things (IoMT) environments.

## 📋 Abstract

SHIELD-Health introduces a novel architecture that secures IoMT data sharing through:

1.  **Blockchain-based Orchestration**: Replaces the central aggregator with a secure, immutable ledger for model updates.
2.  **Resource-Aware Federated Learning**: Adapts model complexity and training schedules to the heterogeneous capabilities of IoT devices (High/Medium/Low tiers).
3.  **Temporal Attention Mechanism**: A custom deep learning layer designed to capture long-term dependencies in physiological sensor data (e.g., PPG, Accelerometer).
4.  **Privacy Preservation**: Integrates Differential Privacy (DP) and Partial Homomorphic Encryption (HE) to protect patient data.

## 🛠️ Repository Structure

The codebase is modularized for clarity and extensibility:

```
SHIELD_Health/
├── src/
│   ├── config.py          # Central configuration (Hyperparameters, Device splits)
│   ├── models.py          # Model architecture & Temporal Attention mechanism
│   ├── blockchain.py      # Adaptive Proof-of-Work & Ledger implementation
│   ├── federated.py       # Aggregation algorithms (Median, Trimmed Mean) & DP/HE logic
│   ├── data_loader.py     # PAMAP2 dataset processing & Non-IID distribution (Dirichlet)
│   ├── experiments.py     # Scripts for IID vs Non-IID and Variance experiments
│   ├── simulations.py     # Network latency & Attack vector simulations
│   └── analysis.py        # plotting & performance metrics generation
├── main.py                # Main entry point for training and evaluation
├── requirements.txt       # Python dependencies
└── README.md              # This documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended: High-performance GPU for accelerated training (Code automatically detects CUDA).

### Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/tusharmali/SHIELD-Health.git
    cd SHIELD-Health
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Dataset Preparation**
    The framework is configured to use the **PAMAP2 Physical Activity Monitoring Dataset**.
    - **Automatic**: The code tries to download/process the dataset on first run.
    - **Manual**: If automatic download fails, place the dataset in `SHIELD_Health/data/PAMAP2_Dataset`.

## ⚙️ Configuration

All experimental parameters are centrally managed in `src/config.py`.

### Execution Modes

**1. Full Experiment**
Run the main script with default settings:

```bash
python main.py
```

_This will execute the training loop, perform blockchain mining, and generate performance plots._

**2. Fast Debug Mode**
Verify the setup works without waiting for full training:

- Open `src/config.py`
- Set `DEBUG_MODE = True`
- Run `python main.py` (Runs 1 round on 1% data)

## 🔍 Implementation Details

### 1. Temporal Attention Mechanism

Implemented in `src/models.py`, this module applies a dot-product attention mechanism to weigh critical time-steps in sensor data streams.

- **Input**: $(Batch, TimeSteps, Features)$
- **Logic**: Calculates attention scores $ \alpha_t $ to highlight significant physiological patterns before classification.

### 2. Adaptive Blockchain

Implemented in `src/blockchain.py`, the system uses a Proof-of-Work (PoW) consensus where difficulty adapts dynamically based on network mining speed and device capabilities.

- **Transaction Types**: Model Updates, Access Policies, Incentive Rewards.

### 3. Secure Aggregation

- **Aggregation**: Robust aggregation (Median) is used to mitigate outlier updates from compromised devices.
- **Privacy**:
  - **Differential Privacy**: Gaussian noise added to gradients before sharing.
  - **Homomorphic Encryption**: Partial HE allows aggregation of encrypted weights.

## 📊 Outputs

Upon completion, the system generates:

1.  **Console Logs**: Detailed per-round metrics (Accuracy, Loss, Block Time).
2.  **Plots**: Visualizations of convergence and energy consumption.
3.  **Research Report**: A text summary of the experiment results.

## ⚠️ System Configuration Note

The code uses a **3/3/4** split for device capabilities (3 High-end, 3 Mid-range, 4 Low-end) to realistically simulate a hospital environment with diverse hardware. This exact split is hardcoded in `main.py` to ensure consistency.

## License

MIT License
