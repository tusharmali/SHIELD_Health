
import os

# Dataset Configuration
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "PAMAP2_Dataset")
SEQUENCE_LENGTH = 10
FEATURES_PER_TIMESTEP = 6  # Derived from 52 features / 10 = 5.2 -> 6 with padding
NUM_CLASSES = 13  # PAMAP2 classes (including null class)

# Debug Configuration
DEBUG_MODE = False  # Set to True for fast verification (1% data, 1 round)

# Federated Learning Configuration
NUM_CLIENTS = 10
NUM_ROUNDS = 15  # Set to <= 3 for DEBUG, 3+ for RESEARCH experiments, 15 for full paper results
AGGREGATION_METHOD = "median"  # Options: fed_avg, trimmed_mean, median, krum
QUANTIZATION_BITS = 8
USE_ENCRYPTION = True
USE_TEMPORAL_MODEL = True
BYZANTINE_TOLERANCE = 0.2
EARLY_STOPPING_PATIENCE = 3
STABILITY_THRESHOLD = 0.05

# Privacy Configuration
PRIVACY_EPSILON = 3.0
PRIVACY_DELTA = 1e-5

# Resource Aware Configuration
DEVICE_CATEGORIES = {
    'high': {'count_ratio': 0.3},
    'medium': {'count_ratio': 0.3},
    'low': {'count_ratio': 0.4}
}
