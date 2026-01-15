"""
SHIELD-Health Research Experiments Module

This module implements all comparative experiments for research validation:
1. IID vs Non-IID data distribution comparison
2. Variance tracking across multiple runs
3. Ablation study (FedAvg baseline vs SHIELD-Health)

All experiments ported from original bc_fed_iot_healthcare.py
"""

import numpy as np
import tensorflow as tf
import random
import time
from sklearn.model_selection import train_test_split
from .config import *
from .blockchain import BlockchainFL
from .models import build_medical_iot_model, ResourceAwareModelSelector
from .federated import federated_aggregation, quantize_weights, dequantize_weights, SimplifiedHomomorphicEncryption, CustomDPOptimizer  
from .data_loader import prepare_client_data
from .utils import generate_key_pair, sign_update

def compare_iid_vs_noniid_experiment(X_train, y_train, X_test, y_test, num_classes, input_shape):
    """
    Run experiments comparing IID vs non-IID data distribution.
    
    Returns:
        Dict with IID and Non-IID experiment results
    """
    print("\n==== IID vs non-IID Data Distribution Experiment ====")
    
    results = {
        "iid": {"accuracy": [], "loss": []},
        "non_iid": {"accuracy": [], "loss": []}
    }
    
    for distribution_type in ["iid", "non_iid"]:
        print(f"\nRunning experiment with {distribution_type.upper()} data distribution")
        
        # Prepare client data
        iid_setting = (distribution_type == "iid")
        client_data = prepare_client_data(X_train, y_train, NUM_CLIENTS, iid=iid_setting)
        
        # Initialize blockchain
        blockchain = BlockchainFL()
        model_selector = ResourceAwareModelSelector()
        
        # Initialize global model
        global_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
        global_weights = global_model.get_weights()
        
        # Training loop
        for round_num in range(NUM_ROUNDS):
            round_weights = []
            
            for client_id in range(NUM_CLIENTS):
                local_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
                local_model.set_weights(global_weights)
                
                c_data = client_data[client_id]
                local_model.fit(c_data['X'], c_data['y'], epochs=3, batch_size=32, verbose=0)
                
                round_weights.append(local_model.get_weights())
            
            # Aggregate
            global_weights = federated_aggregation(round_weights, method=AGGREGATION_METHOD)
            global_model.set_weights(global_weights)
            
            # Evaluate
            loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
            results[distribution_type]["accuracy"].append(acc)
            results[distribution_type]["loss"].append(loss)
            
            print(f"  Round {round_num+1}: Acc={acc:.4f}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results["iid"]["accuracy"], label="IID", marker='o')
    plt.plot(results["non_iid"]["accuracy"], label="Non-IID (SHIELD)", marker='s')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("IID vs Non-IID Convergence")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results["iid"]["loss"], label="IID", marker='o')
    plt.plot(results["non_iid"]["loss"], label="Non-IID (SHIELD)", marker='s')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("research_plots_iid_non-iid/iid_vs_noniid_comparison.png")
    print("\nIID vs Non-IID comparison plot saved.")
    
    return results


def run_variance_tracking(
    client_data, num_clients, num_rounds, input_shape, num_classes,
    X_test, y_test, num_runs=3
):
    """
    Run FL multiple times with different seeds to analyze variance.
    
    Returns:
        variance_results: Dict with mean ± std for metrics
        best_model: Best performing model
    """
    print(f"\n===== Running Variance Tracking ({num_runs} runs) =====")
    
    all_runs_metrics = {
        "accuracy": [],
        "loss": [],
        "final_acc": []
    }
    
    best_accuracy = 0
    best_model = None
    
    for run in range(num_runs):
        print(f"\n----- Run {run+1}/{num_runs} -----")
        
        # Set different seed
        seed = run + 42
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        
        # Fresh blockchain + model
        blockchain = BlockchainFL()
        global_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
        global_weights = global_model.get_weights()
        
        # Train
        for round_num in range(num_rounds):
            round_weights = []
            
            for client_id in range(num_clients):
                local_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
                local_model.set_weights(global_weights)
                
                c_data = client_data[client_id]
                local_model.fit(c_data['X'], c_data['y'], epochs=3, batch_size=32, verbose=0)
                
                round_weights.append(local_model.get_weights())
            
            global_weights = federated_aggregation(round_weights, method=AGGREGATION_METHOD)
            global_model.set_weights(global_weights)
        
        # Evaluate
        loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
        all_runs_metrics["final_acc"].append(acc)
        print(f"  Run {run+1} Final Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = global_model
    
    # Calculate variance
    variance_results = {
        "mean_accuracy": np.mean(all_runs_metrics["final_acc"]),
        "std_accuracy": np.std(all_runs_metrics["final_acc"]),
        "max_accuracy": np.max(all_runs_metrics["final_acc"]),
        "min_accuracy": np.min(all_runs_metrics["final_acc"])
    }
    
    print(f"\nVariance Analysis:")
    print(f"  Mean Accuracy: {variance_results['mean_accuracy']:.4f} ± {variance_results['std_accuracy']:.4f}")
    print(f"  Range: [{variance_results['min_accuracy']:.4f}, {variance_results['max_accuracy']:.4f}]")
    
    return variance_results, best_model


def run_ablation_study(
    client_data, num_clients, num_rounds, input_shape, num_classes,
    X_test, y_test
):
    """
    Compare FedAvg baseline vs SHIELD-Health (with blockchain & robust aggregation).
    
    Returns:
        ablation_results: Dict comparing both methods
    """
    print("\n===== Running Ablation Study =====")
    
    ablation_results = {
        "fed_avg": {"accuracy": [], "loss": []},
        "shield": {"accuracy": [], "loss": []}
    }
    
    # 1. Standard FedAvg (baseline)
    print("\n----- Standard FedAvg (Baseline) -----")
    global_model_fedavg = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
    global_weights_fedavg = global_model_fedavg.get_weights()
    
    for round_num in range(num_rounds):
        round_weights = []
        
        for client_id in range(num_clients):
            local_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
            local_model.set_weights(global_weights_fedavg)
            
            c_data = client_data[client_id]
            local_model.fit(c_data['X'], c_data['y'], epochs=3, batch_size=32, verbose=0)
            
            round_weights.append(local_model.get_weights())
        
        # FedAvg aggregation
        global_weights_fedavg = federated_aggregation(round_weights, method="fed_avg")
        global_model_fedavg.set_weights(global_weights_fedavg)
        
        loss, acc = global_model_fedavg.evaluate(X_test, y_test, verbose=0)
        ablation_results["fed_avg"]["accuracy"].append(acc)
        ablation_results["fed_avg"]["loss"].append(loss)
        print(f"  FedAvg Round {round_num+1}: Acc={acc:.4f}")
    
    # 2. SHIELD-Health (proposed)
    print("\n----- SHIELD-Health (Proposed) -----")
    blockchain = BlockchainFL()
    global_model_shield = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
    global_weights_shield = global_model_shield.get_weights()
    
    for round_num in range(num_rounds):
        round_weights = []
        
        for client_id in range(num_clients):
            local_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
            local_model.set_weights(global_weights_shield)
            
            c_data = client_data[client_id]
            local_model.fit(c_data['X'], c_data['y'], epochs=3, batch_size=32, verbose=0)
            
            # Add quantization + DP noise (SHIELD features)
            weights = local_model.get_weights()
            dp_opt = CustomDPOptimizer(noise_multiplier=0.1)
            weights_noisy = dp_opt.add_noise(weights)
            weights_quant, meta = quantize_weights(weights_noisy, bits=QUANTIZATION_BITS)
            weights_restored = dequantize_weights(weights_quant, meta)
            
            round_weights.append(weights_restored)
            
            # Add to blockchain
            blockchain.new_transaction(f"client_{client_id}", "aggregator", "model_update", "hash", None, None)
        
        # Robust aggregation (median)
        global_weights_shield = federated_aggregation(round_weights, method="median")
        global_model_shield.set_weights(global_weights_shield)
        blockchain.new_block(proof=100+round_num)
        
        loss, acc = global_model_shield.evaluate(X_test, y_test, verbose=0)
        ablation_results["shield"]["accuracy"].append(acc)
        ablation_results["shield"]["loss"].append(loss)
        print(f"  SHIELD Round {round_num+1}: Acc={acc:.4f}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ablation_results["fed_avg"]["accuracy"], label="FedAvg (Baseline)", marker='o')
    plt.plot(ablation_results["shield"]["accuracy"], label="SHIELD-Health", marker='s')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Ablation Study: Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(ablation_results["fed_avg"]["loss"], label="FedAvg (Baseline)", marker='o')
    plt.plot(ablation_results["shield"]["loss"], label="SHIELD-Health", marker='s')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Ablation Study: Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("research_results_ablation/ablation_study.png")
    print("\nAblation study plot saved.")
    
    return ablation_results
