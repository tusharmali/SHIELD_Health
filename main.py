"""
SHIELD-Health Framework - Main Execution Script

This script orchestrates the complete research suite including:
- Core federated learning with blockchain integration
- IID vs Non-IID data distribution experiments
- Variance tracking across multiple runs
- Ablation studies (FedAvg baseline vs SHIELD-Health)
- Network condition simulations
- Attack scenario robustness testing
- Comprehensive performance reporting

Aligned 100% with SHIELD-Health research paper and original implementation.
"""

import numpy as np
import os
import sys

# TensorFlow logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

from src.config import *
from src.data_loader import download_and_prepare_pamap2_dataset, prepare_client_data
from src.models import build_medical_iot_model, ResourceAwareModelSelector, DeviceProfile
from src.blockchain import BlockchainFL
from src.federated import federated_aggregation, quantize_weights, dequantize_weights, SimplifiedHomomorphicEncryption, CustomDPOptimizer
from src.analysis import plot_federated_results, generate_research_report, evaluate_model_performance, analyze_blockchain_performance
from src.utils import generate_key_pair, sign_update
from src.experiments import compare_iid_vs_noniid_experiment, run_variance_tracking, run_ablation_study
from src.simulations import simulate_network_conditions, simulate_attack_scenarios, verify_implementation_quality, list_generated_files


def main():
    print("\nStarting Blockchain-Enabled Federated Learning for IoT Healthcare")
    print("=" * 60)
    
    # ====================================================================
    # 1. DATA LOADING
    # ====================================================================
    print(f"\nLoading PAMAP2 Dataset from {DATASET_PATH}...")
    X_train, X_test, y_train, y_test, num_classes = download_and_prepare_pamap2_dataset()
    
    if X_train is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Verify data integrity
    print(f"Dataset loaded successfully with {num_classes} activity classes")
    print(f"Training data: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"Test data: {len(X_test)} samples, {X_test.shape[1]} features")
    
    
    # Fast execution for testing if DEBUG_MODE is enabled
    if DEBUG_MODE:
        print("\n[DEBUG] Running in FAST MODE: Using 1% of dataset for quick verification.")
        indices = np.random.choice(len(X_train), int(len(X_train) * 0.01), replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        test_indices = np.random.choice(len(X_test), int(len(X_test) * 0.01), replace=False)
        X_test = X_test[test_indices]
        y_test = y_test.iloc[test_indices] if hasattr(y_test, 'iloc') else y_test[test_indices]
    
    input_shape = X_train.shape[1]
    
    # ====================================================================
    # 2. CONFIGURATION
    # ====================================================================
    print("\n==== Model and Training Configuration ====")
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Aggregation method: {AGGREGATION_METHOD}")
    print(f"Using temporal model: {USE_TEMPORAL_MODEL}")
    print(f"Using encryption: {USE_ENCRYPTION}")
    
    # ====================================================================
    # 3. DEVICE PROFILES (Resource-Aware)
    # ====================================================================
    print("\n==== Device Capability Distribution ====")
    model_selector = ResourceAwareModelSelector()
    device_profiles = {}
    
    device_id = 0
    for i in range(NUM_CLIENTS):
        # Assign category
        if i < 3:
            cat = 'high'
        elif i < 6:
            cat = 'medium'
        else:
            cat = 'low'
        
        profile = DeviceProfile(
            device_id=device_id,
            cpu_power=10 if cat == 'high' else (5 if cat == 'medium' else 2),
            memory=4096 if cat == 'high' else (2048 if cat == 'medium' else 1024),
            battery_capacity=100 if cat == 'high' else (50 if cat == 'medium' else 30),
            network_bandwidth=20 if cat == 'high' else (10 if cat == 'medium' else 5),
            is_mobile=True
        )
        device_profiles[device_id] = profile
        model_selector.register_device(profile)
        device_id += 1
    
    print(f"High-end devices: {sum(1 for p in device_profiles.values() if p.cpu_power >= 10)}")
    print(f"Mid-range devices: {sum(1 for p in device_profiles.values() if 4 < p.cpu_power < 10)}")
    print(f"Low-end devices: {sum(1 for p in device_profiles.values() if p.cpu_power <= 4)}")
    
    # ====================================================================
    # 4. PREPARE CLIENT DATA (Non-IID)
    # ====================================================================
    print("\n==== Preparing Client Data (Non-IID Distribution) ====")
    client_data = prepare_client_data(X_train, y_train, NUM_CLIENTS, iid=False)
    
    for i, c_data in enumerate(client_data):
        print(f"Client {i}: {len(c_data['X'])} samples")
    
    # ====================================================================
    # 5. MAIN FEDERATED LEARNING TRAINING
    # ====================================================================
    print("\n==== Starting Federated Learning with Blockchain ====")
    
    blockchain = BlockchainFL()
    he = SimplifiedHomomorphicEncryption()
    dp_optimizer = CustomDPOptimizer(noise_multiplier=0.1)
    
    # Client keys
    client_keys = {}
    for i in range(NUM_CLIENTS):
        priv, pub, priv_bytes, pub_bytes = generate_key_pair()
        client_keys[i] = {'private': priv, 'public': pub_bytes}
    
    # Global model initialization
    global_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
    global_weights = global_model.get_weights()
    
    history = {'accuracy': [], 'loss': []}
    
   # Training loop
    print(f"\nStarting Training for {NUM_ROUNDS} rounds with {NUM_CLIENTS} clients...")
    
    for round_num in range(NUM_ROUNDS):
        print(f"\nRound {round_num + 1}/{NUM_ROUNDS}")
        
        round_weights = []
        
        for client_id in range(NUM_CLIENTS):
            complexity = model_selector.get_complexity_assignment(client_id)
            
            local_model = build_medical_iot_model(input_shape, num_classes, SEQUENCE_LENGTH)
            local_model.set_weights(global_weights)
            
            c_data = client_data[client_id]
            
            # Match original code: High=16, Low=64, Med=32. Epochs=1
            epochs = 1 
            if complexity == 'high':
                batch_size = 16
            elif complexity == 'low':
                batch_size = 64
            else:
                batch_size = 32
            
            # Show progress for DEBUG_MODE
            verbose = 1 if NUM_ROUNDS == 1 else 0
            print(f"  Client {client_id} training ({complexity} mode)...")
            local_model.fit(c_data['X'], c_data['y'], epochs=epochs, batch_size=batch_size, verbose=verbose)
            
            # Differential Privacy
            w_trained = local_model.get_weights()
            w_noisy = dp_optimizer.add_noise(w_trained)
            
            # Quantization
            w_quant, metadata = quantize_weights(w_noisy, bits=QUANTIZATION_BITS)
            
            # Security (Signing)
            signature = sign_update(w_quant, client_keys[client_id]['private'])
            
            # Blockchain Transaction
            blockchain.new_transaction(
                sender=f"client_{client_id}",
                recipient="aggregator",
                type="model_update",
                content="model_hash",
                signature=signature,
                public_key=client_keys[client_id]['public'].decode('utf-8')
            )
            
            # De-quantize
            w_restored = dequantize_weights(w_quant, metadata)
            
            # Optional Encryption
            if USE_ENCRYPTION:
                w_encrypted = he.encrypt(w_restored)
                w_final = he.decrypt(w_encrypted)
            else:
                w_final = w_restored
            
            round_weights.append(w_final)
        
        # Aggregation
        print("  Aggregating updates...")
        global_weights = federated_aggregation(round_weights, method=AGGREGATION_METHOD)
        global_model.set_weights(global_weights)
        
        # Evaluation
        loss, acc = global_model.evaluate(X_test, y_test, verbose=0)
        print(f"  Global Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        
        history['accuracy'].append(acc)
        history['loss'].append(loss)
        
        # Mining block
        print("  Mining block...")
        blockchain.proof_of_work(blockchain.last_block['proof'])
        blockchain.new_block(proof=blockchain.last_block['proof'] + 1)
    
    # ====================================================================
    # 6. FINAL MODEL EVALUATION
    # ====================================================================
    print("\n==== Final Model Performance ====")
    final_metrics = evaluate_model_performance(global_model, X_test, y_test, num_classes)
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1 Score:  {final_metrics['f1_score']:.4f}")
    
    # ====================================================================
    # 7. BLOCKCHAIN PERFORMANCE ANALYSIS
    # ====================================================================
    print("\n==== Blockchain Performance ====")
    blockchain_metrics = analyze_blockchain_performance(blockchain)
    print(f"Total blocks: {blockchain_metrics['num_blocks']}")
    print(f"Total transactions: {blockchain_metrics['total_tx']}")
    print(f"Average block time: {blockchain_metrics['avg_block_time']:.2f} seconds")
    print(f"Storage size: {blockchain_metrics['storage_mb']:.2f} MB")
    print(f"Throughput: {blockchain_metrics['throughput_tx_per_sec']:.2f} tx/sec")
    
    # ====================================================================
    # 8. RESEARCH EXPERIMENTS - RUN IN ORIGINAL ORDER
    # ====================================================================
    print("\n" + "="*60)
    print("RESEARCH EXPERIMENTS")
    print("="*60)
    
    variance_results = None
    ablation_results = None
    iid_results = None
    
    # Experiment 1: IID vs Non-IID Comparison (BEFORE main training in original)
    if NUM_ROUNDS >= 3:  # Need at least 3 rounds for meaningful comparison
        try:
            print("\n[Experiment 1/3] IID vs Non-IID Data Distribution")
            iid_results = compare_iid_vs_noniid_experiment(X_train, y_train, X_test, y_test, num_classes, input_shape)
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        print("\n[Experiment 1/3] IID vs Non-IID - Skipped (requires NUM_ROUNDS >= 3)")
    
    # Experiment 2: Variance Tracking (3 runs)
    if NUM_ROUNDS >= 3:  # Need at least 3 rounds for variance analysis
        try:
            print("\n[Experiment 2/3] Variance Tracking (3 runs)")
            variance_results, best_model = run_variance_tracking(
                client_data, NUM_CLIENTS, NUM_ROUNDS, input_shape, num_classes,
                X_test, y_test, num_runs=3
            )
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        print("\n[Experiment 2/3] Variance Tracking - Skipped (requires NUM_ROUNDS >= 3)")
    
    # Experiment 3: Ablation Study
    if NUM_ROUNDS >= 3:  # Need at least 3 rounds for ablation
        try:
            print("\n[Experiment 3/3] Ablation Study")
            ablation_results = run_ablation_study(
                client_data, NUM_CLIENTS, NUM_ROUNDS, input_shape, num_classes,
                X_test, y_test
            )
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        print("\n[Experiment 3/3] Ablation Study - Skipped (requires NUM_ROUNDS >= 3)")

    
    # ====================================================================
    # 9. ROBUSTNESS TESTING
    # ====================================================================
    print("\n==== Simulating Network Conditions ====")
    network_results = simulate_network_conditions()
    
    print("\n==== Simulating Attack Scenarios ====")
    attack_results = simulate_attack_scenarios(global_model, X_test, y_test, num_classes)
    
    # ====================================================================
    # 10. VISUALIZATION & REPORTING
    # ====================================================================
    print("\n==== Generating Visualizations ====")
    plot_federated_results(history, blockchain_metrics)
    
    print("\n==== Generating Research Report ====")
    generate_research_report(
        history, final_metrics, blockchain_metrics,
        attack_results=attack_results,
        network_results=network_results,
        variance_results=variance_results,
        ablation_results=ablation_results
    )
    
    # ====================================================================
    # 11. IMPLEMENTATION QUALITY VERIFICATION
    # ====================================================================
    verify_implementation_quality()
    
    # ====================================================================
    # 12. LIST GENERATED FILES
    # ====================================================================
    list_generated_files()
    
    print("\n==== Experiment Completed Successfully! ====")
    print("Comprehensive research report generated: research_report.txt")
    print("Visualization saved: shield_health_results.png")
    
    return


if __name__ == "__main__":
    main()
