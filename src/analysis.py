
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

def plot_federated_results(history, blockchain_metrics):
    """
    Plot results for SHIELD-Health.
    Matches the visualization requirements from the paper.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed, skipping plots.")
        return

    plt.figure(figsize=(20, 15))
    
    # Accuracy
    plt.subplot(3, 3, 1)
    plt.plot(history['accuracy'], marker='o', label='Accuracy')
    plt.title('Global Model Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Loss
    plt.subplot(3, 3, 2)
    plt.plot(history['loss'], marker='o', color='orange', label='Loss')
    plt.title('Global Model Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Blockchain - Block Times
    block_times = blockchain_metrics.get('block_times', [])
    if block_times:
        plt.subplot(3, 3, 3)
        plt.plot(block_times, marker='s', color='green')
        plt.title('Block Mining Time')
        plt.xlabel('Block Index')
        plt.ylabel('Time (s)')
        plt.grid(True)
        
    # Blockchain - Difficulty
    difficulty_history = blockchain_metrics.get('difficulty_history', [])
    if difficulty_history:
        plt.subplot(3, 3, 4)
        plt.plot(difficulty_history, marker='x', color='red')
        plt.title('Mining Difficulty Over Time')
        plt.xlabel('Block Index')
        plt.ylabel('Difficulty')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('shield_health_results.png')
    print("Results saved to shield_health_results.png")

def evaluate_model_performance(model, X_test, y_test, num_classes):
    """
    Comprehensive model evaluation with all metrics.
    
    Returns:
        Dict with accuracy, precision, recall, F1, AUC
    """
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Handle potential pandas Series
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    metrics = {
        "accuracy": np.mean(y_pred == y_test),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "loss": 0.0  # Computed elsewhere
    }
    
    # Try to compute AUC if applicable
    try:
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(y_test, y_pred_probs[:, 1])
        else:
            metrics["auc"] = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='weighted')
    except:
        metrics["auc"] = 0.0
    
    return metrics

def analyze_blockchain_performance(blockchain):
    """
    Analyze blockchain performance metrics.
    
    Returns:
        Dict with block times, throughput, storage
    """
    chain = blockchain.chain
    metrics = blockchain.metrics
    
    num_blocks = len(chain)
    total_tx = sum(metrics.get("transaction_counts", []))
    
    block_times = metrics.get("block_times", [])
    avg_block_time = np.mean(block_times) if block_times else 0.0
    
    # Estimate storage (simplified)
    import sys
    storage_bytes = sum(sys.getsizeof(str(block)) for block in chain)
    storage_mb = storage_bytes / (1024 * 1024)
    
    # Throughput
    total_time = sum(block_times) if block_times else 1.0
    throughput_tx_per_sec = total_tx / total_time if total_time > 0 else 0.0
    
    return {
        "num_blocks": num_blocks,
        "total_tx": total_tx,
        "avg_block_time": avg_block_time,
        "block_times": block_times,  # Include for plotting
        "difficulty_history": metrics.get("difficulty_history", []),  # Include for plotting
        "storage_mb": storage_mb,
        "storage_growth_rate_mb_per_block": storage_mb / num_blocks if num_blocks > 0 else 0,
        "throughput_tx_per_sec": throughput_tx_per_sec
    }


def generate_research_report(history, final_metrics, blockchain_metrics, 
                            attack_results=None, network_results=None,
                            variance_results=None, ablation_results=None):
    """
    Generates a comprehensive research report.
    """
    report = f"""
=================================================================
    SHIELD-Health: Comprehensive Research Report
=================================================================

FINAL MODEL PERFORMANCE
-----------------------
Accuracy:  {final_metrics.get('accuracy', 0):.4f}
Precision: {final_metrics.get('precision', 0):.4f}
Recall:    {final_metrics.get('recall', 0):.4f}
F1 Score:  {final_metrics.get('f1_score', 0):.4f}

BLOCKCHAIN PERFORMANCE
----------------------
Total Blocks:     {blockchain_metrics.get('num_blocks', 0)}
Total Transactions: {blockchain_metrics.get('total_tx', 0)}
Avg Block Time:   {blockchain_metrics.get('avg_block_time', 0):.2f} seconds
Storage Size:     {blockchain_metrics.get('storage_mb', 0):.2f} MB
Throughput:       {blockchain_metrics.get('throughput_tx_per_sec', 0):.2f} tx/sec

CONVERGENCE ANALYSIS
--------------------
Total Rounds:     {len(history.get('accuracy', []))}
Final Accuracy:   {history['accuracy'][-1] if history.get('accuracy') else 0:.4f}
Final Loss:       {history['loss'][-1] if history.get('loss') else 0:.4f}

"""
    
    if variance_results:
        report += f"""
VARIANCE ANALYSIS
-----------------
Mean Accuracy:    {variance_results.get('mean_accuracy', 0):.4f} ± {variance_results.get('std_accuracy', 0):.4f}
Range:            [{variance_results.get('min_accuracy', 0):.4f}, {variance_results.get('max_accuracy', 0):.4f}]

"""
    
    if attack_results:
        report += "\nROBUSTNESS TESTING\n------------------\n"
        for scenario, acc in attack_results.items():
            report += f"{scenario}: {acc:.4f}\n"
    
    if network_results:
        report += "\nNETWORK CONDITIONS\n------------------\n"
        for condition, acc in network_results.items():
            report += f"{condition}: {acc:.4f}\n"
    
    report += "\n" + "="*65 + "\n"
    report += "Report generated by SHIELD-Health Framework\n"
    report += "="*65 + "\n"
    
    with open("research_report.txt", "w") as f:
        f.write(report)
    print("Comprehensive research report saved to research_report.txt")

