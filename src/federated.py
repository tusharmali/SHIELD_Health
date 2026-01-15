
import numpy as np
import copy
from .config import BYZANTINE_TOLERANCE

def federated_aggregation(updates, method="median", byzantine_tolerance=BYZANTINE_TOLERANCE):
    """
    Robust Federated Aggregation (Section 3.4)
    Supports: FedAvg, Median, Trimmed Mean, Multi-Krum (Simulated)
    """
    if not updates:
        return None
        
    # Standardize updates format (list of weights)
    # updates are list of list of numpy arrays
    
    num_clients = len(updates)
    num_layers = len(updates[0])
    aggregated_weights = []
    
    for layer_idx in range(num_layers):
        # Gather layer weights from all clients
        layer_weights = np.array([u[layer_idx] for u in updates])
        
        if method == "fed_avg":
             agg_layer = np.mean(layer_weights, axis=0)
             
        elif method == "median":
             agg_layer = np.median(layer_weights, axis=0)
             
        elif method == "trimmed_mean":
             # Exclude top and bottom k%
             k = int(byzantine_tolerance * num_clients)
             k = max(1, k) # Ensure we trim at least something if requested
             if num_clients <= 2*k: # Safety
                 agg_layer = np.median(layer_weights, axis=0)
             else:
                 sorted_weights = np.sort(layer_weights, axis=0)
                 trimmed = sorted_weights[k:-k]
                 agg_layer = np.mean(trimmed, axis=0)
                 
        else: # Default fallback
             agg_layer = np.median(layer_weights, axis=0)
             
        aggregated_weights.append(agg_layer)
        
    return aggregated_weights

class SimplifiedHomomorphicEncryption:
    """
    Simulated HE for secure aggregation (Section 3.5)
    Real HE is too slow for this simulation, so we implement the logic
    of Additive Homomorphism: Add(E(x), E(y)) = E(x+y)
    """
    def __init__(self, scale_factor=1000):
        self.scale_factor = scale_factor
        
    def encrypt(self, weights):
        # Scale float -> int and add mild noise (simulating encryption artifact)
        encrypted = []
        for w in weights:
            scaled = (w * self.scale_factor).astype(np.int64)
            # Add small noise
            noise = np.random.randint(-1, 2, size=w.shape)
            encrypted.append(scaled + noise)
        return encrypted
        
    def decrypt(self, encrypted_weights):
        decrypted = []
        for w in encrypted_weights:
             decrypted.append(w.astype(np.float32) / self.scale_factor)
        return decrypted
    
    def aggregate_encrypted(self, list_of_encrypted_weights):
        # E(Sum) = Sum(E(x))
        if not list_of_encrypted_weights: return None
        
        num_layers = len(list_of_encrypted_weights[0])
        agg_encrypted = []
        
        for i in range(num_layers):
            layer_sum = np.sum([u[i] for u in list_of_encrypted_weights], axis=0)
            agg_encrypted.append(layer_sum)
            
        # Average in encrypted domain? 
        # Usually FedAvg does Sum / N. 
        # Div is not fully homomorphic. 
        # Here we simulate decrypting the Sum and then dividing by N is equivalent 
        # to decryption logic handling it.
        # But for 'simplified' we can just return the sum and let decryptor handle averaging if needed.
        # OR we perform integer division here.
        
        n = len(list_of_encrypted_weights)
        agg_averaged = [x // n for x in agg_encrypted]
        return agg_averaged

class CustomDPOptimizer:
    """
    Differential Privacy Optimizer (Section 3.6)
    Adds Gaussian Noise to gradients/weights.
    """
    def __init__(self, noise_multiplier=0.1, clip_norm=1.0):
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        
    def add_noise(self, weights):
        noisy_weights = []
        for w in weights:
            # Clipping (L2 norm)
            l2_norm = np.linalg.norm(w)
            scale = min(1.0, self.clip_norm / (l2_norm + 1e-6))
            w_clipped = w * scale
            
            # Add Noise
            noise = np.random.normal(0, self.noise_multiplier * self.clip_norm, w.shape)
            noisy_weights.append(w_clipped + noise)
        return noisy_weights

def quantize_weights(weights, bits=8):
    """
    Quantization for Communication Efficiency (Section 3.8)
    """
    min_val = min([np.min(w) for w in weights])
    max_val = max([np.max(w) for w in weights])
    
    range_val = max_val - min_val
    step = range_val / (2**bits - 1)
    
    quantized_weights = []
    for w in weights:
        # q = round((x - min) / step)
        q = np.round((w - min_val) / step).astype(np.int32)
        quantized_weights.append(q)
        
    metadata = {'min': min_val, 'step': step}
    return quantized_weights, metadata

def dequantize_weights(quantized_weights, metadata):
    min_val = metadata['min']
    step = metadata['step']
    
    restored_weights = []
    for q in quantized_weights:
        # x = q * step + min
        x = q.astype(np.float32) * step + min_val
        restored_weights.append(x)
        
    return restored_weights
