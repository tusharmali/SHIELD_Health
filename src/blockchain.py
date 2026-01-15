
import time
import json
import hashlib
import numpy as np
import copy
from collections import defaultdict
from datetime import datetime
from .utils import sign_update, calculate_object_size
from .config import DATASET_PATH

class AccessControlPolicy:
    """
    Manages access control for IoT Resources (Section 3.7.2)
    """
    def __init__(self, resource_id, owner_id, permissions=None):
        self.resource_id = resource_id
        self.owner_id = owner_id
        self.permissions = permissions or {} # {user_id: [read, write, execute]}

    def grant_permission(self, user_id, permission_type):
        if user_id not in self.permissions:
            self.permissions[user_id] = []
        if permission_type not in self.permissions[user_id]:
            self.permissions[user_id].append(permission_type)
            
    def has_permission(self, user_id, permission_type):
        if user_id == self.owner_id:
            return True
        return permission_type in self.permissions.get(user_id, [])

    def to_dict(self):
        return self.__dict__

class IncentiveMechanism:
    """Privacy-preserving incentive mechanism for participating IoT devices"""
    def __init__(self, initial_tokens=100):
        self.device_tokens = defaultdict(lambda: initial_tokens)
        self.contribution_history = defaultdict(list)
        self.reward_history = defaultdict(list)
        self.token_value = 1.0  # Base token value
        
    def record_contribution(self, device_id, quality_score, data_volume, resource_usage):
        """Record a device's contribution to the federated learning process"""
        timestamp = datetime.now()
        contribution = {
            'timestamp': timestamp,
            'quality_score': quality_score,
            'data_volume': data_volume,
            'resource_usage': resource_usage
        }
        self.contribution_history[device_id].append(contribution)
        
    def calculate_reward(self, device_id, round_num):
        """Calculate reward for a device based on its contribution"""
        if not self.contribution_history[device_id]:
            return 0
        
        # Get the latest contribution
        contribution = self.contribution_history[device_id][-1]
        
        # Base reward calculation formula
        quality_factor = contribution['quality_score']
        volume_factor = min(1.0, contribution['data_volume'] / 1000)
        efficiency_factor = max(0.1, 1.0 - contribution['resource_usage'])
        
        # Weighted sum
        reward = (0.6 * quality_factor + 0.2 * volume_factor + 0.2 * efficiency_factor) * 10
        
        # Loyalty bonus
        consecutive_rounds = len(self.contribution_history[device_id])
        if consecutive_rounds > 5:
            reward *= 1.0 + (0.01 * consecutive_rounds)
        
        return round(reward, 2)
    
    def distribute_rewards(self, device_ids, round_num):
        """Distribute rewards to participating devices"""
        rewards = {}
        timestamp = datetime.now()
        
        for device_id in device_ids:
            reward = self.calculate_reward(device_id, round_num)
            self.device_tokens[device_id] += reward
            rewards[device_id] = reward
            
            self.reward_history[device_id].append({
                'timestamp': timestamp,
                'round': round_num,
                'reward': reward,
                'new_balance': self.device_tokens[device_id]
            })
        
        return rewards
    
    def get_device_balance(self, device_id):
        """Get current token balance for a device"""
        return self.device_tokens[device_id]
    
    def get_reward_history(self, device_id):
        """Get reward history for a device"""
        return self.reward_history[device_id]
    
    def can_participate(self, device_id, min_tokens=10):
        """Check if a device has enough tokens to participate"""
        return self.device_tokens[device_id] >= min_tokens


class BlockchainFL:
    """
    Lightweight Blockchain for Federated Learning (Section 3.7)
    """
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.nodes = set()
        self.access_policies = {} # resource_id -> AccessControlPolicy
        self.incentive_mechanism = IncentiveMechanism()
        self.transaction_cache = {}
        
        # Performance metrics
        self.metrics = {
             "block_times": [],
             "transaction_counts": [],
             "difficulty_history": []
        }
        
        # Create genesis block
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, address):
        self.nodes.add(address)

    def new_block(self, proof, previous_hash=None):
        """
        Create a new block and add it to the chain
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
            'difficulty': self.get_adaptive_difficulty()
        }
        
        # Reset current transactions
        self.current_transactions = []
        self.chain.append(block)
        
        # Update metrics
        self.metrics["transaction_counts"].append(len(block['transactions']))
        self.metrics["difficulty_history"].append(block['difficulty'])
        if len(self.chain) > 1:
            self.metrics["block_times"].append(block['timestamp'] - self.chain[-2]['timestamp'])
            
        return block

    def new_transaction(self, sender, recipient, type, content, signature=None, public_key=None):
        """
        Creates a new transaction to go into the next mined Block
        Types: 'model_update', 'access_policy', 'incentive'
        """
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'type': type, # model_update, access_policy, incentive
            'content': content,
            'timestamp': time.time()
        }
        
        if signature:
            transaction['signature'] = signature
            transaction['public_key'] = public_key
            
        self.current_transactions.append(transaction)
        return self.last_block['index'] + 1

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_adaptive_difficulty(self):
        """
        Adaptive Difficulty Adjustment
        D(t) = D_base * (1 + alpha * log(L_chain)) * (1 + beta * N_tx) * (T_target / T_avg)
        """
        if len(self.chain) < 2:
            return 2 # Base difficulty
        
        D_base = 2
        alpha = 0.1 # Chain length factor weight
        beta = 0.05 # Transaction volume factor weight
        T_target = 10.0 # Target block time (seconds)
        
        # Factor 1: Chain Length
        L_chain = len(self.chain)
        chain_factor = 1 + alpha * np.log(max(1, L_chain))
        
        # Factor 2: Transaction Volume
        # Average transactions in last 5 blocks
        last_blocks = self.chain[-5:]
        avg_tx = sum(len(b['transactions']) for b in last_blocks) / len(last_blocks)
        tx_factor = 1 + beta * avg_tx
        
        # Factor 3: Time Factor
        # Average time for last 5 blocks
        if len(self.metrics["block_times"]) > 0:
            avg_time = np.mean(self.metrics["block_times"][-5:])
        else:
            avg_time = T_target
            
        # Avoid division by zero
        avg_time = max(0.1, avg_time)
        time_factor = T_target / avg_time
        
        # Calculate Difficulty
        difficulty = D_base * chain_factor * tx_factor * time_factor
        
        # Clamp difficulty
        return max(1, min(int(difficulty), 6)) # High difficulty is too slow for simulation

    def valid_proof(self, last_proof, proof, difficulty):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:difficulty] == '0' * difficulty

    def proof_of_work(self, last_proof):
        proof = 0
        diff = self.get_adaptive_difficulty()
        while self.valid_proof(last_proof, proof, diff) is False:
            proof += 1
            # Add timeout/safety break for simulation purposes
            if proof > 100000: 
                # Force break to not hang simulation
                break 
        return proof
