
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv1D, BatchNormalization, Activation, MaxPooling1D, Bidirectional, LSTM, Dropout, Reshape, Multiply, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable
import random
from datetime import datetime

@register_keras_serializable()
class TemporalAttention(Layer):
    """
    Custom layer    Implements Dot-Product Attention mechanism:
    alpha_{t,i} = softmax(W_q h_t \cdot W_k h_i^T)
    """
    def __init__(self, attention_units=32, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.attention_units = attention_units
        
    def build(self, input_shape):
        # input_shape: (batch_size, time_steps, features)
        self.W_q = self.add_weight(
            shape=(input_shape[-1], self.attention_units),
            initializer='glorot_uniform',
            trainable=True,
            name='W_q'
        )
        self.W_k = self.add_weight(
            shape=(input_shape[-1], self.attention_units),
            initializer='glorot_uniform',
            trainable=True,
            name='W_k'
        )
        self.W_v = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]), # Maintain feature dim
            initializer='glorot_uniform',
            trainable=True,
            name='W_v'
        )
        super(TemporalAttention, self).build(input_shape)
    
    def call(self, inputs):
        # inputs: [batch_size, T, F]
        
        # Query projection: Q = H * W_q  => [batch, T, units]
        Q = tf.tensordot(inputs, self.W_q, axes=1)
        
        # Key projection: K = H * W_k    => [batch, T, units]
        K = tf.tensordot(inputs, self.W_k, axes=1)
        
        # Value projection: V = H * W_v  => [batch, T, F]
        V = tf.tensordot(inputs, self.W_v, axes=1)
        
        # Attention scores: Q * K^T
        # [batch, T, units] x [batch, units, T] => [batch, T, T]
        scores = tf.matmul(Q, K, transpose_b=True)
        
        # Scale scores (optional but standard for dot product attention)
        d_k = tf.cast(tf.shape(self.W_q)[-1], tf.float32)
        scores = scores / tf.math.sqrt(d_k)
        
        # Softmax over the last dimension (time)
        weights = tf.nn.softmax(scores, axis=-1)
        
        # Context vectors: weights * V
        # [batch, T, T] x [batch, T, F] => [batch, T, F]
        context = tf.matmul(weights, V)
        
        # Determine how to aggregate context. 
        # Paper implies a single context vector c_t per time step, ultimately aggregated for classification.
        # We'll return the simpler global context for classification purposes: sum over time
        global_context = tf.reduce_sum(context, axis=1)
        
        return global_context, weights
        
    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        config.update({"attention_units": self.attention_units})
        return config

class DeviceProfile:
    """Profile for an IoT device's computational capabilities (Section 3.3)"""
    def __init__(self, device_id, cpu_power, memory, battery_capacity, 
                 network_bandwidth, is_mobile=True):
        self.device_id = device_id
        self.cpu_power = cpu_power
        self.memory = memory
        self.battery_capacity = battery_capacity
        self.network_bandwidth = network_bandwidth
        self.is_mobile = is_mobile
        self.current_load = 0
        self.available_time = 24
        
    def estimate_capability_score(self):
        """Estimate capability score for resource-aware scaling"""
        score = (
            0.3 * (self.cpu_power / 10) + 
            0.2 * (self.memory / 1024) + 
            0.2 * (self.battery_capacity / 100) + 
            0.2 * (self.network_bandwidth / 20) + 
            0.1 * (1 - self.current_load)
        )
        if self.is_mobile:
            score *= 0.8
        return score

class ResourceAwareModelSelector:
    """
    Selects model complexity based on device profile.
    Since we want to enforce consistency, we use the SAME architecture 
    but we might vary hyperparameters if needed.
    However, strictly speaking, FL requires identical architectures for aggregation.
    The 'resource-aware' part in the paper often refers to *batch size* or *participation frequency* adaptation,
    OR using techniques like dropout/pruning during training.
    
    For strict paper alignment (Section 3.3), we simulate resource constraints 
    by assigning 'complexity' which dictates BATCH SIZE in the training loop, 
    rather than changing the model architecture itself (which breaks FedAvg).
    """
    def __init__(self):
        self.device_profiles = {}
        
    def register_device(self, device_profile):
        self.device_profiles[device_profile.device_id] = device_profile
        
    def get_complexity_assignment(self, device_id):
        profile = self.device_profiles.get(device_id)
        if not profile:
            return "medium"
        
        score = profile.estimate_capability_score()
        if score > 0.7: return "high"
        elif score > 0.4: return "medium"
        else: return "low"

def build_medical_iot_model(input_shape, num_classes, sequence_length=10):
    """
    Builds the SHIELD-Health temporal model.
    """
    # Calculate features per timestep
    if isinstance(input_shape, tuple):
         flat_dim = input_shape[0]
    else:
         flat_dim = input_shape
         
    features_per_step = flat_dim // sequence_length
    # Handle remainder
    if flat_dim % sequence_length != 0:
        features_per_step += 1
        
    inputs = tf.keras.Input(shape=(flat_dim,))
    
    # Reshape logic to handle padding if needed
    def pad_reshape(x):
        pad_size = (sequence_length * features_per_step) - tf.shape(x)[-1]
        x_padded = tf.pad(x, [[0, 0], [0, pad_size]])
        return tf.reshape(x_padded, [-1, sequence_length, features_per_step])
        
    x = Lambda(pad_reshape)(inputs)
    
    # CNN Layers (Feature Extraction)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    # LSTM Layers (Temporal Dependencies)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Temporal Attention (Key Innovation)
    x, attn_weights = TemporalAttention(attention_units=32)(x)
    
    # Classification
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
