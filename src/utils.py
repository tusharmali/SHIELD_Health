
import hashlib
import base64
import json
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

def generate_key_pair():
    """Generate a new RSA key pair for digital signatures"""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    
    # Serialize keys for storage/transmission
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_key, public_key, private_key_bytes, public_key_bytes

def sign_update(update, private_key):
    """Sign a model update with a private key"""
    # Hash the update first
    update_str = str(update)
    update_hash = hashlib.sha256(update_str.encode()).hexdigest()
    
    # Sign the hash
    signature = private_key.sign(
        update_hash.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    # Encode the signature as base64 string
    return base64.b64encode(signature).decode('utf-8')

def calculate_object_size(obj):
    """Calculate approximate size of a Python object in bytes"""
    import pickle
    import sys
    try:
        # Convert to JSON and measure string length
        json_str = json.dumps(obj)
        return len(json_str.encode('utf-8'))
    except:
        # Fallback: use pickle to get a more accurate size
        try:
            return len(pickle.dumps(obj))
        except:
            # If all else fails, return an estimate
            return sys.getsizeof(str(obj))
