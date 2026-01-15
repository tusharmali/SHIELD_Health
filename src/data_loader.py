
import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import DATASET_PATH

def download_and_prepare_pamap2_dataset():
    """
    Download and prepare the PAMAP2 Physical Activity Monitoring dataset.
    Only supports PAMAP2 as per SHIELD-Health paper specifications.
    
    Returns:
        X_train, X_test, y_train, y_test, num_classes
    """
    # Create dataset directory if it doesn't exist
    if not os.path.exists(os.path.dirname(DATASET_PATH)):
        os.makedirs(os.path.dirname(DATASET_PATH))

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    zip_path = os.path.join(os.path.dirname(DATASET_PATH), "PAMAP2_Dataset.zip")
    extract_dir = DATASET_PATH # This should be the folder containing 'Protocol'
    
    # Check if Protocol directory exists within extract_dir
    protocol_dir = os.path.join(extract_dir, "Protocol")

    # Download the dataset if not already present
    if not os.path.exists(protocol_dir):
        print(f"Downloading PAMAP2 dataset to {zip_path}...")
        try:
            if not os.path.exists(zip_path):
                 urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(DATASET_PATH))
            
            # Cleanup zip
            if os.path.exists(zip_path):
                os.remove(zip_path)
            print("Download and extraction complete")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None, None, None, None, 0
    else:
        print("PAMAP2 Dataset already available")
    
    # List of protocol files to load
    # Note: extracted folder name might be PAMAP2_Dataset/PAMAP2_Dataset/Protocol depending on zip structure
    # Based on user's workspace, it seems it's d:\download-2\SHIELD_1\SHIELD_1\PAMAP2_Dataset\Protocol
    
    subject_ids = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    protocol_files = [os.path.join(protocol_dir, f"subject{sid}.dat") for sid in subject_ids]
    
    # Column names (based on PAMAP2 documentation)
    column_names = ['timestamp', 'activity_id', 'heart_rate']
    for i in range(1, 18):
        for j in ['x', 'y', 'z']:
            column_names.append(f'IMU{i}_{j}')
    
    # Load and preprocess the data
    all_data = []
    for file_path in protocol_files:
        if os.path.exists(file_path):
            try:
                # Load the data with appropriate column names
                data = pd.read_csv(file_path, sep=' ', header=None, names=column_names)
                # Forward fill and backward fill for missing values (Section 4.1 Imputation)
                data = data.ffill().bfill()
                print(f"Loaded {os.path.basename(file_path)}, shape: {data.shape}")
                all_data.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: File not found {file_path}")
    
    if not all_data:
        print("No valid data files found")
        return None, None, None, None, 0
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Extract features and labels
    X = combined_data.drop(['timestamp', 'activity_id'], axis=1)
    y = combined_data['activity_id']
    
    # Remove rows with NaN values if any remain
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # Remap activity labels to a continuous range starting from 0
    unique_labels = sorted(y.unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = y.map(label_map)
    
    # Standardize features (Section 4.1 Normalization)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to numpy arrays
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.to_numpy().astype(np.int32)
    y_test = y_test.to_numpy().astype(np.int32)
    
    num_classes = len(unique_labels)
    print(f"Dataset processed: Train {X_train.shape}, Test {X_test.shape}, Classes {num_classes}")
    
    return X_train, X_test, y_train, y_test, num_classes

def prepare_client_data(X_train, y_train, num_clients, iid=False):
    """
    Distribute data to clients, supporting Non-IID distribution (Dirichlet).
    Matches Section 4.1: Dirichlet distribution-based partitioning.
    """
    client_data = []
    
    if iid:
        # IID distribution
        samples_per_client = len(X_train) // num_clients
        indices = np.random.permutation(len(X_train))
        for i in range(num_clients):
            client_indices = indices[i * samples_per_client : (i + 1) * samples_per_client]
            client_data.append({
                "X": X_train[client_indices],
                "y": y_train[client_indices]
            })
    else:
        # Non-IID Dirichlet distribution (alpha=0.5 per paper)
        alpha = 0.5
        min_size = 0
        N = len(y_train)
        classes = np.unique(y_train)
        n_classes = len(classes)
        
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            for k in classes:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_batch[i]) < N / num_clients) for i, p in enumerate(proportions)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            indices = idx_batch[i]
            np.random.shuffle(indices)
            # 80/20 train/val split for local data
            c_X = X_train[indices]
            c_y = y_train[indices]
            
            c_X_train, c_X_val, c_y_train, c_y_val = train_test_split(
                c_X, c_y, test_size=0.2, random_state=42
            )
            
            client_data.append({
                "X": c_X_train,
                "y": c_y_train,
                "X_val": c_X_val,
                "y_val": c_y_val
            })
            
    return client_data
