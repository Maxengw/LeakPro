"""Utilities for loading and preprocessing tabular datasets for gradient inversion attacks."""

import os
import pickle
import urllib.request
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import torch
from torch import Tensor, float32, tensor
from torch.utils.data import DataLoader, Dataset, Subset


class TabularDataset(Dataset):
    """Tabular dataset with proper categorical/numerical feature handling for GIA."""
    
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        feature_info: Dict,
        one_hot_encoded: bool = True,
        scaler_stats: Dict = None,
        category_mappings: Dict = None
    ):
        """Initialize tabular dataset.
        
        Args:
            x: Input features tensor (one-hot encoded if one_hot_encoded=True)
            y: Labels tensor
            feature_info: Dictionary with feature metadata:
                - 'numerical_features': List of numerical feature names
                - 'categorical_features': List of categorical feature names
                - 'feature_ranges': Dict of (min, max) for numerical features
                - 'dec_to_onehot': Mapping from original feature indices to one-hot indices
            one_hot_encoded: Whether categorical features are one-hot encoded
            scaler_stats: Dictionary with 'mean' and 'std' for numerical features
            category_mappings: Dictionary mapping categorical feature names to category lists
        """
        self.x = x
        self.y = y
        self.feature_info = feature_info
        self.one_hot_encoded = one_hot_encoded
        self.scaler_stats = scaler_stats or {}
        self.category_mappings = category_mappings or {}
        
        # Store useful properties
        self.num_features = x.shape[1]
        self.num_samples = len(y)
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get a single sample."""
        return self.x[idx], self.y[idx]
    
    def subset(self, indices: List[int]) -> 'TabularDataset':
        """Create a subset of the dataset."""
        return TabularDataset(
            self.x[indices],
            self.y[indices],
            self.feature_info,
            self.one_hot_encoded,
            self.scaler_stats,
            self.category_mappings
        )


def download_adult_dataset(data_dir: str) -> None:
    """Download the Adult Dataset if not already present.
    
    Args:
        data_dir: Directory to save the dataset files
    """
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    data_file = os.path.join(data_dir, "adult.data")
    test_file = os.path.join(data_dir, "adult.test")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Download data files if not present
    if not os.path.exists(data_file):
        print("Downloading adult.data...")
        urllib.request.urlretrieve(base_url + "adult.data", data_file)
        print("Download complete.")
    
    if not os.path.exists(test_file):
        print("Downloading adult.test...")
        urllib.request.urlretrieve(base_url + "adult.test", test_file)
        print("Download complete.")


def preprocess_adult_dataset(data_dir: str, force_reprocess: bool = False) -> TabularDataset:
    """Load and preprocess the Adult dataset.
    
    Args:
        data_dir: Directory containing the raw data files
        force_reprocess: If True, reprocess even if cached version exists
        
    Returns:
        TabularDataset with preprocessed data
    """
    cache_file = os.path.join(data_dir, "adult_preprocessed.pkl")
    
    # Load from cache if available
    if os.path.exists(cache_file) and not force_reprocess:
        print(f"Loading preprocessed data from {cache_file}")
        with open(cache_file, "rb") as f:
            dataset_info = joblib.load(f)
        
        dataset = TabularDataset(
            x=dataset_info['x'],
            y=dataset_info['y'],
            feature_info=dataset_info['feature_info'],
            one_hot_encoded=True,
            scaler_stats=dataset_info['scaler_stats'],
            category_mappings=dataset_info['category_mappings']
        )
        print(f"Loaded {len(dataset)} samples with {dataset.num_features} features")
        return dataset
    
    # Process from raw data
    print("Processing Adult dataset from raw data...")
    
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    
    # Load and clean data
    data_file = os.path.join(data_dir, "adult.data")
    test_file = os.path.join(data_dir, "adult.test")
    
    df_train = pd.read_csv(data_file, names=column_names, skipinitialspace=True)
    df_test = pd.read_csv(test_file, names=column_names, skiprows=1, skipinitialspace=True)
    
    # Clean test set income column (remove trailing period)
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
    
    # Concatenate and clean
    df_concatenated = pd.concat([df_train, df_test], axis=0)
    df_clean = df_concatenated.replace(" ?", np.nan).replace("?", np.nan).dropna()
    
    print(f"Total samples after cleaning: {len(df_clean)}")
    
    # Split features and labels
    x = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]
    
    # Identify categorical and numerical features
    categorical_features = [col for col in x.columns if x[col].dtype == "object"]
    numerical_features = [col for col in x.columns if x[col].dtype in ["int64", "float64"]]
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Scale numerical features
    scaler = StandardScaler()
    x_numerical = pd.DataFrame(
        scaler.fit_transform(x[numerical_features]),
        columns=numerical_features,
        index=x.index
    )
    scaler_stats = {
        'mean': scaler.mean_,
        'std': scaler.scale_
    }
    
    # One-hot encode categorical features
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    x_categorical_one_hot = one_hot_encoder.fit_transform(x[categorical_features])
    one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_features)
    x_categorical_one_hot_df = pd.DataFrame(
        x_categorical_one_hot,
        columns=one_hot_feature_names,
        index=x.index
    )
    
    # Store category mappings for reconstruction evaluation
    category_mappings = {}
    for i, cat_feature in enumerate(categorical_features):
        category_mappings[cat_feature] = one_hot_encoder.categories_[i].tolist()
    
    # Concatenate numerical and one-hot encoded categorical features
    x_final = pd.concat([x_numerical, x_categorical_one_hot_df], axis=1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y))
    
    # Build feature info mapping
    dec_to_onehot_mapping = {}
    
    # Add numerical features
    for i, feature in enumerate(numerical_features):
        dec_to_onehot_mapping[i] = [x_final.columns.get_loc(feature)]
    
    # Add one-hot encoded categorical features
    for i, categorical_feature in enumerate(categorical_features):
        j = i + len(numerical_features)
        one_hot_columns = [col for col in one_hot_feature_names if col.startswith(categorical_feature)]
        dec_to_onehot_mapping[j] = [x_final.columns.get_loc(col) for col in one_hot_columns]
    
    # Compute feature ranges for numerical features
    feature_ranges = {}
    for i, feature in enumerate(numerical_features):
        # Store normalized ranges (since we'll work in normalized space)
        feature_ranges[feature] = (x_numerical[feature].min(), x_numerical[feature].max())
    
    feature_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'feature_ranges': feature_ranges,
        'dec_to_onehot': dec_to_onehot_mapping,
        'one_hot_feature_names': one_hot_feature_names.tolist(),
        'num_features_original': len(numerical_features) + len(categorical_features),
        'num_features_encoded': x_final.shape[1]
    }
    
    # Convert to tensors
    x_tensor = tensor(x_final.values, dtype=float32)
    y_tensor = tensor(y_encoded.values, dtype=float32)
    
    # Create dataset
    dataset = TabularDataset(
        x=x_tensor,
        y=y_tensor,
        feature_info=feature_info,
        one_hot_encoded=True,
        scaler_stats=scaler_stats,
        category_mappings=category_mappings
    )
    
    # Cache the preprocessed data
    dataset_info = {
        'x': x_tensor,
        'y': y_tensor,
        'feature_info': feature_info,
        'scaler_stats': scaler_stats,
        'category_mappings': category_mappings
    }
    
    with open(cache_file, "wb") as f:
        pickle.dump(dataset_info, f)
    print(f"Saved preprocessed data to {cache_file}")
    
    print(f"Final dataset: {len(dataset)} samples, {dataset.num_features} features")
    return dataset


def create_dataloaders(
    dataset: TabularDataset,
    batch_size: int,
    train_fraction: float = 0.3,
    test_fraction: float = 0.3,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    """Create train and test dataloaders from a tabular dataset.
    
    Args:
        dataset: TabularDataset to split
        batch_size: Batch size for dataloaders
        train_fraction: Fraction of dataset to use for training
        test_fraction: Fraction of dataset to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, test_loader, train_indices, test_indices)
    """
    np.random.seed(random_seed)
    
    dataset_size = len(dataset)
    train_size = int(train_fraction * dataset_size)
    test_size = int(test_fraction * dataset_size)
    
    # Sample indices
    selected_indices = np.random.choice(
        np.arange(dataset_size),
        train_size + test_size,
        replace=False
    )
    
    # Split into train and test
    train_indices, test_indices = train_test_split(
        selected_indices,
        test_size=test_size,
        random_state=random_seed
    )
    
    # Create subsets
    train_subset = Subset(dataset, train_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_indices.tolist(), test_indices.tolist()


def get_data_statistics(dataset: TabularDataset) -> Dict:
    """Compute statistical properties of the dataset for use in attacks.
    
    Args:
        dataset: TabularDataset to analyze
        
    Returns:
        Dictionary with data statistics including means, stds, category frequencies
    """
    x = dataset.x.numpy()
    y = dataset.y.numpy()
    
    stats = {
        'x_mean': x.mean(axis=0),
        'x_std': x.std(axis=0),
        'x_min': x.min(axis=0),
        'x_max': x.max(axis=0),
        'y_distribution': np.bincount(y.astype(int)) / len(y),
        'num_samples': len(dataset),
        'num_features': dataset.num_features
    }
    
    # Add per-feature category frequencies for one-hot encoded features
    feature_info = dataset.feature_info
    categorical_stats = {}
    
    for cat_feature in feature_info['categorical_features']:
        # Find the one-hot columns for this categorical feature
        feature_idx = feature_info['categorical_features'].index(cat_feature)
        dec_idx = len(feature_info['numerical_features']) + feature_idx
        onehot_indices = feature_info['dec_to_onehot'][dec_idx]
        
        # Compute frequency of each category
        category_frequencies = x[:, onehot_indices].mean(axis=0)
        categorical_stats[cat_feature] = category_frequencies
    
    stats['categorical_frequencies'] = categorical_stats
    
    return stats
