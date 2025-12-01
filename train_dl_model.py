#!/usr/bin/env python3
"""
Train neural network model to predict speedup.

Input features:
- instruction_type, LS, LS_equals_VS, K_remainder, remainder_strategy, X_times_Y, LS_div_K

Output: speedup

Special handling: X_times_Y has linear relationship - model should know this.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Model export to ONNX will be skipped.")
import argparse
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SpeedupDataset(Dataset):
    """Dataset for speedup prediction"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SpeedupPredictor(nn.Module):
    """
    Neural network model for speedup prediction.
    
    Architecture:
    - Separate paths for non-linear features and linear X_times_Y feature
    - Non-linear branch: instruction_type, LS, LS_equals_VS, K_remainder, remainder_strategy, LS_div_K
    - Linear branch: X_times_Y (explicit linear connection)
    - Combine both branches at output
    """
    def __init__(self, input_dim=7, hidden_dims=[64, 32, 16], dropout=0.1):
        super(SpeedupPredictor, self).__init__()
        
        # Non-linear branch (6 features: instruction_type, LS, LS_equals_VS, K_remainder, remainder_strategy, LS_div_K)
        self.nonlin_branch = nn.Sequential(
            nn.Linear(6, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
        )
        
        # Linear branch for X_times_Y (explicit linear connection)
        self.linear_branch = nn.Linear(1, 1, bias=False)  # Linear only, no bias initially
        
        # Combine branches
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[2] + 1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Initialize linear branch to encourage linear relationship
        nn.init.xavier_uniform_(self.linear_branch.weight)
    
    def forward(self, x):
        # Split input: first 6 features go to non-linear branch, last (X_times_Y) goes to linear
        nonlin_features = x[:, :6]  # instruction_type, LS, LS_equals_VS, K_remainder, remainder_strategy, LS_div_K
        x_times_y = x[:, 6:7]  # X_times_Y (keep as 2D for linear layer)
        
        # Non-linear branch
        nonlin_out = self.nonlin_branch(nonlin_features)
        
        # Linear branch (explicit linear transformation)
        linear_out = self.linear_branch(x_times_y)
        
        # Combine
        combined = torch.cat([nonlin_out, linear_out], dim=1)
        output = self.output(combined)
        
        return output


def load_and_prepare_data(csv_path):
    """Load and prepare training data"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['instruction_type', 'LS', 'LS_equals_VS', 'K_remainder', 
                     'remainder_strategy', 'X_times_Y', 'LS_div_K', 'speedup']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract features and target
    feature_cols = ['instruction_type', 'LS', 'LS_equals_VS', 'K_remainder', 
                    'remainder_strategy', 'X_times_Y', 'LS_div_K']
    X = df[feature_cols].values
    y = df['speedup'].values
    
    print(f"\nFeature statistics:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: min={X[:, i].min()}, max={X[:, i].max()}, mean={X[:, i].mean():.2f}")
    print(f"\nTarget (speedup): min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    
    # Handle missing/invalid values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nValid samples after filtering: {len(X)}")
    
    return X, y, feature_cols


def normalize_features(X_train, X_test, X_times_Y_idx=5):
    """Normalize features, handling X_times_Y specially for linear relationship"""
    scaler = StandardScaler()
    
    # Normalize all features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # For X_times_Y, also create log transform to help with linear relationship
    # But we'll keep the original normalized version for the linear branch
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()
    X_train_log[:, X_times_Y_idx] = np.log1p(X_train[:, X_times_Y_idx])
    X_test_log[:, X_times_Y_idx] = np.log1p(X_test[:, X_times_Y_idx])
    
    # Normalize log version too
    scaler_log = StandardScaler()
    X_train_log_scaled = scaler_log.fit_transform(X_train_log)
    X_test_log_scaled = scaler_log.transform(X_test_log)
    
    return (X_train_scaled, X_test_scaled), (scaler, scaler_log)


def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001):
    """Train the neural network model"""
    print(f"\nTraining model...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = SpeedupDataset(X_train, y_train)
    val_dataset = SpeedupDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SpeedupPredictor(input_dim=X_train.shape[1], hidden_dims=[64, 32, 16], dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model performance"""
    model.eval()
    device = next(model.parameters()).device
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().squeeze()
    
    # Metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'predictions': predictions
    }


def export_to_onnx(model, output_path, input_dim=6, input_names=None, output_names=None):
    """Export model to ONNX format for C++ use"""
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(1, input_dim).to(device)
    
    if input_names is None:
        input_names = ['features']
    if output_names is None:
        output_names = ['speedup']
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'features': {0: 'batch_size'},
            'speedup': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"Model exported to ONNX: {output_path}")
    
    # Verify ONNX model (if available)
    if ONNX_AVAILABLE:
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verified ✓")
        except Exception as e:
            print(f"Warning: ONNX verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train NN model for speedup prediction')
    parser.add_argument('--data', default='dl_training_data/training_data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--output-dir', default='models',
                       help='Output directory for saved models')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set split ratio')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation set split ratio')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X, y, feature_cols = load_and_prepare_data(args.data)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_split, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Normalize features
    (X_train_scaled, X_val_scaled), (scaler, scaler_log) = normalize_features(X_train, X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model, train_losses, val_losses = train_model(
        X_train_scaled, y_train, X_val_scaled, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    
    # Save PyTorch model
    torch_model_path = os.path.join(args.output_dir, 'speedup_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': test_metrics
    }, torch_model_path)
    print(f"\nPyTorch model saved: {torch_model_path}")
    
    # Export to ONNX for C++ use
    onnx_path = os.path.join(args.output_dir, 'speedup_model.onnx')
    try:
        export_to_onnx(model, onnx_path, input_dim=X_train_scaled.shape[1],
                       input_names=['features'], output_names=['speedup'])
    except Exception as e:
        print(f"\nONNX export failed: {e}")
        print("  The PyTorch model (.pth) has been saved and can still be used.")
        print("  For ONNX export, install: pip install onnx onnxruntime onnxscript")
    
    # Save scaler for C++ preprocessing
    import pickle
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")
    
    print(f"\nModel training complete!")
    if os.path.exists(onnx_path):
        print(f"  ONNX Model: {onnx_path}")
    print(f"  PyTorch Model: {torch_model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Ready for C++ integration!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

