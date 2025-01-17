############################################################################
#### OpenPyStruct FNN with Residual Blocks Based Multi Load Case Optimizer ####
#### Coder: Danny Smyl, PhD, PE, Georgia Tech, 2025                     ####
############################################################################

import os
import json
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, RegularPolygon, Rectangle
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import mode

# Use new AMP API to avoid deprecation warnings
from torch.cuda.amp import autocast, GradScaler

#######################################
# 1) CONFIGURATION & HYPERPARAMETERS
#######################################

# Model and training configuration #
n_cases = 6                 # Number of sub-cases per sample
nelem = 100                 # Final output dimension per sample: (B, n_elem)
box_constraint_coeff = 5e-1 # Coefficient for box constraint penalty
hidden_units = 128          # Number of hidden units in MLP
dropout_rate = 0.5          # Dropout rate for regularization
num_blocks = 2              # Number of blocks (unused in current model)
num_epochs = 500            # Maximum number of training epochs
batch_size = 1024            # Batch size for training
patience = 10               # Early stopping patience
learning_rate = 3e-3        # Learning rate for optimizer
weight_decay = 3e-3         # Weight decay (L2 regularization) for optimizer
train_split = 0.8           # Fraction of data used for training
sigma_0 = 0.01              # Initial Gaussian noise for input
gamma_noise = 0.99          # Decay rate for noise during training
gamma = 0.95                # Learning rate scheduler decay rate
initial_alpha = 0.5         # Initial alpha value for loss weighting
c = 0.5                     # Parameter to adjust label aggregation (higher c = more more conservative I estimate)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################
# 2) DATA LOADING & PREPROCESSING
#######################################

def pad_sequences(data_list, max_length, pad_val=0.0):
    """
    Pad each 1D array in data_list to 'max_length'.
    Returns a NumPy array of shape (num_samples, max_length).
    """
    out = np.full((len(data_list), max_length), pad_val, dtype=np.float32)
    for i, arr in enumerate(data_list):
        arr_np = np.array(arr, dtype=np.float32)
        length = min(len(arr_np), max_length)
        out[i, :length] = arr_np[:length]
    return out

## mean ##
def unify_label_with_c(I_3d, c=c):
    """
    Aggregate labels by computing the mean across cases and adding c times the standard deviation.
    
    Parameters:
    - I_3d: NumPy array of shape (B, n_cases, n_elem)
    - c: Scalar multiplier for standard deviation
    
    Returns:
    - Y: NumPy array of shape (B, n_elem)
    """
    I_mean = I_3d.mean(axis=1)  # Mean across cases
    I_std  = I_3d.std(axis=1)   # Standard deviation across cases
    return I_mean + c * I_std

def fit_transform_3d(arr_3d, scaler):
    """
    Fit and transform a 3D array using the provided scaler over axis=0.
    
    Parameters:
    - arr_3d: NumPy array of shape (B, NC, M)
    - scaler: Scaler instance (e.g., StandardScaler)
    
    Returns:
    - scaled_arr: NumPy array of shape (B, NC, M)
    """
    B, NC, M = arr_3d.shape
    flat = arr_3d.reshape(B * NC, M)  # Combine B and NC for fitting
    scaled = scaler.fit_transform(flat)
    return scaled.reshape(B, NC, M)

def merge_sub_features(*arrays):
    """
    Concatenate multiple feature arrays along the feature dimension.
    
    Parameters:
    - arrays: Variable number of NumPy arrays to concatenate
    
    Returns:
    - merged_array: Concatenated NumPy array
    """
    return np.concatenate(arrays, axis=2)

def pad_feat_dim_to_multiple_of_nheads(X_3d, nheads):
    """
    Pad the feature dimension to be a multiple of nheads.
    
    Parameters:
    - X_3d: NumPy array of shape (B, Nc, original_dim)
    - nheads: Integer, number of attention heads
    
    Returns:
    - X_3d_padded: Padded NumPy array
    - new_dim: New feature dimension after padding
    """
    B, Nc, original_dim = X_3d.shape
    remainder = original_dim % nheads
    if remainder == 0:
        return X_3d, original_dim
    new_dim = ((original_dim // nheads) + 1) * nheads
    diff = new_dim - original_dim
    X_3d_padded = np.pad(X_3d, ((0,0), (0,0), (0,diff)), mode='constant')
    return X_3d_padded, new_dim

def scale_user_inputs(
    user_roller, user_force_x, user_force_vals, user_node_pos, 
    scalers, n_cases, max_lengths
):
    """
    Scales the user inputs using the fitted scalers.
    
    Parameters:
    - user_roller: List of roller locations per case
    - user_force_x: List of force positions per case
    - user_force_vals: List of force values per case
    - user_node_pos: List of node positions per case
    - scalers: Dictionary of fitted scalers for each feature
    - n_cases: Number of cases
    - max_lengths: Dictionary of maximum lengths for padding
    
    Returns:
    - feat_3d: NumPy array of shape (1, n_cases, feat_dim)
    """
    def pad_to_length(seq, req_len):
        arr = np.zeros((req_len,), dtype=np.float32)
        length = min(len(seq), req_len)
        arr[:length] = seq[:length]
        return arr

    feat_arrays = []
    for i in range(n_cases):
        # Pad each input type to its respective max length
        r_pad  = pad_to_length(user_roller[i],   max_lengths['roller_x'])
        fx_pad = pad_to_length(user_force_x[i],  max_lengths['force_x'])
        fv_pad = pad_to_length(user_force_vals[i], max_lengths['force_values'])
        nd_pad = pad_to_length(user_node_pos[i], max_lengths['node_positions'])

        # Scale using the fitted scalers
        r_scaled  = scalers["roller_x"].transform(r_pad.reshape(1, -1)).flatten()
        fx_scaled = scalers["force_x"].transform(fx_pad.reshape(1, -1)).flatten()
        fv_scaled = scalers["force_values"].transform(fv_pad.reshape(1, -1)).flatten()
        nd_scaled = scalers["node_positions"].transform(nd_pad.reshape(1, -1)).flatten()

        # Concatenate scaled features
        sub_feat = np.concatenate([r_scaled, fx_scaled, fv_scaled, nd_scaled])
        feat_arrays.append(sub_feat)

    feat_2d = np.stack(feat_arrays, axis=0)  # (n_cases, total_feat_dim)
    feat_3d = feat_2d[np.newaxis, ...]       # (1, n_cases, total_feat_dim)
    return feat_3d

# Load data
try:
    with open("StructDataLite.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("The file 'training_data_PINN.json' was not found.")

# Extract data
roller_x       = data.get("roller_x_locations", [])
force_x        = data.get("force_x_locations", [])
force_values   = data.get("force_values", [])
node_positions = data.get("node_positions", [])
I_values       = data.get("I_values", [])

num_samples = len(I_values)
req_keys = ["roller_x_locations","force_x_locations","force_values","node_positions"]
if not all(len(data.get(k, [])) == num_samples for k in req_keys):
    raise ValueError("Mismatch in sample counts among roller_x, force_x, force_values, node_positions.")

# Determine maximum lengths for padding
max_lengths = {
    "roller_x": max(len(r) for r in roller_x)       if roller_x       else 0,
    "force_x": max(len(r) for r in force_x)        if force_x        else 0,
    "force_values": max(len(r) for r in force_values)   if force_values   else 0,
    "node_positions": max(len(r) for r in node_positions) if node_positions else 0,
    "I_values": max(len(r) for r in I_values)       if I_values       else 0
}

# Pad sequences
roller_x_pad  = pad_sequences(roller_x,    max_lengths["roller_x"])
force_x_pad   = pad_sequences(force_x,     max_lengths["force_x"])
force_val_pad = pad_sequences(force_values, max_lengths["force_values"])
node_pos_pad  = pad_sequences(node_positions, max_lengths["node_positions"])
I_values_pad  = pad_sequences(I_values,    max_lengths["I_values"])

# Group data by n_cases
total_grouped = num_samples // n_cases
if total_grouped == 0:
    raise ValueError(f"n_cases={n_cases} > total samples={num_samples}.")

trim_len = total_grouped * n_cases
roller_x_pad  = roller_x_pad[:trim_len]
force_x_pad   = force_x_pad[:trim_len]
force_val_pad = force_val_pad[:trim_len]
node_pos_pad  = node_pos_pad[:trim_len]
I_values_pad  = I_values_pad[:trim_len]

roller_grouped    = roller_x_pad.reshape(total_grouped, n_cases, -1)
force_x_grouped   = force_x_pad.reshape(total_grouped, n_cases, -1)
force_val_grouped = force_val_pad.reshape(total_grouped, n_cases, -1)
node_grouped      = node_pos_pad.reshape(total_grouped, n_cases, -1)
I_grouped         = I_values_pad.reshape(total_grouped, n_cases, -1)

# Train/Validation Split
indices   = np.random.permutation(total_grouped)
train_sz  = int(train_split * total_grouped)
train_idx = indices[:train_sz]
val_idx   = indices[train_sz:]

roller_train    = roller_grouped[train_idx]
roller_val      = roller_grouped[val_idx]
force_x_train   = force_x_grouped[train_idx]
force_x_val     = force_x_grouped[val_idx]
force_val_train = force_val_grouped[train_idx]
force_val_val   = force_val_grouped[val_idx]
node_train      = node_grouped[train_idx]
node_val        = node_grouped[val_idx]
I_train         = I_grouped[train_idx]
I_val           = I_grouped[val_idx]

# Initialize Scalers
scalers_inputs = {
    "roller_x":       StandardScaler(),
    "force_x":        StandardScaler(),
    "force_values":   StandardScaler(),
    "node_positions": StandardScaler()
}

scaler_Y = StandardScaler()

# Fit and transform training data
roller_train_std    = fit_transform_3d(roller_train,    scalers_inputs["roller_x"])
force_x_train_std   = fit_transform_3d(force_x_train,   scalers_inputs["force_x"])
force_val_train_std = fit_transform_3d(force_val_train, scalers_inputs["force_values"])
node_train_std      = fit_transform_3d(node_train,      scalers_inputs["node_positions"])

# **Corrected**: Transform validation data using the already fitted scalers
roller_val_std    = scalers_inputs["roller_x"].transform(roller_val.reshape(-1, roller_val.shape[-1])).reshape(roller_val.shape)
force_x_val_std   = scalers_inputs["force_x"].transform(force_x_val.reshape(-1, force_x_val.shape[-1])).reshape(force_x_val.shape)
force_val_val_std = scalers_inputs["force_values"].transform(force_val_val.reshape(-1, force_val_val.shape[-1])).reshape(force_val_val.shape)
node_val_std      = scalers_inputs["node_positions"].transform(node_val.reshape(-1, node_val.shape[-1])).reshape(node_val.shape)

# Prepare final input features by merging sub-features
X_train_3d = merge_sub_features(
    roller_train_std,
    force_x_train_std,
    force_val_train_std,
    node_train_std
)

X_val_3d = merge_sub_features(
    roller_val_std,
    force_x_val_std,
    force_val_val_std,
    node_val_std
)

# Flatten the feature dimension for FNN
X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], -1)  # Shape: (B, n_cases * feat_dim)
X_val_flat = X_val_3d.reshape(X_val_3d.shape[0], -1)        # Shape: (B, n_cases * feat_dim)

# Unify the label by aggregating across cases
Y_train_2d = unify_label_with_c(I_train, c=c)   # Shape: (B, n_elem)
Y_val_2d   = unify_label_with_c(I_val,   c=c)   # Shape: (B, n_elem)

# Fit scaler_Y on training targets
scaler_Y.fit(Y_train_2d)

# Transform targets
Y_train_std = scaler_Y.transform(Y_train_2d)    # Shape: (B, n_elem)
Y_val_std   = scaler_Y.transform(Y_val_2d)      # Shape: (B, n_elem)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_std,       dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_flat,   dtype=torch.float32)
Y_val_tensor   = torch.tensor(Y_val_std,         dtype=torch.float32)

min_constraint = torch.min(Y_train_tensor)
max_constraint = torch.max(Y_train_tensor)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   Y_val_tensor)

print("X_train_flat shape:", X_train_flat.shape,  # e.g. (10000, 10*125)
      "Y_train_2d shape:", Y_train_2d.shape,    # => (10000, 100)
      "\nBut after padding, X_train_flat shape:",
      X_train_flat.shape,                       # e.g. (10000, 1250)
     )

#######################################
# 3) DEFINE THE MODEL COMPONENTS
#######################################

class ResidualBlock(nn.Module):
    """
    A standard residual block with two linear layers and a skip connection.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.PReLU = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.PReLU(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out += residual
        out = self.norm(out)
        out = self.PReLU(out)
        return out

class FNNWithResidual(nn.Module):
    """
    Feedforward Neural Network with Residual Blocks.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_residual_blocks,
        output_dim,
        dropout_rate
    ):
        super(FNNWithResidual, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.PReLU = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate) 
            for _ in range(num_residual_blocks)
        ])
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.input_fc(x)
        out = self.PReLU(out)
        out = self.dropout(out)
        for block in self.residual_blocks:
            out = block(out)
        out = self.output_fc(out)
        return out

#######################################
# 4) DEFINE CUSTOM LOSS
#######################################

class TrainableL1L2Loss(nn.Module):
    """
    Combines L1 and L2 loss with a trainable alpha parameter and
    penalizes predictions outside [min_constraint, max_constraint].
    """
    def __init__(
        self,
        initial_alpha=initial_alpha,
        min_constraint=min_constraint,
        max_constraint=max_constraint,
        penalty_weight=box_constraint_coeff
    ):
        super().__init__()
        # Initialize alpha as a learnable parameter
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32, requires_grad=True))
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.min_constraint = min_constraint
        self.max_constraint = max_constraint
        self.penalty_weight = penalty_weight

    def forward(self, preds, targets):
        """
        Compute the combined loss.
        
        Parameters:
        - preds: Predicted tensor
        - targets: Target tensor
        
        Returns:
        - total_loss: Scalar tensor representing the loss
        """
        # Clamp alpha to avoid extreme weighting
        alpha = torch.clamp(self.alpha, 1e-6, 1.0)
        
        # Compute L1 and L2 losses
        l1_loss = self.l1(preds, targets)
        l2_loss = self.l2(preds, targets)

        # Initialize penalty
        penalty = 0.0
        if self.min_constraint is not None:
            # Penalize predictions below the minimum constraint
            below_min_penalty = torch.sum(torch.relu(self.min_constraint - preds))
            penalty += below_min_penalty
        if self.max_constraint is not None:
            # Penalize predictions above the maximum constraint
            above_max_penalty = torch.sum(torch.relu(preds - self.max_constraint))
            penalty += above_max_penalty

        # Combine losses with trainable alpha and add penalty
        total_loss = alpha * l1_loss + (1 - alpha) * l2_loss + self.penalty_weight * penalty
        return total_loss

def permute_data(X, Y):
    """
    Permutes the data indices for both X and Y consistently.
    
    Parameters:
    - X (torch.Tensor): The input data tensor.
    - Y (torch.Tensor): The target data tensor.
    
    Returns:
    - X_permuted (torch.Tensor): The permuted input data.
    - Y_permuted (torch.Tensor): The permuted target data.
    """
    assert X.size(0) == Y.size(0), "X and Y must have the same number of samples to permute."
    
    # Generate a random permutation of indices
    perm = torch.randperm(X.size(0), device=X.device)
    
    # Apply the permutation
    X_permuted = X[perm]
    Y_permuted = Y[perm]
    
    return X_permuted, Y_permuted

#######################################
# 5) INITIALIZE & TRAIN
#######################################

# Determine input dimension for FNN
input_dim = X_train_flat.shape[1]
output_dim = nelem  # As defined earlier

# Initialize the model
model = FNNWithResidual(
    input_dim=input_dim,
    hidden_dim=hidden_units,
    num_residual_blocks=4,  # Number of residual blocks can be adjusted
    output_dim=output_dim,
    dropout_rate=dropout_rate
).to(device)

# Initialize optimizer, scheduler, and loss criterion
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ExponentialLR(optimizer, gamma=gamma)
criterion = TrainableL1L2Loss()

# Initialize DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# Initialize AMP scaler for mixed precision training
scaler_amp = GradScaler()  # 'cuda' device is inferred automatically

# Setup live plotting for training progress
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

def live_plot(epoch, train_losses, val_losses):
    """
    Updates the training and validation loss plot during training.
    
    Parameters:
    - epoch: Current epoch number
    - train_losses: List of training losses up to current epoch
    - val_losses: List of validation losses up to current epoch
    """
    ax.clear()
    ax.plot(range(1, epoch + 1), train_losses, label="Train Loss", marker='o', color='blue')
    ax.plot(range(1, epoch + 1), val_losses, label="Validation Loss", marker='x', color='red')

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.pause(0.01)

# Initialize lists to store losses
train_losses = []
val_losses = []

best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    noise_level = sigma_0 * (gamma_noise ** epoch)  # Decaying noise level

    total_train_loss = 0.0
    t0 = time.time()

    for Xb, Yb in train_loader:
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        
        # Permute Xb and Yb consistently
        Xb, Yb = permute_data(Xb, Yb)

        # Add optional Gaussian noise
        Xb_noisy = Xb + torch.randn_like(Xb) * noise_level

        optimizer.zero_grad()
        with autocast():
            preds = model(Xb_noisy)  # Predictions: (B, n_elem)
            # Mild penalty on deviation of alpha from its initial value
            L_alpha = (initial_alpha - criterion.alpha) ** 2
            loss = criterion(preds, Yb) + L_alpha

        # Backpropagation with mixed precision
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store the training loss

    # Validation phase
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad(), autocast():
        for Xb, Yb in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            preds = model(Xb)
            val_loss = criterion(preds, Yb)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # Store the validation loss
    scheduler.step()                # Update learning rate

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model_fnn_residual.pth")  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    dt = time.time() - t0
    print(f"Epoch {epoch}/{num_epochs} | "
          f"Train Loss={avg_train_loss:.6f}, "
          f"Val Loss={avg_val_loss:.6f}, "
          f"Time={dt:.2f}s")
    
    # Update live plot
    live_plot(epoch, train_losses, val_losses)



#######################################
# 6) EVALUATION
#######################################

# Load the best model
model.load_state_dict(torch.load("best_model_fnn_residual.pth", map_location=device))
model.eval()

# Create a DataLoader for evaluation
val_loader_eval = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
all_preds, all_labels = [], []

# Collect all predictions and labels
with torch.no_grad():
    for Xb, Yb in val_loader_eval:
        Xb = Xb.to(device)
        preds = model(Xb)  # Predictions: (B, n_elem)
        all_preds.append(preds.cpu())
        all_labels.append(Yb)

# Concatenate all predictions and labels
all_preds  = torch.cat(all_preds, dim=0).numpy()  # Shape: (N, n_elem)
all_labels = torch.cat(all_labels, dim=0).numpy() # Shape: (N, n_elem)

# Un-standardize using scaler_Y
all_preds_unstd  = scaler_Y.inverse_transform(all_preds)
all_labels_unstd = scaler_Y.inverse_transform(all_labels)

# Clip predictions and labels to a reasonable range if necessary
all_preds_unstd  = np.clip(all_preds_unstd,  0.0, 1e10)
all_labels_unstd = np.clip(all_labels_unstd, 0.0, 1e10)

# Compute R² score to evaluate model performance
r2_val = r2_score(all_labels_unstd.ravel(), all_preds_unstd.ravel())
print(f"R² on Validation: {r2_val:.4f}")




#######################################
# 7) EXAMPLE INFERENCE & PLOT
#######################################

# Example User Inputs
L_beam = 200  # Beam length (m)
Fmin_user = -355857  # Min point load to be randomized (N)
Fmax_user = Fmin_user / 10  # Max point load to be randomized (N)
user_rollers = [2*9, 2*29, 2*69, 2*85, 2*100] # Roller locations (m) ## if using data from multi-core generator, subtract 1 from the node number

def build_user_input_no_agg(
    roller_list, force_x_list, force_val_list, node_pos_list,
    scalers, n_cases,
    max_lengths
):
    feat_3d = scale_user_inputs(
        roller_list, force_x_list, force_val_list, node_pos_list,
        scalers, n_cases, max_lengths
    )
    feat_flat = feat_3d.reshape(1, -1)  # Flatten for FNN
    return feat_flat

# Assign the same rollers to each case
user_roller = [user_rollers.copy() for _ in range(n_cases)]

# Generate diverse force positions and values for each case
user_force_x = []
user_force_vals = []
for _ in range(n_cases):
    num_forces = random.randint(1, 3)  # Each case has 1 to 3 forces
    fx = sorted([random.uniform(0, L_beam) for _ in range(num_forces)])
    fv = [random.uniform(Fmin_user, Fmax_user) for _ in range(num_forces)]
    user_force_x.append(fx)
    user_force_vals.append(fv)

# Node positions remain consistent across cases
user_node_pos = [
    np.linspace(0, L_beam, nelem + 1).tolist() for _ in range(n_cases)
]

# Scale inputs for prediction using the correct function name
X_user_flat = build_user_input_no_agg(
    user_roller, user_force_x, user_force_vals, user_node_pos, 
    scalers_inputs, n_cases, max_lengths
)

X_user_t = torch.tensor(X_user_flat, dtype=torch.float32).to(device)

# Perform inference
model.eval()
with torch.no_grad():
    pred_1x = model(X_user_t)  # Predictions: (1, n_elem)

# Un-standardize predictions
pred_1x_np = pred_1x.cpu().numpy().squeeze()
pred_1x_unstd = scaler_Y.inverse_transform(pred_1x_np.reshape(1, -1)).squeeze()

# Visualization Parameters
unique_rollers = sorted(set([x for sublist in user_roller for x in sublist] + [L_beam]))
case_colors = sns.color_palette("Set1", n_colors=n_cases)
case_labels = [f'Force Case {i+1} (N)' for i in range(n_cases)]

beam_y = 0
beam_x = [0, L_beam]
beam_y_vals = [beam_y, beam_y]

pin_x = 0
pin_y = beam_y  # Align the pin vertically with the beam
pin_size = 0.25 # Increased size for better visibility
pin_triangle = RegularPolygon(
    (pin_x, pin_y - 0.75*pin_size),  # Position the pin slightly below the beam
    numVertices=3,
    radius=pin_size,
    orientation=np.pi/0.5,  # Pointing upwards
    color='red',
    label='Pin'
)

# Collect force positions and values for plotting
force_positions = []
force_vals_plot = []
for fx, fv in zip(user_force_x, user_force_vals):
    for xx, val in zip(fx, fv):
        force_positions.append(xx)
        force_vals_plot.append(val)

all_force_vals = force_vals_plot
max_force = max(abs(val) for val in all_force_vals) if all_force_vals else 1.0
desired_max_arrow_length = 2.0
arrow_scale = desired_max_arrow_length / max_force if max_force != 0 else 1.0

beam_positions = user_node_pos[0][:nelem]

# Normalize I for visualization purposes using a colormap
I_normalized = (pred_1x_unstd - pred_1x_unstd.min()) / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8)
cmap = cm.winter  # Changed from cm.viridis to cm.winter
norm = plt.Normalize(pred_1x_unstd.min(), pred_1x_unstd.max())

# Define block dimensions
block_width = L_beam / nelem * 0.8  # Slightly narrower blocks for better visibility
block_height = 1  # Maximum height for visualization

# Create a single plot with extra space for the colorbar
fig, ax = plt.subplots(figsize=(18, 7))

# Plot Beam
ax.plot(beam_x, beam_y_vals, color='black', linewidth=3, label='Beam')
ax.add_patch(pin_triangle)

# Plot Rollers
ax.scatter(unique_rollers, [beam_y]*len(unique_rollers),
           marker='o', color='seagreen', s=200,
           label='Rollers', zorder=5, edgecolors='k')

# Plot Forces
for case_idx in range(n_cases):
    fx_list = user_force_x[case_idx]
    fv_list = user_force_vals[case_idx]
    color = case_colors[case_idx]
    label = case_labels[case_idx]

    for idx, (fx, fv) in enumerate(zip(fx_list, fv_list)):
        arrow_length = abs(fv) * arrow_scale
        start_point = (fx, beam_y + arrow_length)
        end_point   = (fx, beam_y)

        arrow = FancyArrowPatch(
            posA=start_point, posB=end_point,
            arrowstyle='-|>',
            mutation_scale=20,
            color=color,
            linewidth=2,
            alpha=0.8,
            label=label if idx == 0 else ""
        )
        ax.add_patch(arrow)
        ax.text(fx, beam_y + arrow_length + desired_max_arrow_length * 0.02,
                f"{fv:.0f}", ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold')

# Plot Moments of Inertia (I) as semi-translucent blocks centered on the beam
for idx, (x_pos, I_val) in enumerate(zip(beam_positions, pred_1x_unstd)):
    # Normalize I for color mapping
    color = cmap(norm(I_val))
    
    # Define the rectangle parameters
    rect_x = x_pos - block_width / 2  # Center the rectangle horizontally
    rect_y = beam_y - (I_val / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8)) * block_height / 2  # Center vertically
    
    rect = Rectangle((rect_x, rect_y), block_width, (I_val / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8)) * block_height,
                     linewidth=0, edgecolor=None,
                     facecolor=color, alpha=0.6)
    ax.add_patch(rect)

# Create a ScalarMappable for the colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib

# Add the colorbar to the figure (vertically to the right)
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Predicted I (m$^4$)', fontsize=16)
cbar.ax.tick_params(labelsize=10)

# Set plot titles and labels
ax.set_title("Beam Setup with Applied Forces and Predicted I",
             fontsize=22, fontweight='bold', pad=20)
ax.set_xlabel("Beam Length (m)", fontsize=16, fontweight='semibold')
#ax.set_ylabel("Force Representation and Moments of Inertia, I (m$^4$)", fontsize=16, fontweight='semibold')
ax.set_xlim(-5, L_beam + 5)
ax.set_ylim(-2.5, 2.5)  # Adjusted to accommodate I blocks
ax.set_xticks(np.arange(0, L_beam + 5, 5))
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Create custom legend
legend_elements = [
    Line2D([0], [0], color='black', lw=3, label='Beam'),
    Line2D([0], [0], marker=(3, 0, -90), color='red', label='Pin',
           markerfacecolor='red', markersize=15),
    Line2D([0], [0], marker='o', color='seagreen', label='Rollers',
           markerfacecolor='seagreen', markeredgecolor='k', markersize=15),
]

for color, label in zip(case_colors, case_labels):
    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))
    
# Remove the 'Predicted I' legend entry since the colorbar now represents it
# legend_elements.append(Rectangle((0,0),1,1, facecolor='green', alpha=0.6, label='Predicted I'))

ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

plt.tight_layout()
plt.show()
