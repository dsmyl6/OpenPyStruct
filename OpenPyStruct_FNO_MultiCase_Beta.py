############################################################################
#### OpenPyStruct FNO-Based Multi Load Case Optimizer                   ####
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
from matplotlib.patches import FancyArrowPatch, Rectangle
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
from torch.amp import autocast, GradScaler

#######################################
# 1) CONFIGURATION & HYPERPARAMETERS
#######################################

# Model and training configuration #
n_cases = 6                 # Number of sub-cases per sample
nelem = 100                 # Final output dimension per sample: (B, n_elem)
box_constraint_coeff = 5e-1 # Coefficient for box constraint penalty
hidden_units = 512          # Number of hidden units in final MLP
dropout_rate = 0.1          # Dropout rate for regularization
num_fno_layers = 4          # Number of FNO layers (analogous to "blocks")
num_epochs = 500            # Maximum number of training epochs
batch_size = 512            # Batch size for training
patience = 10               # Early stopping patience
learning_rate = 3e-3        # Learning rate for optimizer
weight_decay = 1e-6         # Weight decay (L2 regularization) for optimizer
train_split = 0.8           # Fraction of data used for training
sigma_0 = 0.01              # Initial Gaussian noise for input
gamma_noise = 0.95          # Decay rate for noise during training
gamma = 0.975               # Learning rate scheduler decay rate
initial_alpha = 0.5         # Initial alpha value for loss weighting
c = 0.5                     # Parameter to adjust label aggregation

# FNO hyperparameters
fno_modes = 4               # Adjusted to satisfy modes <= n_cases//2 + 1 (which is 4 for n_cases=6)
fno_width = 128              # Channel width (hidden size) in FNO layers

# For optional feature-dimension padding
# (Typically for Transformers, but we leave it at 1 for "no-op" padding in FNO.)
nheads_for_padding = 1

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

def unify_label_with_c(I_3d, c=0.5):
    """
    Aggregate labels by computing the mean across cases and adding
    c times the standard deviation.
    
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
    flat = arr_3d.reshape(B * NC, M)
    scaled = scaler.fit_transform(flat)
    return scaled.reshape(B, NC, M)

def transform_3d(arr_3d, scaler):
    """
    Transform (but do not fit) a 3D array using the provided scaler,
    maintaining (B, NC, M) shape.
    
    Parameters:
    - arr_3d: NumPy array of shape (B, NC, M)
    - scaler: Fitted scaler instance
    
    Returns:
    - scaled_arr: NumPy array of shape (B, NC, M)
    """
    B, NC, M = arr_3d.shape
    flat = arr_3d.reshape(B * NC, M)
    scaled = scaler.transform(flat)
    return scaled.reshape(B, NC, M)

def merge_sub_features(*arrays):
    """
    Concatenate multiple feature arrays along the feature dimension.
    e.g., if each array is (B, NC, F?), we get (B, NC, sum_of_F).
    """
    return np.concatenate(arrays, axis=2)

def pad_feat_dim_to_multiple_of_nheads(X_3d, nheads=1):
    """
    Pad the feature dimension to be a multiple of 'nheads'.
    For FNO, this is not strictly necessary, but left here as a no-op
    if nheads=1.
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
        r_pad  = pad_to_length(user_roller[i],   max_lengths['roller_x'])
        fx_pad = pad_to_length(user_force_x[i],  max_lengths['force_x'])
        fv_pad = pad_to_length(user_force_vals[i], max_lengths['force_values'])
        nd_pad = pad_to_length(user_node_pos[i], max_lengths['node_positions'])

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
    with open("StructDataMedium.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("The file 'StructDataMedium.json' was not found.")

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
    "roller_x":       max(len(r) for r in roller_x)       if roller_x       else 0,
    "force_x":        max(len(r) for r in force_x)        if force_x        else 0,
    "force_values":   max(len(r) for r in force_values)   if force_values   else 0,
    "node_positions": max(len(r) for r in node_positions) if node_positions else 0,
    "I_values":       max(len(r) for r in I_values)       if I_values       else 0
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

# Transform validation data (without refit)
roller_val_std    = transform_3d(roller_val,    scalers_inputs["roller_x"])
force_x_val_std   = transform_3d(force_x_val,   scalers_inputs["force_x"])
force_val_val_std = transform_3d(force_val_val, scalers_inputs["force_values"])
node_val_std      = transform_3d(node_val,      scalers_inputs["node_positions"])

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

# (Optional) Pad feature dimensions
X_train_3d_padded, feat_dim_padded = pad_feat_dim_to_multiple_of_nheads(X_train_3d, nheads=nheads_for_padding)
X_val_3d_padded, _                 = pad_feat_dim_to_multiple_of_nheads(X_val_3d,   nheads=nheads_for_padding)

# Unify the label by aggregating across cases
Y_train_2d = unify_label_with_c(I_train, c=c)   # Shape: (B, n_elem)
Y_val_2d   = unify_label_with_c(I_val,   c=c)   # Shape: (B, n_elem)

# Fit scaler_Y on training targets
scaler_Y.fit(Y_train_2d)

# Transform targets
Y_train_std = scaler_Y.transform(Y_train_2d)  # (B, n_elem)
Y_val_std   = scaler_Y.transform(Y_val_2d)    # (B, n_elem)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_3d_padded, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_std,       dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_3d_padded,   dtype=torch.float32)
Y_val_tensor   = torch.tensor(Y_val_std,         dtype=torch.float32)

min_constraint = torch.min(Y_train_tensor)
max_constraint = torch.max(Y_train_tensor)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   Y_val_tensor)

print("X_train_3d shape:", X_train_3d.shape,
      "Y_train_2d shape:", Y_train_2d.shape,
      "\nAfter optional padding => X_train_3d_padded shape:",
      X_train_3d_padded.shape)


#######################################
# 3) DEFINE THE FNO MODEL
#######################################

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. Performs Fourier transform along the last dimension.
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Real & Imag parts of the Fourier coefficients
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes)
        )

    def forward(self, x):
        """
        x: (B, in_channels, n)
        """
        B, inC, n = x.shape
        x_ft = torch.fft.rfft(x, n=n)  # => (B, inC, n//2+1)

        # Ensure that modes do not exceed the available frequency components
        actual_modes = min(self.modes, x_ft.shape[-1])
        if actual_modes < self.modes:
            print(f"Adjusted modes from {self.modes} to {actual_modes} based on signal length {n}.")
        
        x_ft = x_ft[:, :, :actual_modes]

        # Adjust weights if actual_modes < self.modes
        if actual_modes < self.modes:
            w_r = self.weights_real[:, :, :actual_modes].unsqueeze(0)  # => (1, inC, outC, actual_modes)
            w_i = self.weights_imag[:, :, :actual_modes].unsqueeze(0)
        else:
            w_r = self.weights_real.unsqueeze(0)  # => (1, inC, outC, modes)
            w_i = self.weights_imag.unsqueeze(0)

        x_ft_real = x_ft.real
        x_ft_imag = x_ft.imag

        # Perform complex multiplication
        out_ft_real = torch.einsum("bim, iojm -> bojm", x_ft_real, w_r) - \
                      torch.einsum("bim, iojm -> bojm", x_ft_imag, w_i)
        out_ft_imag = torch.einsum("bim, iojm -> bojm", x_ft_real, w_i) + \
                      torch.einsum("bim, iojm -> bojm", x_ft_imag, w_r)

        # Sum over input channels
        out_ft_real = out_ft_real.sum(dim=2)  # => (B, outC, modes)
        out_ft_imag = out_ft_imag.sum(dim=2)  # => (B, outC, modes)

        out_ft = torch.complex(out_ft_real, out_ft_imag)

        # Pad back to original size in frequency domain
        pad_size = (0, (n//2 + 1) - actual_modes)
        if pad_size[1] > 0:
            out_ft = nn.functional.pad(out_ft, pad_size)

        # Inverse FFT to get back to spatial domain
        x_out = torch.fft.irfft(out_ft, n=n)  # => (B, outC, n)
        return x_out

class FNOBlock1d(nn.Module):
    """
    One block of a 1D FNO: 
    - SpectralConv1d
    - Pointwise Conv
    - BatchNorm + Activation
    """
    def __init__(self, width, modes):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.bn = nn.BatchNorm1d(width)

    def forward(self, x):
        """
        x shape: (B, width, n)
        """
        x1 = self.conv(x)
        x2 = self.w(x)
        x_out = x1 + x2
        x_out = self.bn(x_out)
        return nn.functional.gelu(x_out)

class FNO1dModel(nn.Module):
    """
    An FNO that operates on dimension = n_cases as the 1D dimension.
    Then flattens and passes an MLP to get final predictions of size (n_elem).
    """
    def __init__(
        self,
        n_cases,
        feat_dim,
        n_elem,
        fno_modes,
        fno_width,
        num_fno_layers=4,
        hidden_units=512,
        dropout=0.1
    ):
        super().__init__()
        self.n_cases = n_cases
        self.feat_dim = feat_dim
        self.n_elem = n_elem
        self.modes = fno_modes
        self.width = fno_width
        self.num_fno_layers = num_fno_layers

        # Map input feat_dim -> fno_width
        self.fc0 = nn.Linear(feat_dim, fno_width)

        # Multiple FNO blocks
        self.fno_blocks = nn.ModuleList(
            [FNOBlock1d(fno_width, fno_modes) for _ in range(num_fno_layers)]
        )

        # Output MLP: (fno_width*n_cases) -> hidden_units -> n_elem
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(fno_width * n_cases, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, n_elem),
        )

    def forward(self, x):
        """
        x shape: (B, n_cases, feat_dim) -> (B, n_elem)
        """
        B, Nc, Fdim = x.shape
        assert Nc == self.n_cases and Fdim == self.feat_dim, \
            f"Input shape {x.shape} does not match (B, {self.n_cases}, {self.feat_dim})."

        # Reshape to (B, feat_dim, n_cases)
        x = x.permute(0, 2, 1)  # => (B, feat_dim, n_cases)

        # Map feat_dim -> fno_width
        # fc0 expects shape (B*n_cases, feat_dim), so we transpose last two dims
        x = self.fc0(x.transpose(-1, -2))   # => (B, n_cases, fno_width)
        x = x.transpose(-1, -2)             # => (B, fno_width, n_cases)

        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)

        # Flatten to (B, fno_width*n_cases)
        x = x.reshape(B, -1)
        x = self.dropout(x)

        # Final MLP
        out = self.fc_out(x)  # => (B, n_elem)
        return out

#######################################
# 4) DEFINE CUSTOM LOSS
#######################################

class TrainableL1L2Loss(nn.Module):
    """
    Combines L1 and L2 loss with a trainable alpha parameter and
    a penalty for predictions outside [min_constraint, max_constraint].
    """
    def __init__(
        self,
        initial_alpha=0.5,
        min_constraint=None,
        max_constraint=None,
        penalty_weight=0.5
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32, requires_grad=True))
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.min_constraint = min_constraint
        self.max_constraint = max_constraint
        self.penalty_weight = penalty_weight

    def forward(self, preds, targets):
        alpha_clamped = torch.clamp(self.alpha, 1e-6, 1.0)
        
        l1_loss = self.l1(preds, targets)
        l2_loss = self.l2(preds, targets)

        penalty = 0.0
        if self.min_constraint is not None:
            below_min_penalty = torch.sum(torch.relu(self.min_constraint - preds))
            penalty += below_min_penalty
        if self.max_constraint is not None:
            above_max_penalty = torch.sum(torch.relu(preds - self.max_constraint))
            penalty += above_max_penalty

        total_loss = alpha_clamped * l1_loss + (1 - alpha_clamped) * l2_loss
        total_loss += self.penalty_weight * penalty
        return total_loss

def permute_data(X, Y):
    """
    Permutes data indices for both X and Y consistently.
    """
    assert X.size(0) == Y.size(0), "X and Y must have same batch size."
    perm = torch.randperm(X.size(0), device=X.device)
    return X[perm], Y[perm]

#######################################
# 5) INITIALIZE & TRAIN
#######################################

# Initialize the FNO model
model = FNO1dModel(
    n_cases=n_cases,
    feat_dim=feat_dim_padded,  # after optional padding
    n_elem=nelem,
    fno_modes=fno_modes,
    fno_width=fno_width,
    num_fno_layers=num_fno_layers,
    hidden_units=hidden_units,
    dropout=dropout_rate
).to(device)

# Initialize optimizer, scheduler, and loss criterion
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ExponentialLR(optimizer, gamma=gamma)
criterion = TrainableL1L2Loss(
    min_constraint=min_constraint,
    max_constraint=max_constraint,
    penalty_weight=box_constraint_coeff
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# Initialize GradScaler for AMP (Not using AMP due to FFT constraints)
# If you decide to use AMP in future, ensure FFT dimensions are powers of two
scaler_amp = GradScaler()

# Live Plotting Setup
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0

def live_plot(epoch, train_losses, val_losses):
    ax.clear()
    ax.plot(range(1, epoch + 1), train_losses, label="Train Loss", marker='o', color='blue')
    ax.plot(range(1, epoch + 1), val_losses, label="Validation Loss", marker='x', color='red')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.pause(0.01)

# Training Loop
for epoch in range(1, num_epochs + 1):
    model.train()
    noise_level = sigma_0 * (gamma_noise ** epoch)

    total_train_loss = 0.0
    t0 = time.time()

    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)

        # Optional permutation
        Xb, Yb = permute_data(Xb, Yb)

        # Add Gaussian noise
        Xb_noisy = Xb + torch.randn_like(Xb) * noise_level

        optimizer.zero_grad()
        # Disable AMP to avoid FFT precision issues
        with torch.cuda.amp.autocast(enabled=False):
            preds = model(Xb_noisy)
            L_alpha = (initial_alpha - criterion.alpha) ** 2
            loss = criterion(preds, Yb) + L_alpha

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            with torch.cuda.amp.autocast(enabled=False):
                preds = model(Xb)
                val_loss = criterion(preds, Yb)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step()

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model_fno.pth")
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

    live_plot(epoch, train_losses, val_losses)

#######################################
# 6) EVALUATION
#######################################

# Load the best model
model.load_state_dict(torch.load("best_model_fno.pth", map_location=device))
model.eval()

# Evaluation on Validation Set
val_loader_eval = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
all_preds, all_labels = [], []

with torch.no_grad():
    for Xb, Yb in val_loader_eval:
        Xb = Xb.to(device)
        preds = model(Xb)
        all_preds.append(preds.cpu())
        all_labels.append(Yb)

all_preds  = torch.cat(all_preds, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

# Inverse transform to original scale
all_preds_unstd  = scaler_Y.inverse_transform(all_preds)
all_labels_unstd = scaler_Y.inverse_transform(all_labels)

# Example clipping (adjust based on domain knowledge)
all_preds_unstd  = np.clip(all_preds_unstd,  0.0, 1e10)
all_labels_unstd = np.clip(all_labels_unstd, 0.0, 1e10)

# Calculate R² score
r2_val = r2_score(all_labels_unstd.ravel(), all_preds_unstd.ravel())
print(f"R² on Validation: {r2_val:.4f}")

#######################################
# 7) EXAMPLE INFERENCE & PLOT
#######################################

# Example user inputs
L_beam = 200
Fmin_user = -355857
Fmax_user = Fmin_user / 10
user_rollers = [2*9, 2*29, 2*69, 2*85, 2*100]

def build_user_input_no_agg(
    roller_list, force_x_list, force_val_list, node_pos_list,
    scalers, n_cases, max_lengths
):
    """
    Build user input without aggregation by scaling and padding.
    """
    feat_3d = scale_user_inputs(
        roller_list, force_x_list, force_val_list, node_pos_list,
        scalers, n_cases, max_lengths
    )
    return feat_3d

# Create random multi-case loads
user_roller = [user_rollers.copy() for _ in range(n_cases)]
user_force_x = []
user_force_vals = []
for _ in range(n_cases):
    num_forces = random.randint(1, 3)
    fx = sorted([random.uniform(0, L_beam) for _ in range(num_forces)])
    fv = [random.uniform(Fmin_user, Fmax_user) for _ in range(num_forces)]
    user_force_x.append(fx)
    user_force_vals.append(fv)

user_node_pos = [
    np.linspace(0, L_beam, nelem + 1).tolist() for _ in range(n_cases)
]

# Build user input
X_user_3d = build_user_input_no_agg(
    user_roller, user_force_x, user_force_vals, user_node_pos,
    scalers_inputs, n_cases, max_lengths
)

# Optional padding
X_user_3d_padded, _ = pad_feat_dim_to_multiple_of_nheads(X_user_3d, nheads=nheads_for_padding)
X_user_t = torch.tensor(X_user_3d_padded, dtype=torch.float32).to(device)

# Predict
model.eval()
with torch.no_grad():
    pred_1x = model(X_user_t)  # => (1, n_elem)

pred_1x_np = pred_1x.cpu().numpy().squeeze()
pred_1x_unstd = scaler_Y.inverse_transform(pred_1x_np.reshape(1, -1)).squeeze()

# ------------------ PLOT ------------------
unique_rollers = sorted(set([x for sublist in user_roller for x in sublist] + [L_beam]))
case_colors = sns.color_palette("Set1", n_colors=n_cases)
case_labels = [f'Force Case {i+1} (N)' for i in range(n_cases)]

beam_y = 0
beam_x = [0, L_beam]
beam_y_vals = [beam_y, beam_y]

force_positions = []
force_vals_plot = []
for fx, fv in zip(user_force_x, user_force_vals):
    for xx, val in zip(fx, fv):
        force_positions.append(xx)
        force_vals_plot.append(val)

max_force = max(abs(val) for val in force_vals_plot) if force_vals_plot else 1.0
desired_max_arrow_length = 2.0
arrow_scale = desired_max_arrow_length / max_force if max_force != 0 else 1.0

beam_positions = user_node_pos[0][:nelem]

I_normalized = (pred_1x_unstd - pred_1x_unstd.min()) / (
    pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8
)
cmap = cm.winter
norm = plt.Normalize(pred_1x_unstd.min(), pred_1x_unstd.max())

block_width = L_beam / nelem * 0.8
block_height = 1

fig, ax = plt.subplots(figsize=(18, 7))

# Plot beam
ax.plot(beam_x, beam_y_vals, color='black', linewidth=3, label='Beam')
ax.scatter(beam_x[0], beam_y - 0.15, marker='^', color='red', s=300, zorder=6)

# Plot rollers
ax.scatter(unique_rollers, [beam_y]*len(unique_rollers),
           marker='o', color='seagreen', s=200,
           label='Rollers', zorder=5, edgecolors='k')

# Plot forces
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

# Plot predicted I as rectangles
for idx, (x_pos, I_val) in enumerate(zip(beam_positions, pred_1x_unstd)):
    color = cmap(norm(I_val))
    rect_x = x_pos - block_width / 2
    rect_y = beam_y - (
        I_val / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8)
    ) * block_height / 2

    rect = Rectangle(
        (rect_x, rect_y),
        block_width,
        (I_val / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8)) * block_height,
        linewidth=0,
        facecolor=color,
        alpha=0.6
    )
    ax.add_patch(rect)

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Predicted I (m$^4$)', fontsize=16)
cbar.ax.tick_params(labelsize=10)

ax.set_title("Beam Setup with Applied Forces (FNO-Predicted I)",
             fontsize=22, fontweight='bold', pad=20)
ax.set_xlabel("Beam Length (m)", fontsize=16, fontweight='semibold')
ax.set_xlim(-5, L_beam + 5)
ax.set_ylim(-2.5, 2.5)
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

ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

plt.tight_layout()
plt.show()
