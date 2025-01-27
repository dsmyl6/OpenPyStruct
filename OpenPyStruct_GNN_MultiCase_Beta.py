############################################################################
#### OpenPyStruct Chain GNN With Speedups                               ####
#### Coder: Danny Smyl, PhD, PE, Georgia Tech, 2025                     ####
############################################################################


#### EXPERIMENTAL CODE (WORKS BUT NOT OPTIMIZED ####

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import mode

# For faster adjacency multiplication
from einops import rearrange
# For Mixed Precision
from torch.cuda.amp import autocast, GradScaler

#######################################
# 1) CONFIGURATION & HYPERPARAMETERS
#######################################

n_cases = 6
nelem = 100
box_constraint_coeff = 5e-1
encoder_hidden_dim = 128   
gnn_hidden_dim = 128       
num_gnn_layers = 2         
dropout_rate = 0.5
num_epochs = 500       
batch_size = 512
patience = 10
learning_rate = 3e-3
weight_decay = 1e-2
train_split = 0.8
sigma_0 = 0.01
gamma_noise = 0.99
gamma = 0.975
initial_alpha = 0.5
c = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

#######################################
# 2) DATA LOADING & PREPROCESSING
#######################################

def pad_sequences(data_list, max_length, pad_val=0.0):
    out = np.full((len(data_list), max_length), pad_val, dtype=np.float32)
    for i, arr in enumerate(data_list):
        arr_np = np.array(arr, dtype=np.float32)
        length = min(len(arr_np), max_length)
        out[i, :length] = arr_np[:length]
    return out

def unify_label_with_c(I_3d, cval):
    I_mean = I_3d.mean(axis=1) 
    I_std  = I_3d.std(axis=1)
    return I_mean + cval * I_std

def fit_transform_3d(arr_3d, scaler):
    B, NC, M = arr_3d.shape
    flat = arr_3d.reshape(B*NC, M)
    scaled = scaler.fit_transform(flat)
    return scaled.reshape(B, NC, M)

def merge_sub_features(*arrays):
    return np.concatenate(arrays, axis=2)

def scale_user_inputs(user_roller, user_force_x, user_force_vals, user_node_pos, 
                      scalers, n_cases, max_lengths):
    """
    Scales a single user input sample, returns shape => (1, n_cases, feat_dim)
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

        sub_feat = np.concatenate([r_scaled, fx_scaled, fv_scaled, nd_scaled])
        feat_arrays.append(sub_feat)

    feat_2d = np.stack(feat_arrays, axis=0)  # (n_cases, total_feat_dim)
    feat_3d = feat_2d[np.newaxis, ...]       # (1, n_cases, total_feat_dim)
    return feat_3d

# Load data
try:
    with open("training_data_PINN_Case_two.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("The file 'training_data_PINN_Case_two.json' was not found.")

roller_x       = data.get("roller_x_locations", [])
force_x        = data.get("force_x_locations", [])
force_values   = data.get("force_values", [])
node_positions = data.get("node_positions", [])
I_values       = data.get("I_values", [])

num_samples = len(I_values)
req_keys = ["roller_x_locations","force_x_locations","force_values","node_positions"]
if not all(len(data.get(k, [])) == num_samples for k in req_keys):
    raise ValueError("Mismatch in sample counts among roller_x, force_x, force_values, node_positions.")

max_lengths = {
    "roller_x":       max(len(r) for r in roller_x)       if roller_x       else 0,
    "force_x":        max(len(r) for r in force_x)        if force_x        else 0,
    "force_values":   max(len(r) for r in force_values)   if force_values   else 0,
    "node_positions": max(len(r) for r in node_positions) if node_positions else 0,
    "I_values":       max(len(r) for r in I_values)       if I_values       else 0
}

# Pad
roller_x_pad  = pad_sequences(roller_x,    max_lengths["roller_x"])
force_x_pad   = pad_sequences(force_x,     max_lengths["force_x"])
force_val_pad = pad_sequences(force_values,max_lengths["force_values"])
node_pos_pad  = pad_sequences(node_positions, max_lengths["node_positions"])
I_values_pad  = pad_sequences(I_values,    max_lengths["I_values"])

# Group by n_cases
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

scalers_inputs = {
    "roller_x":       StandardScaler(),
    "force_x":        StandardScaler(),
    "force_values":   StandardScaler(),
    "node_positions": StandardScaler()
}
scaler_Y = StandardScaler()

roller_train_std    = fit_transform_3d(roller_train,    scalers_inputs["roller_x"])
force_x_train_std   = fit_transform_3d(force_x_train,   scalers_inputs["force_x"])
force_val_train_std = fit_transform_3d(force_val_train, scalers_inputs["force_values"])
node_train_std      = fit_transform_3d(node_train,      scalers_inputs["node_positions"])

roller_val_std    = fit_transform_3d(roller_val,    scalers_inputs["roller_x"])
force_x_val_std   = fit_transform_3d(force_x_val,   scalers_inputs["force_x"])
force_val_val_std = fit_transform_3d(force_val_val, scalers_inputs["force_values"])
node_val_std      = fit_transform_3d(node_val,      scalers_inputs["node_positions"])

# Merge features => (B, n_cases, feat_in)
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

B_train, nc, fin = X_train_3d.shape
B_val,   _,   _  = X_val_3d.shape
enc_in_dim = nc * fin  # flatten sub-cases

# Flatten => (B, enc_in_dim)
X_train_2d = X_train_3d.reshape(B_train, -1)
X_val_2d   = X_val_3d.reshape(B_val,   -1)

Y_train_2d = unify_label_with_c(I_train, c)
Y_val_2d   = unify_label_with_c(I_val,   c)

scaler_Y.fit(Y_train_2d)
Y_train_std = scaler_Y.transform(Y_train_2d)
Y_val_std   = scaler_Y.transform(Y_val_2d)

X_train_tensor = torch.tensor(X_train_2d, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_std, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val_2d,   dtype=torch.float32)
Y_val_tensor   = torch.tensor(Y_val_std,  dtype=torch.float32)

min_constraint = torch.min(Y_train_tensor)
max_constraint = torch.max(Y_train_tensor)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor,   Y_val_tensor)

print(f"Train set shape => X: {X_train_2d.shape}, Y: {Y_train_2d.shape}")
print(f"Val   set shape => X: {X_val_2d.shape}, Y: {Y_val_2d.shape}")


#######################################
# 3) DEFINE A CHAIN GNN WITH SPEEDUPS
#######################################

def precompute_normalized_adjacency(n):
    """
    Returns a precomputed (n, n) adjacency for a chain, normalized by D^{-1/2}AD^{-1/2}.
    """
    A = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n - 1):
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    degrees = A.sum(dim=1)
    D_inv_sqrt = torch.pow(degrees + 1e-8, -0.5)
    # elementwise multiply for normalization => A_hat = D^-1/2 * A * D^-1/2
    A_hat = A * D_inv_sqrt.unsqueeze(0)
    A_hat = A_hat * D_inv_sqrt.unsqueeze(1)
    return A_hat

class GCNLayer(nn.Module):
    """
    Single GCN layer: out = A_hat @ (X W).
    We'll use a single adjacency across the entire batch.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, A_hat):
        """
        x shape => (B, n, in_dim)
        A_hat => (n, n), normalized adjacency
        returns => (B, n, out_dim)
        """
        # Step 1: Wx => (B, n, out_dim)
        Wx = self.linear(x)
        # Step 2: multiply adjacency => out = A_hat @ Wx per sample in batch
        # We'll do => out[b] = A_hat (n,n) @ Wx[b] (n, out_dim)
        # Use einops: A_hat => (n,n), Wx => (B, n, out_dim)
        # out => (B,n,out_dim) => (b,i,d) = sum_j A_hat[i,j]*Wx[b,j,d]
        out = torch.einsum('ij,bjd->bid', A_hat, Wx)
        return out

class ChainGNN(nn.Module):
    """
    1) Encoder MLP => (B, n_elem, gnn_hidden_dim)
    2) Several GCN Layers
    3) Output => (B, n_elem)
    """
    def __init__(self, enc_in_dim, n_elem, enc_hidden_dim, gnn_hidden_dim,
                 num_gnn_layers, dropout):
        super().__init__()
        self.n_elem = n_elem
        self.enc_in_dim = enc_in_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gnn_layers = num_gnn_layers

        # Precompute chain adjacency
        A_hat = precompute_normalized_adjacency(n_elem)
        self.register_buffer("A_hat", A_hat)  # stays on same device as model

        # Encoder: flatten sub-cases => node embeddings
        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, enc_hidden_dim),
            nn.ReLU(),
            nn.Linear(enc_hidden_dim, n_elem * gnn_hidden_dim)
        )

        # GCN stack
        layers = []
        norms = []
        drops = []
        in_dim = gnn_hidden_dim
        for _ in range(num_gnn_layers):
            layers.append(GCNLayer(in_dim, in_dim))  # keep dimension stable
            norms.append(nn.LayerNorm(in_dim))
            drops.append(nn.Dropout(dropout))
        self.gcn_layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.drops = nn.ModuleList(drops)

        # Output readout => (n_elem, 1)
        self.out_layer = nn.Linear(in_dim, 1)

    def forward(self, x):
        """
        x => (B, enc_in_dim)
        returns => (B, n_elem)
        """
        B = x.size(0)
        # 1) Encode => (B, n_elem * gnn_hidden_dim)
        enc = self.encoder(x)
        node_feats = enc.view(B, self.n_elem, self.gnn_hidden_dim)

        # 2) GCN layers
        out = node_feats
        for i, gcn in enumerate(self.gcn_layers):
            out_in = self.norms[i](out)
            out_g = gcn(out_in, self.A_hat)  # (B, n, gnn_hidden_dim)
            out = out + self.drops[i](out_g) # residual

        # 3) Output => (B, n_elem, 1) => (B, n_elem)
        out = self.out_layer(out).squeeze(-1)
        return out

#######################################
# 4) CUSTOM LOSS WITH CONSTRAINT PENALTY
#######################################

class TrainableL1L2Loss(nn.Module):
    def __init__(self, initial_alpha, min_constraint, max_constraint, penalty_weight):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32, requires_grad=True))
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.min_constraint = min_constraint
        self.max_constraint = max_constraint
        self.penalty_weight = penalty_weight

    def forward(self, preds, targets):
        alpha = torch.clamp(self.alpha, 1e-6, 1.0)
        l1_loss = self.l1(preds, targets)
        l2_loss = self.l2(preds, targets)

        penalty = 0.0
        if self.min_constraint is not None:
            below_min = torch.sum(torch.relu(self.min_constraint - preds))
            penalty += below_min
        if self.max_constraint is not None:
            above_max = torch.sum(torch.relu(preds - self.max_constraint))
            penalty += above_max

        total_loss = alpha * l1_loss + (1 - alpha) * l2_loss + self.penalty_weight * penalty
        return total_loss


#######################################
# 5) INIT MODEL & TRAIN (SPEEDUPS)
#######################################

model = ChainGNN(
    enc_in_dim       = enc_in_dim,
    n_elem           = nelem,
    enc_hidden_dim   = encoder_hidden_dim,
    gnn_hidden_dim   = gnn_hidden_dim,
    num_gnn_layers   = num_gnn_layers,
    dropout          = dropout_rate
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
criterion = TrainableL1L2Loss(
    initial_alpha  = initial_alpha,
    min_constraint = min_constraint.to(device),
    max_constraint = max_constraint.to(device),
    penalty_weight = box_constraint_coeff
).to(device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

scaler_amp = GradScaler()

# We'll do live plotting every 5 epochs to reduce overhead
plot_frequency = 5
plt.ion()
fig, ax = plt.subplots(figsize=(10,6))

def live_plot(epoch, train_losses, val_losses):
    ax.clear()
    ax.plot(range(1, epoch + 1), train_losses, label="Train Loss", marker='o', color='blue')
    ax.plot(range(1, epoch + 1), val_losses, label="Val Loss", marker='x', color='red')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.pause(0.01)

train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(1, num_epochs+1):
    model.train()
    noise_level = sigma_0 * (gamma_noise ** epoch)
    total_train_loss = 0.0
    t0 = time.time()

    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        # Add optional noise to inputs
        Xb_noisy = Xb + torch.randn_like(Xb) * noise_level

        optimizer.zero_grad()
        with autocast():
            preds = model(Xb_noisy)
            loss = criterion(preds, Yb)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad(), autocast():
        for Xb, Yb in val_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            preds = model(Xb)
            val_loss = criterion(preds, Yb)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_gnn_model.pth")
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

model.load_state_dict(torch.load("best_gnn_model.pth", map_location=device))
model.eval()

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

all_preds_unstd  = scaler_Y.inverse_transform(all_preds)
all_labels_unstd = scaler_Y.inverse_transform(all_labels)

all_preds_unstd  = np.clip(all_preds_unstd,  0.0, 1e10)
all_labels_unstd = np.clip(all_labels_unstd, 0.0, 1e10)

r2_val = r2_score(all_labels_unstd.ravel(), all_preds_unstd.ravel())
print(f"R² on Validation: {r2_val:.4f}")


#######################################
# 7) EXAMPLE INFERENCE & PLOT
#######################################

# Example user input
L_beam = 200
Fmin_user = -355857
Fmax_user = Fmin_user / 10
user_rollers = [2*9, 2*29, 2*69, 2*85, 2*100]
user_roller = [user_rollers for _ in range(n_cases)]

user_force_x = []
user_force_vals = []
for _ in range(n_cases):
    num_forces = random.randint(1,3)
    fx = sorted([random.uniform(0, L_beam) for _ in range(num_forces)])
    fv = [random.uniform(Fmin_user, Fmax_user) for _ in range(num_forces)]
    user_force_x.append(fx)
    user_force_vals.append(fv)

user_node_pos = [np.linspace(0, L_beam, nelem+1).tolist() for _ in range(n_cases)]

def build_user_input_no_agg(user_roller, user_force_x, user_force_vals, user_node_pos, 
                            scalers, n_cases, max_lengths):
    feat_3d = scale_user_inputs(
        user_roller, user_force_x, user_force_vals, user_node_pos,
        scalers, n_cases, max_lengths
    )
    return feat_3d

X_user_3d = build_user_input_no_agg(
    user_roller, user_force_x, user_force_vals, user_node_pos,
    scalers_inputs, n_cases, max_lengths
)

B_usr, Nc_usr, Fin_usr = X_user_3d.shape
X_user_2d = X_user_3d.reshape(B_usr, -1)
X_user_t  = torch.tensor(X_user_2d, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    pred_1x = model(X_user_t)  # => (1, n_elem)
pred_1x_np = pred_1x.cpu().numpy().squeeze()
pred_1x_unstd = scaler_Y.inverse_transform(pred_1x_np.reshape(1, -1)).squeeze()

# Visualization
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

all_force_vals = force_vals_plot
max_force = max(abs(val) for val in all_force_vals) if all_force_vals else 1.0
desired_max_arrow_length = 2.0
arrow_scale = desired_max_arrow_length / max_force if max_force != 0 else 1.0

beam_positions = user_node_pos[0][:nelem]
I_normalized = (pred_1x_unstd - pred_1x_unstd.min()) / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8)
cmap = cm.winter
norm = plt.Normalize(pred_1x_unstd.min(), pred_1x_unstd.max())

block_width = L_beam / nelem * 0.8
block_height = 1

fig, ax = plt.subplots(figsize=(18, 7))
ax.plot(beam_x, beam_y_vals, color='black', linewidth=3, label='Beam')
ax.scatter(beam_x[0], beam_y - 0.15, marker='^', color='red', s=300, zorder=6)

# Plot Rollers
ax.scatter(unique_rollers, [beam_y]*len(unique_rollers),
           marker='o', color='seagreen', s=200, label='Rollers',
           zorder=5, edgecolors='k')

# Plot Forces
for case_idx in range(n_cases):
    fx_list = user_force_x[case_idx]
    fv_list = user_force_vals[case_idx]
    color = case_colors[case_idx]
    label = case_labels[case_idx]

    for idx, (fx, fv) in enumerate(zip(fx_list, fv_list)):
        arrow_length = abs(fv)*arrow_scale
        start_point = (fx, beam_y + arrow_length)
        end_point   = (fx, beam_y)

        arrow = FancyArrowPatch(
            posA=start_point, posB=end_point,
            arrowstyle='-|>',
            mutation_scale=20,
            color=color, linewidth=2, alpha=0.8,
            label=label if idx == 0 else ""
        )
        ax.add_patch(arrow)
        ax.text(fx, beam_y + arrow_length + desired_max_arrow_length*0.02,
                f"{fv:.0f}", ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold')

# Plot predicted I
for idx, (x_pos, I_val) in enumerate(zip(beam_positions, pred_1x_unstd)):
    color = cmap(norm(I_val))
    rect_x = x_pos - block_width/2
    rect_y = beam_y - (I_val / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8))*block_height/2
    rect = Rectangle(
        (rect_x, rect_y),
        block_width,
        (I_val / (pred_1x_unstd.max() - pred_1x_unstd.min() + 1e-8))*block_height,
        linewidth=0, edgecolor=None, facecolor=color, alpha=0.6
    )
    ax.add_patch(rect)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Predicted I (m$^4$)', fontsize=16)
cbar.ax.tick_params(labelsize=10)

ax.set_title("Beam Setup with Applied Forces & GNN (Speedups) Predicted I",
             fontsize=22, fontweight='bold', pad=20)
ax.set_xlabel("Beam Length (m)", fontsize=16, fontweight='semibold')
ax.set_xlim(-5, L_beam+5)
ax.set_ylim(-2.5, 2.5)
ax.set_xticks(np.arange(0, L_beam+5, 5))
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

legend_elements = [
    Line2D([0], [0], color='black', lw=3, label='Beam'),
    Line2D([0], [0], marker=(3,0,-90), color='red', label='Pin',
           markerfacecolor='red', markersize=15),
    Line2D([0], [0], marker='o', color='seagreen', label='Rollers',
           markerfacecolor='seagreen', markeredgecolor='k', markersize=15),
]
for color, label in zip(case_colors, case_labels):
    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))
ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

plt.tight_layout()
plt.show()
