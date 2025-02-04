################################################################
#### OpenPyStruct Single Frame Optimizer                    ####
#### Coder: Danny Smyl, PhD, PE, Georgia Tech, 2025         ####
################################################################

import openseespy.opensees as ops
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

##############################
# User-Defined and Material Parameters
##############################

# Maximum numbers for random generation:
max_bays = 10  # Maximum number of bays (span divisions)
max_stories = 10  # Maximum number of stories

# Geometric dimensions (in meters)
bay_width = 6.0  # Width of each bay
story_height = 3.0  # Height of each story

# Material and cross-sectional properties:
E = 200e9  # Young's Modulus (Pa)
nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + nu))  # Shear Modulus (Pa)
A = 0.02  # Cross-sectional area (m^2) used in the analysis (assumed constant)
I0 = 5e-4  # Initial guess for moment of inertia (m^4)

# Loss function coefficients
alpha_moment = 1e-2
alpha_shear = 1e-2
k = 0.03  # Coefficient to define local area: A_local = k * sqrt(I)

# Applied loads:
lateral_load = 1e4  # Lateral nodal load (N) applied at left-hand side nodes (x = 0)
vertical_load = (
    -1e4
)  # Uniform vertical load (N) applied to beam elements (negative for downward)

# Optimization settings
num_epochs = 5000
lr = 0.005
tolerance = 1e-3
patience = 10

##############################
# Random Generation of Frame Geometry
##############################

num_bays = random.randint(1, max_bays)
num_stories = random.randint(1, max_stories)
print(f"Generated frame with {num_bays} bay(s) and {num_stories} story(ies).")

# Create node grid and store original coordinates (nodes numbered from 1)
node_coords = {}
num_nodes = (num_stories + 1) * (num_bays + 1)
for i in range(num_stories + 1):
    for j in range(num_bays + 1):
        tag = i * (num_bays + 1) + j + 1
        x = j * bay_width
        y = i * story_height
        node_coords[tag] = (x, y)

# Count elements:
# Columns: vertical members (each column between successive stories)
num_columns = num_stories * (num_bays + 1)
# Beams: horizontal members (each beam on each story except the ground)
num_beams = num_stories * num_bays
total_elems = num_columns + num_beams

##############################
# OpenSees Model Setup Functions
##############################


def setup_frame_model(I_tensor):
    """
    Build the 2D frame model in OpenSees using the moment-of-inertia vector I_tensor.
    The first num_columns entries of I_tensor are used for columns;
    the remainder for beams.

    Loads are applied as follows:
      - Nodal lateral loads on all left-hand side nodes (x=0) except ground (y=0).
      - Uniform vertical loads (using eleLoad) on all beam elements.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.geomTransf("Linear", 1)

    # Create nodes using stored coordinates
    for tag, (x, y) in node_coords.items():
        ops.node(tag, x, y)

    # Fix nodes at the ground (nodes with y=0)
    for tag, (x, y) in node_coords.items():
        if y == 0.0:
            ops.fix(tag, 1, 1, 1)

    # Create column elements (vertical members)
    elem_tag = 1  # element numbering starts at 1
    for i in range(num_stories):
        for j in range(num_bays + 1):
            node_i = i * (num_bays + 1) + j + 1
            node_j = (i + 1) * (num_bays + 1) + j + 1
            I_val = I_tensor[
                elem_tag - 1
            ].item()  # columns occupy indices 0 .. num_columns-1
            ops.element("elasticBeamColumn", elem_tag, node_i, node_j, A, E, I_val, 1)
            elem_tag += 1

    # Create beam elements (horizontal members)
    for i in range(1, num_stories + 1):
        for j in range(num_bays):
            node_i = i * (num_bays + 1) + j + 1
            node_j = i * (num_bays + 1) + j + 2
            I_val = I_tensor[elem_tag - 1].item()  # beams occupy remaining indices
            ops.element("elasticBeamColumn", elem_tag, node_i, node_j, A, E, I_val, 1)
            elem_tag += 1

    # Define load pattern:
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    # --- Apply lateral loads on left-hand side nodes (x = 0) except the ground ---
    # Left-hand side nodes are those with j == 0 and y > 0.
    for tag, (x, y) in node_coords.items():
        if x == 0.0 and y != 0.0:
            ops.load(tag, lateral_load, 0.0, 0.0)

    # --- Apply uniform vertical load to all beam elements ---
    # Beam elements have tags from num_columns+1 to total_elems.
    for ele in range(num_columns + 1, total_elems + 1):
        """
        For a uniform beam load (applied in the local vertical direction)
        the command below applies vertical load (downward) at both ends of the element
        """
        ops.eleLoad("-ele", ele, "-type", "-beamUniform", vertical_load, vertical_load)

    # Analysis settings
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Newton")
    ops.analysis("Static")


def compute_combined_loss(I_tensor):
    """
    Run the analysis and compute the combined loss:
      total_loss = sum(I)
      + α_moment*(∑[bending_moment²/(2·E·I)])
      + α_shear*(∑[shear_force²/(G·A_local)])
    where A_local = k * sqrt(I).
    """
    bending_energy = 0.0
    shear_energy = 0.0
    for elem_id in range(1, total_elems + 1):
        # Retrieve element forces (assumed ordering: [axial, shear, bending moment, ..])
        response = ops.eleResponse(elem_id, "forces")
        shear_force = response[1]
        bending_moment = response[2]
        I_val = I_tensor[elem_id - 1]
        bending_energy += (bending_moment**2) / (2 * E * I_val + 1e-8)
        A_local = k * (I_val**0.5)
        shear_energy += (shear_force**2) / (G * A_local)
    primary_loss = torch.sum(I_tensor)
    total_loss = (
        primary_loss + alpha_moment * bending_energy + alpha_shear * shear_energy
    )
    return (
        total_loss,
        primary_loss,
        alpha_moment * bending_energy,
        alpha_shear * shear_energy,
    )


##############################
# Optimization Initialization
##############################

# Initialize moment-of-inertia values for all elements (columns + beams)
I_values = [I0 for _ in range(total_elems)]
I_tensor = torch.tensor(I_values, dtype=torch.float32, requires_grad=True)

optimizer = torch.optim.Adam([I_tensor], lr=lr)
loss_history = []
best_loss = float("inf")
no_improve = 0

##############################
# Optimization Loop
##############################

for epoch in range(num_epochs):
    optimizer.zero_grad()
    ops.wipe()  # Clear previous model
    setup_frame_model(I_tensor)  # Rebuild the model with current I values
    ops.analyze(1)
    total_loss, primary_loss, bending_loss, shear_loss = compute_combined_loss(I_tensor)
    total_loss.backward()
    optimizer.step()

    # Prevent I from going to zero or negative
    with torch.no_grad():
        I_tensor.clamp_(min=1e-8)

    current_loss = total_loss.item()
    loss_history.append(current_loss)

    if current_loss < best_loss - tolerance:
        best_loss = current_loss
        no_improve = 0
    else:
        no_improve += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d}: Total Loss = {current_loss:.6e}, "
            f"Primary Loss = {primary_loss.item():.6e}"
        )

    if no_improve >= patience:
        print(
            f"Stopping early at epoch {epoch+1} (no improvement for {patience} epochs)."
        )
        break

print(f"\nOptimization complete. Best Loss: {best_loss:.6e}")

# Extract the optimized I values as a NumPy array.
opt_I = I_tensor.detach().numpy()

##############################
# Re-run the Analysis with Optimized I
##############################

ops.wipe()
setup_frame_model(I_tensor)
ops.analyze(1)

##############################
# Plot Loss History
##############################

plt.figure(figsize=(8, 5))
plt.plot(loss_history, "b-", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Optimization Loss History")
plt.grid(True)
plt.show()

##############################
# Visualization: Plot the Frame with Optimized I Distribution
##############################

plt.figure(figsize=(12, 8))

# Plot the original (undeformed) frame in light gray for reference
for i in range(num_stories):
    for j in range(num_bays + 1):
        node_i = i * (num_bays + 1) + j + 1
        node_j = (i + 1) * (num_bays + 1) + j + 1
        x_i, y_i = node_coords[node_i]
        x_j, y_j = node_coords[node_j]
        plt.plot([x_i, x_j], [y_i, y_j], "--", color="lightgray")
for i in range(1, num_stories + 1):
    for j in range(num_bays):
        node_i = i * (num_bays + 1) + j + 1
        node_j = i * (num_bays + 1) + j + 2
        x_i, y_i = node_coords[node_i]
        x_j, y_j = node_coords[node_j]
        plt.plot([x_i, x_j], [y_i, y_j], "--", color="lightgray")

# Plot the optimized frame: scale line width based on optimized I (cube-root scaling)
max_I_val = max(opt_I)
elem_index = 0

# Plot columns (in blue)
for i in range(num_stories):
    for j in range(num_bays + 1):
        node_i = i * (num_bays + 1) + j + 1
        node_j = (i + 1) * (num_bays + 1) + j + 1
        x_i, y_i = node_coords[node_i]
        x_j, y_j = node_coords[node_j]
        I_elem = opt_I[elem_index]
        lw = 15 * (I_elem / max_I_val) ** (1 / 3)
        plt.plot(
            [x_i, x_j],
            [y_i, y_j],
            "b-",
            linewidth=lw,
            label="Column" if elem_index == 0 else "",
        )
        elem_index += 1

# Plot beams (in red)
for i in range(1, num_stories + 1):
    for j in range(num_bays):
        node_i = i * (num_bays + 1) + j + 1
        node_j = i * (num_bays + 1) + j + 2
        x_i, y_i = node_coords[node_i]
        x_j, y_j = node_coords[node_j]
        I_elem = opt_I[elem_index]
        lw = 15 * (I_elem / max_I_val) ** (1 / 3)
        plt.plot(
            [x_i, x_j],
            [y_i, y_j],
            "r-",
            linewidth=lw,
            label="Beam" if elem_index == num_columns else "",
        )
        elem_index += 1

plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.title("Frame Structure with Optimized Moment of Inertia Distribution")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
