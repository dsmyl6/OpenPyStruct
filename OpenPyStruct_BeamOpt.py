
################################################################
#### OpenPyStruct Single Load Optimizer                     ####
#### Coder: Danny Smyl, PhD, PE, Georgia Tech, 2025         ####
################################################################

import os
import openseespy.opensees as ops
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

# Disable torch._dynamo optimizations
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "NONE"



##############
# Parameters #
##############

E = 200e9                     # Young's Modulus (Pa)
nu = 0.3                      # Poisson ratio
G = E / (2 * (1 + nu))        # shear modulus (Pa)
A = 0.01                      # Cross-sectional area (m^2)
L = 200.0                     # Length of the beam (m)
num_nodes = 101               # Number of nodes
num_elements = num_nodes - 1  # Number of elements
N_rollers = 5                 # Number of additional roller supports
M_forces = 5                  # Number of point forces
L_min = 15                    # min distance between rollers (m)
max_force = -355857           # Max point load (N) - 80,0000 lb semi
uniform_udl = -5000           # unniformly distributed load (N)
I_0 = 0.5                     # initial guess for I


## opt parameters ##
num_epochs = 1000
lr = 0.01            #initial opt (momentum) rate
gamma = 0.98         # lr decay rate
alpha_moment = 1e-2  # coefficient for bending energy loss term
alpha_shear = 1e-2   # coefficient for shear energy loss term

# Parameters for stopping criterion #
tolerance = 1e-2  # Minimum improvement in loss
patience = 10     # Number of epochs to wait for improvement

# Initialize the moments of inertia
I_values = [I_0 for _ in range(num_elements)]

# Define nodal positions
node_positions = np.linspace(0, L, num_nodes)

# Predefine roller nodes, ensuring a minimum distance between them
roller_nodes = []
available_nodes = [n for n in range(2, num_nodes) if n != num_nodes]  # List of nodes to choose from

# Start by selecting the first roller node
first_roller_node = random.choice(available_nodes)
roller_nodes.append(first_roller_node)
available_nodes.remove(first_roller_node)

# Now iteratively select the remaining roller nodes
for _ in range(1, N_rollers):
    valid_choice = False
    while not valid_choice:
        # Randomly pick a new roller node
        new_roller_node = random.choice(available_nodes)
        
        # Check that the distance to all existing roller nodes is >= L_min
        if all(abs(new_roller_node - existing_node) >= L_min for existing_node in roller_nodes):
            roller_nodes.append(new_roller_node)
            available_nodes.remove(new_roller_node)
            valid_choice = True

force_nodes = [n for n in range(2, num_nodes) if n not in roller_nodes and n != 1]
force_nodes = random.sample(force_nodes, min(M_forces, len(force_nodes)))
force_values = [random.uniform(0.5 * max_force, max_force) for _ in force_nodes]





######################################
## Configure Model Helper Functions ##
######################################


def setup_model(I_tensor, node_positions, roller_nodes, force_nodes, force_values, A, E, uniform_udl):
    """
    Set up the OpenSees model using updated moments of inertia and apply uniform UDL to all elements.
    """
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Define nodes
    for i, x in enumerate(node_positions):
        ops.node(i + 1, x, 0.0)

    # Apply supports
    ops.fix(1, 1, 1, 0)  # First node (constrained in X, rotationally free)
    for node in roller_nodes:
        ops.fix(node, 0, 1, 0)  # All other nodes (rotationally free, free in X)

    # Define elements with updated moments of inertia
    ops.geomTransf('Linear', 1)
    for i in range(len(node_positions) - 1):
        ops.element('elasticBeamColumn', i + 1, i + 1, i + 2, A, E, I_tensor[i].item(), 1)

    # Apply point loads
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for node, force in zip(force_nodes, force_values):
        ops.load(node, 0.0, force, 0.0)

    # Apply uniform UDL to all elements
    for elem_id in range(1, len(node_positions)):
        ops.eleLoad('-ele', elem_id, '-type', '-beamUniform', uniform_udl, uniform_udl)

    # Static analysis setup
    ops.system('BandSPD')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')

def compute_combined_loss(I_tensor, E , G, alpha_moment, alpha_shear, node_positions):
    shear_forces = []
    bending_moments = []
    rotations = []
    deflections = []

    # Retrieve element responses
    for elem_id in range(1, len(I_tensor) + 1):
        response = ops.eleResponse(elem_id, 'forces')
        shear_forces.append(response[1])  # Shear force
        bending_moments.append(response[2])  # Moment at the start of the element
        
    for node_id in range(1, len(node_positions) + 1):  # Node IDs start from 1
        uy = ops.nodeDisp(node_id, 2)  # Translation in Y
        theta = ops.nodeDisp(node_id, 3)  # Rotation
        deflections.append(uy)
        rotations.append(theta)

    
    deflections = torch.tensor(np.array(deflections))
    
    # Convert bending moments and shear forces to PyTorch tensors
    bending_moments = torch.tensor(bending_moments, dtype=torch.float32, requires_grad=True)
    shear_forces = torch.tensor(shear_forces, dtype=torch.float32, requires_grad=True)

    # Compute bending energy
    bending_energy = torch.sum((bending_moments**2) / (2 * E * I_tensor + 1e-6))  # Avoid division by zero

    # make a proportional A
    k = 0.03 # assuming  0.01 is suitable for a built up section
    A = k* I_tensor** 0.5
    # Compute shear energy
    shear_energy = torch.sum(shear_forces**2 / (G*A))  # Relative to A

    # Compute primary loss: minimize the sum of moments of inertia
    primary_loss = torch.sum(I_tensor)

    # Combine losses
    total_loss = primary_loss + alpha_moment * bending_energy + alpha_shear * shear_energy

    return total_loss, primary_loss, alpha_moment * bending_energy, alpha_shear * shear_energy





##############################################
#### Optimization Initialization and loop ####
##############################################


# Convert I_values to a PyTorch tensor for optimization
I_tensor = torch.tensor(I_values, dtype=torch.float32, requires_grad=True)

# Define the optimizer
optimizer = torch.optim.Adam([I_tensor], lr=lr)
scheduler = ExponentialLR(optimizer, gamma=gamma)

# Store loss history for debugging
loss_history = {
    "total": [],
    "primary": [],
    "bending_energy": [],
    "shear_energy": []
}

# Initialize variables for stopping criterion
best_loss = float('inf')  # Initialize best loss as infinity
no_improvement_epochs = 0  # Counter for epochs with no improvement

# Optimization loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Clear and rebuild the model
    ops.wipe()
    setup_model(I_tensor, node_positions, roller_nodes, force_nodes, force_values, A, E, uniform_udl)
    
    ops.analysis('Static')
    ops.analyze(1)

    # Compute the combined loss
    total_loss, primary_loss, bending_energy, shear_energy = compute_combined_loss(
        I_tensor, E, G, alpha_moment, alpha_shear, node_positions)

    # Backpropagate and optimize
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    # Clamp values to prevent negative inertia
    with torch.no_grad():
        I_tensor.clamp_(min=1e-8)

    # Store loss components for debugging
    loss_history["total"].append(total_loss.item())
    loss_history["primary"].append(primary_loss.item())
    loss_history["bending_energy"].append(bending_energy.item())
    loss_history["shear_energy"].append(shear_energy.item())

    # Check for stopping criterion
    if total_loss.item() < best_loss - tolerance:
        best_loss = total_loss.item()
        no_improvement_epochs = 0  # Reset counter if improvement
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= patience:
        print(f"Stopping early at epoch {epoch + 1}: No improvement in loss for {patience} epochs.")
        break

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Total Loss: {total_loss.item():.6f}")
        print(f"Primary Loss: {primary_loss.item():.6f}")
        print(f"Bending Energy: {bending_energy.item():.6f}, Shear Energy: {shear_energy.item():.6f}")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history["total"], label="Total Loss")
plt.plot(loss_history["primary"], label="Primary Loss (I Sum)")
plt.plot(loss_history["bending_energy"], label="Bending Energy Loss")
plt.plot(loss_history["shear_energy"], label="Shear Energy Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Components During Optimization")
plt.show()





#########################################
# Update I_values with optimized values #
#########################################

I_values = I_tensor.detach().numpy()

# Retrieve Shear and Moment Data #
shear_forces = []
bending_moments = []
for elem_id in range(1, num_elements + 1):
    response = ops.eleResponse(elem_id, 'forces')
    shear_forces.append(response[1])  # Shear force
    bending_moments.append(response[2])  # Moment at the start of the element

# Convert Shear Forces to kN and Bending Moments to kN·m
shear_forces_kn = [sf / 1e3 for sf in shear_forces]  # Convert to kN
bending_moments_knm = [bm / 1e3 for bm in bending_moments]  # Convert to kN·m




#################
# Visualization #
#################


fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

# Plot Moments of Inertia as Scaled Beam Elements
for i in range(len(I_values)):
    thickness = 15 * (I_values[i] / max(I_values))**(1/3)  # Scale thickness
    axs[0].plot([node_positions[i], node_positions[i + 1]], [0, 0], linewidth=thickness, color='blue', alpha=0.3, label='Approx section height' if i == 0 else "")

# Highlight Pin and Roller Supports
axs[0].scatter(0, 0, color='green', s=200, marker='^', label='Pin Support')  # Pin at x = 0
for node in roller_nodes:
    x = node_positions[node - 1]
    axs[0].scatter(x, 0, color='red', s=200, marker='o', label='Roller Support' if node == roller_nodes[0] else "")
if num_nodes not in roller_nodes:
    axs[0].scatter(L, 0, color='red', s=200, marker='o')  # Explicitly add roller at x = L

# Plot Point Loads
for node, force in zip(force_nodes, force_values):
    x = node_positions[node - 1]
    arrow_length = -0.0125  # Arrow length downward
    axs[0].arrow(x, -arrow_length + 0.0125, 0, arrow_length, head_width=3.5, head_length=0.0125,
                 fc='red', ec='red', label='Point Load' if node == force_nodes[0] else "")

# Set titles, labels, and legend for Moment of Inertia plot
#axs[0].set_title('Beam Elements Highlighted by Moment of Inertia', fontsize=20)
axs[0].set_ylabel('(Normalized I)$^{1/3}$', fontsize=20)
axs[0].grid(True)
axs[0].legend(fontsize=22)
axs[0].set_xlim([0, max(node_positions)])  # Ensure x-axis spans the entire beam

# Plot Shear Force Diagram (in kN)
axs[1].step(node_positions[:-1], shear_forces_kn, where='post', color='red', label='Shear Force')
axs[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axs[1].set_title('Shear Force Diagram', fontsize=20)
axs[1].set_ylabel('Shear Force (kN)', fontsize=20)
axs[1].grid(True)
axs[1].set_xlim([0, L])

# Plot Bending Moment Diagram (in kN·m)
moment_positions = (node_positions[:-1] + node_positions[1:]) / 2  # Midpoints
axs[2].plot(moment_positions, bending_moments_knm, color='blue', marker='o', label='Bending Moment')
axs[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axs[2].set_title('Bending Moment Diagram', fontsize=20)
axs[2].set_ylabel('Bending Moment (kN·m)', fontsize=20)
axs[2].set_xlabel('Beam Span (m)', fontsize=20)  # Label x-axis with units
axs[2].grid(True)
axs[2].set_xlim([0, L])

# Adjust layout and display
plt.tight_layout()
plt.show()
