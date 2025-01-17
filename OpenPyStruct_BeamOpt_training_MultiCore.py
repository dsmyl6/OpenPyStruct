################################################################
#### OpenPyStruct Multicore Optimizer / Data Generator      ####
#### Coder: Danny Smyl, PhD, PE, Georgia Tech, 2025         ####
################################################################

from joblib import Parallel, delayed
import os
import openseespy.opensees as ops
import numpy as np
import random
import torch
import json
from torch.optim.lr_scheduler import ExponentialLR
import time

##############
# Parameters #
##############

E = 200e9                     # Young's Modulus (Pa)
nu = 0.3                      # Poisson ratio
G = E / (2 * (1 + nu))        # shear modulus (Pa)
A = 0.01                      # Cross-sectional area (m^2)
L_max = 200.0                     # Length of the beam (m)
num_nodes = 101               # Number of nodes
num_elements = num_nodes - 1  # Number of elements
N_rollers_max = 4                 # Number of additional roller supports
M_forces_max = 4                  # Number of point forces
L_min = 15                    # min distance between rollers (m)
max_force = -355857           # Max point load (N) - 80,0000 lb semi
min_force = max_force / 10
uniform_udl = -1000           # uniformly distributed load (N)
I_0 = 0.5                     # initial guess for I

## opt parameters ##
max_e = 600
lr = 0.01            # initial opt (momentum) rate
gamma = 0.98         # lr decay rate
alpha_moment = 1e-2  # coefficient for bending energy loss term
alpha_shear = 1e-2   # coefficient for shear energy loss term

# Parameters for stopping criterion #
tolerance = 5e-3  # Minimum improvement in loss
patience = 5     # Number of epochs to wait for improvement

num_samples = 100000  # Number of training data samples to generate

random_bridge = 0     # input 1 if you want the bridge length and roller locations randomized
flag = random_bridge  # flag for generating fixed or random bridge

# User-specified number of CPU cores #
num_workers = 22

#################################################################################################
##   generate a random geometry and support condition (only used if not randomizing a bridge)  ##
#################################################################################################

L = L_max
node_positions = np.linspace(0, L, num_nodes)

# Randomize roller node locations - or fix, if you prefer #
roller_nodes, available_nodes = [], list(range(2, num_nodes))  # Exclude node 1
num_rollers = 4

# assign rollers to nodes #
roller_nodes = list([10, 30, 70, 85, num_nodes-1]) 

# Remove the fixed roller nodes from available_nodes #
for node in roller_nodes:
    available_nodes.remove(node)

#################################################################################################
#################################################################################################
#################################################################################################

# Training data structure
training_data = {
    "roller_x_locations": [],
    "force_x_locations": [],
    "force_values": [],
    "I_values": [],
    "shear_forces": [],
    "bending_moments": [],
    "node_positions": [],
    "roller_nodes": [],
    "force_nodes": [],
    "num_nodes": [],
    "L": [],
    "rotations": [],
    "deflections": [],
}

def setup_model(I_tensor, node_positions, roller_nodes, force_nodes, force_values, A, E, uniform_udl):
    """
    Set up the OpenSees model using updated moments of inertia and apply uniform UDL to all elements.
    """
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Define nodes
    for i, x in enumerate(node_positions):
        ops.node(i + 1, x, 0.0)

    # Apply supports
    ops.fix(1, 1, 1, 0)  # First node (constrained in X-Y, rotationally free)
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

def generate_sample(sample_idx, num_nodes, flag, L, node_positions, roller_nodes, available_nodes, patience=10):
    """
    Generate a single sample for training data with optimized OpenSeesPy modeling and early stopping.
    """
    import openseespy.opensees as ops  # import in function (may or may not work without this)
    
    # Random bridge length / rollers/ nodes
    if flag == 1:
        L = L_min + random.uniform(0, L_max)
        node_positions = np.linspace(0, L, num_nodes)
        
        # Randomize roller node locations
        roller_nodes, available_nodes = [], list(range(2, num_nodes))  # Exclude node 1
        roller_nodes.sort(reverse=True)
        num_rollers = random.randint(1, N_rollers_max)
        
        # Add first roller node
        first_roller_node = random.choice(available_nodes)
        roller_nodes.append(first_roller_node)
        available_nodes.remove(first_roller_node)
        
        # Add additional rollers without minimum distance constraint
        for _ in range(num_rollers - 1):
            if available_nodes:  # Ensure there are available nodes
                new_roller_node = random.choice(available_nodes)
                roller_nodes.append(new_roller_node)
                available_nodes.remove(new_roller_node)
    

    # Randomize point forces and their values
    num_forces = random.randint(1, M_forces_max)
    force_nodes = random.sample(available_nodes, min(num_forces, len(available_nodes)))
    force_values = [random.uniform(min_force, max_force) for _ in force_nodes]

    # Initialize moments of inertia tensor
    I_tensor = torch.tensor([I_0] * num_elements, dtype=torch.float32, requires_grad=True)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam([I_tensor], lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    # Optimization loop with early stopping
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_e):
        optimizer.zero_grad()
        ops.wipe()  # Reset OpenSees model

        setup_model(I_tensor, node_positions, roller_nodes, force_nodes, force_values, A, E, uniform_udl)
        
        ops.analysis('Static')
        success = ops.analyze(1)

        if success != 0:
            # If analysis fails, skip this sample
            return None

        # Compute losses
        try:
            bending_moments = torch.tensor([ops.eleResponse(i, 'forces')[2] for i in range(1, len(I_tensor) + 1)], dtype=torch.float32)
            shear_forces = torch.tensor([ops.eleResponse(i, 'forces')[1] for i in range(1, len(I_tensor) + 1)], dtype=torch.float32)
        except:
            # If response retrieval fails, skip this sample
            return None

        bending_energy = torch.sum((bending_moments ** 2) / (2 * E * I_tensor + 1e-6))
        A_approx = 0.03 * I_tensor ** 0.5
        shear_energy = torch.sum(shear_forces ** 2 / (G * A_approx))
        primary_loss = torch.sum(I_tensor)
        total_loss = primary_loss + alpha_moment * bending_energy + alpha_shear * shear_energy

        # Backpropagate and update
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Prevent negative inertia values
        with torch.no_grad():
            I_tensor.clamp_(min=1e-8)

        # Early stopping logic
        if total_loss.item() < best_loss - tolerance:
            best_loss = total_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Extract rotations and deflections
    rotations = [ops.nodeDisp(i, 3) if i < len(node_positions) else 0.0 for i in range(1, len(node_positions) + 1)]
    deflections = [ops.nodeDisp(i, 2) if i < len(node_positions) else 0.0 for i in range(1, len(node_positions) + 1)]

    # Return the results
    return {
        "roller_x_locations": [node_positions[node - 1] for node in roller_nodes],
        "force_x_locations": [node_positions[node - 1] for node in force_nodes],
        "force_values": force_values,
        "I_values": I_tensor.detach().numpy().tolist(),
        "shear_forces": shear_forces.detach().tolist(),
        "bending_moments": bending_moments.detach().tolist(),
        "node_positions": node_positions.tolist(),
        "roller_nodes": roller_nodes,
        "force_nodes": force_nodes,
        "num_nodes": num_nodes,
        "L": L,
        "rotations": rotations,
        "deflections": deflections,
    }

def main():
    start_time = time.time()  # Start the timer

    # Define batch size
    batch_size = 500
    total_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        # Generate current batch indices
        current_indices = range(start_idx, end_idx)

        # Process the current batch in parallel
        results = Parallel(n_jobs=num_workers, backend="loky")(
            delayed(generate_sample)(
                i, num_nodes, flag, L, node_positions, roller_nodes.copy(), available_nodes.copy()
            ) for i in current_indices
        )

        # Filter out any None results due to failures
        filtered_results = [res for res in results if res is not None]

        # Append the results to the training data
        for result in filtered_results:
            for key, value in result.items():
                training_data[key].append(value)

        # Update and print the counter every 500 samples
        processed = end_idx
        print(f"{processed} samples processed.")

    # Save the training data to a JSON file
    with open("training_data_PINN_mini.json", "w") as f:
        json.dump(training_data, f)

    end_time = time.time()  # End the timer

    print("Data generation complete.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    
    
# For sanity: Load the training data from the JSON file #
with open("training_data_PINN_mini.json", "r") as file:
    training_data = json.load(file)

# Print a summary of the dataset
print("Data loaded successfully!")
print(f"Number of samples: {len(training_data['roller_x_locations'])}")
print("Keys available in the dataset:")
for key in training_data.keys():
    print(f"- {key} (Number of entries: {len(training_data[key])})")

# Access all data into variables
roller_x_locations = training_data["roller_x_locations"]
force_x_locations = training_data["force_x_locations"]
force_values = training_data["force_values"]
I_values = training_data["I_values"]
shear_forces = training_data["shear_forces"]
bending_moments = training_data["bending_moments"]
node_positions = training_data["node_positions"]
roller_nodes = training_data["roller_nodes"]
force_nodes = training_data["force_nodes"]
num_nodes = training_data["num_nodes"]
beam_lengths = training_data["L"]
rotations = training_data["rotations"]
deflections = training_data["deflections"]
