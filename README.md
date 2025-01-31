# **OpenPyStruct**
 is an open-source toolkit designed for machine learning-based structural optimization. Leveraging Physics-Informed Neural Networks (PINNs), Transformer-Diffusion Modules, and other state-of-the-art techniques, the framework offers unparalleled capabilities for tackling single and multi-load case optimization problems with diverse boundary and loading conditions.

## **Features**
**Physics-Informed Neural Networks (PINNs):** Embeds structural mechanics into the learning process for highly accurate predictions.\
**Transformer-Diffusion Modules:** Incorporates advanced attention mechanisms and diffusion-based techniques for modeling complex structural behavior.\
**Feedforward Neural Networks (FNNs):** Provides scalable solutions for simpler structural optimization tasks.\
**Multi-Core and GPU-Accelerated Optimization:** Enables large-scale data generation and rapid computations.\
**OpenSeesPy Integration:** Facilitates physics-based finite element simulations.\
**Flexible Loss Functions and Parameter Design:** Supports user-defined constraints, objectives, and optimization goals.

## **Requirements**

### **Install openseespy first:**

```zsh
pip install openseespy
```

Ensure you have Python 3.8+ installed. The required libraries include:

numpy\
torch\
matplotlib\
seaborn\
scikit-learn\
openseespy\
Installation

Notes:
- do we want to delete this in favor of conda env install?
- Should we specify package versions also?
- Does this work on Mac, or just Windows. Should make a note


## **Conda Environement Install**
```zsh
conda env create -f environment.yml
conda activate OpenPyStruct
```

# Code Documentation

# **OpenPyStruct Single Load Optimizer**

## **Overview**
This script optimizes the distribution of the moment of inertia along a beam to minimize structural response (such as deflection and internal forces) while maintaining structural efficiency. The optimization process iteratively adjusts the moment of inertia of beam elements to reduce bending and shear energy losses.

## **Key Features**
- Uses **OpenSeesPy** for finite element analysis.
- Implements **PyTorch** for gradient-based optimization.
- Simulates **beam behavior** under various loads, including **point forces** and **uniformly distributed loads**.
- Incorporates **randomized placement** of roller supports and point loads for testing.
- Optimizes the **moment of inertia** to minimize structural energy loss.
- Implements **early stopping** to improve computational efficiency.
- **Visualizes** optimization progress and structural response.

## **Workflow**

1. **Define Model Parameters:**
   - Material properties (Young’s modulus, Poisson’s ratio).
   - Beam geometry (length, cross-section).
   - Boundary conditions (roller and pinned supports).
   - Loading conditions (point loads, uniformly distributed loads).

2. **Set Up the Finite Element Model:**
   - Define nodes and elements in OpenSeesPy.
   - Apply supports and loads.
   - Assign varying moment of inertia values.

3. **Optimize the Moment of Inertia:**
   - Use PyTorch’s **Adam optimizer** with an **exponential learning rate decay**.
   - Compute loss as a combination of:
     - **Primary loss:** Total moment of inertia.
     - **Bending energy loss:** Based on internal moments.
     - **Shear energy loss:** Based on shear forces.
   - Update inertia values using **gradient descent**.
   - Clamp inertia values to prevent negative values.
   - Stop optimization early if loss stagnates.

4. **Run Finite Element Analysis (FEA):**
   - Perform static analysis in OpenSeesPy.
   - Extract **deflections, rotations, shear forces, and bending moments**.

5. **Visualization and Results:**
   - Plot **loss history** over optimization epochs.
   - Display **beam deformation, moment of inertia distribution, shear forces, and bending moments**.
   - Highlight **support locations and applied forces**.


## Contributer Guide

I suggest we set up pre-commit and linting to ensure uniform coding and markdown styles are enforced automatically. If you want I can set this up.

## **MIT Open License**

Copyright <2025>  <Danny Smyl>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
