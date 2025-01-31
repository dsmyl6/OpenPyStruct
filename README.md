# **OpenPyStruct**

**OpenPyStruct** is an open-source toolkit designed for machine learning-based structural optimization. Leveraging **Physics-Informed Neural Networks (PINNs)**, **Transformer-Diffusion Modules**, and other state-of-the-art techniques, the framework provides powerful tools for tackling **single and multi-load case optimization** problems with diverse boundary and loading conditions.

## **Table of Contents**

- [**OpenPyStruct**](#openpystruct)
  - [**Table of Contents**](#table-of-contents)
  - [**Features**](#features)
  - [**Requirements**](#requirements)
    - [Option 1, Manual Install](#option-1-manual-install)
    - [Option 2, Conda Environment Install](#option-2-conda-environment-install)
  - [**Script Usage Documentation**](#script-usage-documentation)
    - [**Single Load Optimizer**](#single-load-optimizer)
      - [**Overview**](#overview)
      - [**Key Features**](#key-features)
      - [**Workflow**](#workflow)
    - [**Physics-Informed Neural Network MultiCase**](#physics-informed-neural-network-multicase)
      - [**Overview**](#overview-1)
      - [**Key Features**](#key-features-1)
      - [**Workflow**](#workflow-1)
  - [**Contributor Guide**](#contributor-guide)
  - [**License**](#license)

---

## **Features**

- **Physics-Informed Neural Networks (PINNs):** Embeds structural mechanics into the learning process for highly accurate predictions.
- **Transformer-Diffusion Modules:** Incorporates advanced attention mechanisms and diffusion-based techniques for modeling complex structural behavior.
- **Feedforward Neural Networks (FNNs):** Provides scalable solutions for simpler structural optimization tasks.
- **Multi-Core and GPU-Accelerated Optimization:** Enables large-scale data generation and rapid computations.
- **OpenSeesPy Integration:** Facilitates physics-based finite element simulations.
- **Flexible Loss Functions and Parameter Design:** Supports user-defined constraints, objectives, and optimization goals.

---

## **Requirements**

**Edvard Notes:**
- Should we specify package versions?
- Does this work on macOS, or just Windows? We should clarify this.

### Option 1, Manual Install

Create a new conda environment with python 3.8+ installed

```zsh
conda install python=3.8
```

First, install OpenSeesPy:

```zsh
pip install openseespy
```

Install rest of packages

```zsh
conda install numpy, torch, matplotlib, seaborn, scikit-learn
```



### Option 2, Conda Environment Install

To create a Conda environment with all dependencies, run:

```zsh
conda env create -f environment.yml
conda activate OpenPyStruct
```

---

## **Script Usage Documentation**

### **Single Load Optimizer**

 [Link to file in repo](./OpenPyStruct_BeamOpt.py)

#### **Overview**

This script optimizes the **moment of inertia distribution** along a beam to minimize structural response (e.g., deflection, internal forces) while maintaining structural efficiency. The optimization iteratively adjusts the moment of inertia of beam elements to reduce bending and shear energy losses.

#### **Key Features**

- **OpenSeesPy-based** finite element analysis.
- **PyTorch-powered** gradient-based optimization.
- Simulates **beam behavior** under various loads (point forces, distributed loads).
- **Randomized placement** of roller supports and point loads for robustness.
- Optimizes **moment of inertia** to minimize structural energy loss.
- Implements **early stopping** for computational efficiency.
- **Visualization** of optimization progress and structural response.

#### **Workflow**

1. **Define Model Parameters:**
   - Material properties (Young’s modulus, Poisson’s ratio)
   - Beam geometry (length, cross-section)
   - Boundary conditions (roller and pinned supports)
   - Loading conditions (point loads, distributed loads)

2. **Set Up the Finite Element Model:**
   - Define nodes and elements in **OpenSeesPy**
   - Apply supports and loads
   - Assign varying moment of inertia values

3. **Optimize the Moment of Inertia:**
   - Use **Adam optimizer** with **exponential learning rate decay**
   - Compute loss as a combination of:
     - **Primary loss:** Total moment of inertia
     - **Bending energy loss:** Based on internal moments
     - **Shear energy loss:** Based on shear forces
   - Update inertia values using **gradient descent**
   - Clamp inertia values to prevent negative values
   - Implement **early stopping** if loss stagnates

4. **Run Finite Element Analysis (FEA):**
   - Perform static analysis in **OpenSeesPy**
   - Extract **deflections, rotations, shear forces, and bending moments**

5. **Visualization and Results:**
   - Plot **loss history** over optimization epochs
   - Display **beam deformation, moment of inertia distribution, shear forces, and bending moments**
   - Highlight **support locations and applied forces**


### **Physics-Informed Neural Network MultiCase**

#### **Overview**

to write

#### **Key Features**

to write

#### **Workflow**

to write



---

## **Contributor Guide**

Edvard note: To ensure uniform coding standards, suggest setting up **pre-commit hooks** and linting tools:
- **Flake8** for Python style enforcement.
- Markdown style enforcement via **prettier** or **markdownlint**.

Do you want me to set this up, let me know!

---

## **License**

**MIT License**

```
MIT License

Copyright (c) 2025 Danny Smyl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```