## **OpenPyStruct**
 is an open-source toolkit designed for machine learning-based structural optimization. Leveraging Physics-Informed Neural Networks (PINNs), Transformer-Diffusion Modules, and other state-of-the-art techniques, the framework offers unparalleled capabilities for tackling single and multi-load case optimization problems with diverse boundary and loading conditions.

## **Features**
**Physics-Informed Neural Networks (PINNs):** Embeds structural mechanics into the learning process for highly accurate predictions.\
**Transformer-Diffusion Modules:** Incorporates advanced attention mechanisms and diffusion-based techniques for modeling complex structural behavior.\
**Feedforward Neural Networks (FNNs):** Provides scalable solutions for simpler structural optimization tasks.\
**Multi-Core and GPU-Accelerated Optimization:** Enables large-scale data generation and rapid computations.\
**OpenSeesPy Integration:** Facilitates physics-based finite element simulations.\
**Flexible Loss Functions and Parameter Design:** Supports user-defined constraints, objectives, and optimization goals.

## **Requirements**
Ensure you have Python 3.8+ installed. The required libraries include:

numpy\
torch\
matplotlib\
seaborn\
scikit-learn\
openseespy\
Installation


## **Install openseespy first:**

pip install openseespy

```zsh
conda env create -f environment.yml
conda activate OpenPyStruct
```

## **MIT Open License**

Copyright <2025>  <Danny Smyl>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
