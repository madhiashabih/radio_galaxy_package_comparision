# Comparative Analysis of PyTorch, TensorFlow, and JAX for Morphological Classification of Radio Galaxies

This project provides a systematic comparison of three leading machine learning frameworks—**PyTorch**, **TensorFlow**, and **JAX**—focusing on their application in the morphological classification of radio galaxies. 

## Overview

The study evaluates these frameworks across several performance metrics:
- **Model Accuracy**
- **F1 Score**
- **Training and Inference Times**
- **Memory Utilization**
- **API Design and Usability**

### Datasets and Models

For radio galaxy classification, we employed four state-of-the-art architectures:
- **ConvXpress**
- **First-class**
- **MCRGNet**
- **Toothless**

These models were tested on the **MiraBest** and **FR-DEEP** datasets.

In addition, a simple convolutional neural network (CNN) was implemented on the **MNIST** dataset to serve as a standardized benchmark across frameworks. Due to certain implementation constraints, **JAX** evaluations were limited to the MNIST experiments.

### Key Findings

Each framework demonstrated unique advantages:
- **TensorFlow**: Excelled in training efficiency, making it suitable for large-scale training tasks.
- **PyTorch**: Offered better memory management and a developer-friendly experience, enhancing flexibility.
- **JAX**: Achieved the fastest inference times, making it particularly useful for real-time classification tasks.

## Requirements

To set up the environment, install all dependencies with:

```bash
pip install -r requirements.txt
Running the Experiments
Navigate to the code directory and execute the scripts as follows:

TensorFlow (MNIST and Radio Galaxy Models)

cd code
python3 mnist_tf.py <run>
PyTorch (MNIST and Radio Galaxy Models)

cd code
python3 mnist_py.py <run>
JAX (MNIST only)

cd code
python3 mnist_jax.py <run>
Replace <run> with any specific command-line arguments if required by the script.

Directory Structure
code/: Contains framework-specific scripts for each model and dataset.
requirements.txt: Lists all required Python packages.
Additional Notes
This comparative analysis provides practical guidance for researchers and practitioners, highlighting framework-specific trade-offs to aid in selecting appropriate tools for radio galaxy classification and similar computer vision tasks.
