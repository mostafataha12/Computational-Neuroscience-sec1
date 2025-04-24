# MNIST Digit Classification with PyTorch

![MNIST Examples](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Project Description

This repository contains a simple neural network implementation for classifying handwritten digits from the MNIST dataset using PyTorch. The MNIST dataset contains 70,000 grayscale images of digits 0-9, with 60,000 for training and 10,000 for testing.

## Features

- Simple two-layer neural network architecture
- Data loading and preprocessing pipeline
- Training loop with progress reporting
- Model evaluation on test set
- Visualization of correct and incorrect classifications

## Requirements

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-pytorch.git
cd mnist-pytorch
```

2. Install the required packages:
```bash
pip install torch torchvision matplotlib numpy
```

## Usage

Run the main script:
```bash
python mnist_classification.py
```

This will:
1. Download the MNIST dataset (if not already present)
2. Train the neural network for 2 epochs
3. Evaluate the model on the test set
4. Display sample correct and incorrect classifications

## Model Architecture

The neural network consists of:
1. Input layer (784 units - flattened 28x28 image)
2. Hidden layer (256 units) with ReLU activation
3. Output layer (10 units - one for each digit)

```
NeuralNet(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)
```

## Results

After 2 epochs of training, the model typically achieves:
- Training loss: ~0.15
- Test accuracy: ~95-97%

Sample output:
```
Epoch [1/2], Step [100/938], Loss: 0.4231
Epoch [1/2], Step [200/938], Loss: 0.2812
...
Epoch [2/2], Step [900/938], Loss: 0.1023
Accuracy of the network on the 10000 test images: 96.54%
```

## Visualization

The script includes visualization of:
- 5 correctly classified test images
- 5 incorrectly classified test images

Each image is displayed with both the predicted and true label.

