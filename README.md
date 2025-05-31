# Multi-GPU-Distributed-CNN-for-Multi-Class-Character-Recognition

A comprehensive study of parallel computing techniques for accelerating Convolutional Neural Network (CNN) training on character recognition tasks, comparing multi-GPU and multi-CPU parallelization strategies.

## Project Overview

This project implements and analyzes various parallel computing approaches for training CNNs on the TMNIST dataset, which contains 274,093 character images across 94 unique character classes. We explore distributed training techniques to optimize both training speed and model accuracy through systematic performance evaluation.

## Key Features

- **Multi-GPU Parallelization**: Implementation using PyTorch's Distributed Data Parallel (DDP)
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) for enhanced performance
- **Comprehensive Performance Analysis**: Detailed comparison of speedup, efficiency, and accuracy metrics
- **Memory Optimization**: Analysis of memory usage across different configurations
- **Throughput Evaluation**: Training samples processed per second across various setups


## Technologies Used

- **Deep Learning Framework**: PyTorch
- **Parallel Computing**:
    - PyTorch DistributedDataParallel (DDP)
    - PyTorch Automatic Mixed Precision (AMP)
    - Python Joblib for CPU parallelization
- **Hardware**:
    - NVIDIA Tesla P100-PCIE-12GB GPUs
    - NVIDIA H100 80GB HBM3 GPUs
    - Multi-core CPU systems
- **Dataset**: TMNIST (Typeface MNIST) with 94 character classes

## 📊 Dataset

**TMNIST Dataset Specifications:**

- **Size**: 274,093 character samples
- **Classes**: 94 unique characters (uppercase, lowercase, digits, symbols)
- **Image Resolution**: 28×28 pixels (grayscale)
- **Format**: CSV file (940.82 MB)
- **Features**: 786 columns (1 font name + 1 label + 784 pixel values)


## 🏗️ Model Architecture

**CNN Architecture:**

- **Input Layer**: 28×28 grayscale images
- **Convolutional Layers**: 2 layers with 32 and 64 filters (3×3 kernels)
- **Activation**: ReLU activation functions
- **Pooling**: Max pooling operations
- **Dense Layers**: Fully connected layers for classification
- **Output**: 94-class character classification

## Key Results

### Multi-GPU Performance (DDP)

- **Best Configuration**: 4 GPUs with batch size 64
- **Speedup**: 3.7× compared to single GPU
- **Efficiency**: 92% parallel efficiency
- **Accuracy**: 93.25% test accuracy
- **Training Time**: 45.4 seconds


### Mixed Precision Training (AMP)

- **Performance Gain**: 38-66% faster than standard precision
- **Best Throughput**: 208,131 samples/second (4 GPUs, batch 2048)
- **Memory Efficiency**: Effective memory utilization
- **Accuracy Preservation**: Comparable to full-precision training


### Multi-CPU Performance

- **Limited Speedup**: Maximum 1.17× with 4 CPUs
- **Ensemble Benefits**: Up to 1.1% accuracy improvement
- **Best Configuration**: 2 CPUs with ensemble approach


##  Performance Comparison

| Configuration | Training Time | Speedup | Test Accuracy | Efficiency |
| :-- | :-- | :-- | :-- | :-- |
| 1 GPU (DDP) | 166.22s | 1.0× | 93.25% | 100% |
| 4 GPU (DDP) | 45.43s | 3.7× | 93.25% | 92% |
| 4 GPU (AMP) | 10.71s | 15.5× | 88.68% | 97% |

## Analysis Highlights

### GPU Parallelization Insights

- **Near-linear scaling** up to 4 GPUs with diminishing returns beyond
- **Batch size impact**: Smaller batches maintain higher accuracy
- **Communication overhead**: Minimal impact on performance
- **Memory distribution**: 31% reduction in per-GPU memory usage


### Mixed Precision Benefits

- **Dramatic speedup** with maintained accuracy
- **Memory efficiency** for larger batch processing
- **Throughput optimization** reaching 3× improvement
- **Numerical stability** preserved through gradient scaling

## References
1. [TMNIST Alphabet (94 characters) Dataset](https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters)[^1]
2. [PyTorch Distributed Data Parallel Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)[^1]
3. [Joblib for Parallel Model Training](https://discuss.pytorch.org/t/use-joblib-to-train-an-ensemble-of-small-models-on-the-same-gpu-in-parallel/157831)[^1]

##  Future Work

- **FSDP Implementation**: Fully Sharded Data Parallel for larger models
- **Advanced Mixed Precision**: Exploration of FP8 and other precision formats
- **Cross-node Scaling**: Multi-node distributed training analysis
- **Alternative Architectures**: Testing with Transformer-based models
- **Resource Optimization**: Dynamic batch sizing and adaptive parallelization
