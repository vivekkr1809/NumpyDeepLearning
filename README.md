# NumPy Deep Learning Framework

A comprehensive deep learning framework built from scratch using NumPy, featuring automatic differentiation, GPU support, and modern architectures.

## Features

### 🎯 Core Capabilities
- **Automatic Differentiation**: Full backpropagation support with computational graph
- **CPU/GPU Support**: Seamless switching between CPU and CUDA (via CuPy)
- **Modular Design**: Clean, extensible architecture similar to PyTorch

### 🏗️ Supported Architectures
- **MLP** (Multi-Layer Perceptron): Fully connected networks
- **CNN** (Convolutional Neural Networks): Image processing and computer vision
- **RNN/LSTM/GRU**: Sequence modeling and NLP tasks
- **U-Net**: Image segmentation
- **ResNet**: Deep residual networks (ResNet-18/34/50/101/152)
- **Multi-Task Learning**: Hard/soft parameter sharing with state-of-the-art loss weighting

### 🛠️ Components
- **Layers**: Linear, Conv2d, MaxPool2d, AvgPool2d, BatchNorm, LayerNorm, Dropout
- **Activations**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, CrossEntropy, NLLLoss, BCE, BCEWithLogits
- **Multi-Task Loss Weighting**: Uncertainty Weighting, GradNorm, Dynamic Weight Average
- **Optimizers**: SGD, Adam, AdamW, RMSprop
- **Data Loading**: Dataset, DataLoader with batching and shuffling
- **Experiment Tracking**: Comprehensive logging and visualization
- **Configuration**: YAML/JSON-based configuration management

## Installation

### Basic Installation
```bash
git clone https://github.com/vivekkr1809/NumpyDeepLearning.git
cd NumpyDeepLearning
pip install -e .
```

### With GPU Support
```bash
pip install -e ".[gpu]"
```

### With All Features
```bash
pip install -e ".[gpu,data,docs,dev]"
```

## Quick Start

### Simple Classification Example
```python
import numpy_dl as ndl
from numpy_dl.models import MLP
from numpy_dl.optim import Adam
from numpy_dl.loss import CrossEntropyLoss

# Create model
model = MLP(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    dropout=0.5
)

# Setup training
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for x_batch, y_batch in dataloader:
        # Convert to tensors
        x = ndl.tensor(x_batch, requires_grad=False)
        y = ndl.tensor(y_batch)

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Using GPU
```python
import numpy_dl as ndl

# Set default device
ndl.utils.set_default_device('cuda')

# Or move specific tensors/models
model = model.to('cuda')
x = x.cuda()
```

### Configuration-Based Training
```python
from numpy_dl.utils import Config

# Load configuration
config = Config.from_file('configs/train_config.yaml')

# Access parameters
batch_size = config.get('training.batch_size')
learning_rate = config.get('training.learning_rate')
```

## Examples

The `examples/` directory contains complete training scripts:

### Vision
- `examples/vision/mnist_classification.py` - CNN for MNIST digit classification
- `examples/vision/image_segmentation.py` - U-Net for image segmentation

### NLP
- `examples/nlp/text_classification.py` - RNN/LSTM for text classification

### Audio
- `examples/audio/audio_classification.py` - CNN for audio spectrogram classification

Run an example:
```bash
python examples/vision/mnist_classification.py
```

## Architecture Overview

```
numpy_dl/
├── core/           # Tensor operations and autograd
│   ├── tensor.py
│   ├── parameter.py
│   ├── module.py
│   └── functional.py
├── nn/             # Neural network layers
│   ├── linear.py
│   ├── conv.py
│   ├── rnn.py
│   ├── activation.py
│   └── normalization.py
├── models/         # Pre-built architectures
│   ├── mlp.py
│   ├── cnn.py
│   ├── resnet.py
│   ├── unet.py
│   └── rnn_models.py
├── optim/          # Optimizers
│   ├── sgd.py
│   ├── adam.py
│   └── rmsprop.py
├── loss/           # Loss functions
│   ├── mse.py
│   ├── cross_entropy.py
│   └── bce.py
├── data/           # Data loading utilities
│   ├── dataset.py
│   └── dataloader.py
├── utils/          # Utilities
│   ├── device.py
│   ├── config.py
│   ├── metrics.py
│   └── visualization.py
└── tracking/       # Experiment tracking
    └── tracker.py
```

## Advanced Usage

### Custom Models
```python
from numpy_dl import Module
from numpy_dl.nn import Linear, ReLU

class CustomModel(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### Experiment Tracking
```python
from numpy_dl.tracking import ExperimentTracker

tracker = ExperimentTracker('my_experiment')

# Log hyperparameters
tracker.log_hyperparameters(
    learning_rate=0.001,
    batch_size=32,
    epochs=10
)

# Log metrics
tracker.log_metrics(epoch=0, train_loss=0.5, train_acc=0.85)

# Save checkpoints
tracker.log_model_checkpoint(model.state_dict(), epoch=5)

# Finish experiment
tracker.finish()
```

### Visualization
```python
from numpy_dl.utils import plot_training_history, plot_confusion_matrix

# Plot training curves
history = {'loss': [...], 'accuracy': [...]}
plot_training_history(history, save_path='plots/history.png')

# Plot confusion matrix
cm = confusion_matrix(predictions, targets)
plot_confusion_matrix(cm, class_names=['cat', 'dog'])
```

## Performance Considerations

### CPU vs GPU
- **CPU**: Works out of the box with NumPy
- **GPU**: Requires CuPy installation, ~10-50x speedup on large models

### Memory Management
- Use `tensor.detach()` to break computation graph when not needed
- Call `model.eval()` during inference to disable dropout/batch norm training mode
- Use smaller batch sizes for memory-constrained environments

### Optimization Tips
- Use `Adam` or `AdamW` for faster convergence
- Apply learning rate scheduling for better final performance
- Use batch normalization for deeper networks
- Apply gradient clipping for RNNs: `clip_grad_norm_(model.parameters(), max_norm=1.0)`

## API Documentation

Full API documentation is available in the `docs/` directory. Build it using:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in your browser.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black numpy_dl/

# Lint
flake8 numpy_dl/

# Type checking
mypy numpy_dl/
```

## Limitations

- This is an educational framework demonstrating deep learning concepts
- Performance may not match production frameworks like PyTorch or TensorFlow
- Some advanced features (e.g., distributed training) are not implemented
- Convolution operations use a simplified im2col approach

## Roadmap

- [ ] Additional architectures (Transformers, GANs)
- [ ] More optimizers (Adagrad, Adadelta)
- [ ] Learning rate schedulers
- [ ] Model zoo with pretrained weights
- [ ] Better integration with HuggingFace datasets
- [ ] Kaggle API integration for data loading
- [ ] Mixed precision training
- [ ] Model quantization

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by PyTorch's design philosophy
- Built for educational purposes to understand deep learning internals
- Thanks to the NumPy and CuPy teams for the underlying array operations

## Citation

If you use this framework in your research or projects, please cite:

```bibtex
@software{numpy_deep_learning,
  title={NumPy Deep Learning Framework},
  author={NumpyDeepLearning Contributors},
  year={2025},
  url={https://github.com/vivekkr1809/NumpyDeepLearning}
}
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Note**: This framework is designed for educational purposes and understanding deep learning internals. For production use cases, consider using established frameworks like PyTorch or TensorFlow.
