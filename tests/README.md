# NumpyDeepLearning Test Suite

Comprehensive test suite for the NumpyDeepLearning framework, aiming for >99% code coverage.

## Overview

The test suite is organized into three levels:

1. **Unit Tests** (`tests/unit/`) - Test individual components in isolation
2. **Functional Tests** (`tests/functional/`) - Test complete workflows and interactions
3. **Integration Tests** (`tests/integration/`) - Test end-to-end scenarios

## Test Structure

```
tests/
├── README.md                    # This file
├── test_utils.py               # Shared test utilities
├── unit/                       # Unit tests
│   ├── test_tensor.py          # Tensor operations
│   ├── test_nn_layers.py       # Neural network layers
│   ├── test_optimizers.py      # Optimization algorithms
│   ├── test_loss_functions.py  # Loss functions
│   └── test_models.py          # Model architectures
├── functional/                 # Functional tests
│   └── test_training_workflow.py  # Training workflows
└── integration/                # Integration tests
    └── test_complete_pipelines.py # Complete pipelines
```

## Running Tests

### Using the test runner script

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run only functional tests
python run_tests.py --functional

# Run only integration tests
python run_tests.py --integration

# Run with coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py tests/unit/test_tensor.py

# Run with verbose output
python run_tests.py -v

# Run fast tests only (skip slow tests)
python run_tests.py --fast

# Stop on first failure
python run_tests.py -x
```

### Using pytest directly

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=numpy_dl --cov-report=html

# Run specific test file
pytest tests/unit/test_tensor.py

# Run specific test class
pytest tests/unit/test_tensor.py::TestTensorCreation

# Run specific test method
pytest tests/unit/test_tensor.py::TestTensorCreation::test_tensor_zeros

# Run tests matching pattern
pytest -k "test_forward"

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Run with verbose output
pytest -vv
```

### Using unittest

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.unit.test_tensor

# Run specific test class
python -m unittest tests.unit.test_tensor.TestTensorCreation

# Run specific test method
python -m unittest tests.unit.test_tensor.TestTensorCreation.test_tensor_zeros
```

## Test Coverage

The test suite aims for >99% code coverage across all components:

### Viewing Coverage Reports

After running tests with coverage:

```bash
python run_tests.py --coverage
```

View the HTML report:

```bash
# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

Or view in terminal:

```bash
pytest --cov=numpy_dl --cov-report=term-missing
```

## Test Categories

### Unit Tests

Test individual components in isolation:

- **test_tensor.py**: Core tensor operations
  - Creation (zeros, ones, randn, rand)
  - Arithmetic operations (+, -, *, /, @, **)
  - Reductions (sum, mean)
  - Reshaping (reshape, transpose)
  - Element-wise operations (exp, log, sqrt, clip)
  - Indexing and slicing
  - Gradient computation

- **test_nn_layers.py**: Neural network layers
  - Linear layers
  - Convolutional layers (Conv2d, ConvTranspose2d)
  - Pooling layers (MaxPool2d, AvgPool2d)
  - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
  - Dropout
  - Normalization (BatchNorm, LayerNorm)
  - Embedding
  - Attention mechanisms

- **test_optimizers.py**: Optimization algorithms
  - SGD (with momentum, weight decay, Nesterov)
  - Adam (with bias correction)
  - AdamW (decoupled weight decay)
  - RMSprop (with momentum)

- **test_loss_functions.py**: Loss functions
  - MSELoss
  - CrossEntropyLoss
  - NLLLoss
  - BCELoss
  - BCEWithLogitsLoss
  - VAELoss
  - KLDivergenceLoss

- **test_models.py**: Model architectures
  - MLP (Multi-Layer Perceptron)
  - SimpleCNN, VGG
  - ResNet (ResNet18, ResNet34, etc.)
  - UNet
  - RNN models (SimpleRNN, Seq2Seq)
  - Autoencoders (Autoencoder, VAE)
  - Transformers (GPT, Encoder, Decoder)

### Functional Tests

Test complete workflows:

- **test_training_workflow.py**: Training workflows
  - Basic training loops
  - Batch training with DataLoader
  - Multi-epoch training
  - Optimizer comparison
  - Train/eval mode switching
  - Gradient flow
  - Overfitting small datasets

### Integration Tests

Test end-to-end scenarios:

- **test_complete_pipelines.py**: Complete pipelines
  - MNIST-style classification (MLP and CNN)
  - Autoencoder reconstruction
  - VAE training
  - GPT character-level language modeling
  - Full classification pipeline
  - Model save/load
  - Robustness testing

## Writing New Tests

### Test Template

```python
"""Tests for new component."""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from numpy_dl.core.tensor import Tensor
from tests.test_utils import assert_tensors_close, assert_arrays_close


class TestNewComponent(unittest.TestCase):
    """Test new component."""

    def test_creation(self):
        """Test creating component."""
        component = NewComponent()
        self.assertIsNotNone(component)

    def test_forward(self):
        """Test forward pass."""
        component = NewComponent()
        x = Tensor(np.random.randn(2, 10))
        y = component(x)
        self.assertEqual(y.shape, (2, 10))

    def test_backward(self):
        """Test backward pass."""
        component = NewComponent()
        x = Tensor(np.random.randn(2, 10), requires_grad=True)
        y = component(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


if __name__ == '__main__':
    unittest.main()
```

### Best Practices

1. **Test Naming**: Use descriptive names starting with `test_`
2. **Isolation**: Each test should be independent
3. **Assertions**: Use appropriate assertions (assertEqual, assertAlmostEqual, etc.)
4. **Coverage**: Test both success and failure cases
5. **Edge Cases**: Test boundary conditions, empty inputs, etc.
6. **Documentation**: Add docstrings explaining what each test does
7. **Fixtures**: Use setUp/tearDown for common setup code
8. **Parametrization**: Use subtests or parametrize for testing multiple cases

## Test Utilities

Common utilities in `test_utils.py`:

```python
# Assert tensors are close
assert_tensors_close(t1, t2, rtol=1e-5, atol=1e-8)

# Assert arrays are close
assert_arrays_close(a1, a2, rtol=1e-5, atol=1e-8)

# Compute numerical gradient
grad = numerical_gradient(func, x, eps=1e-5)

# Check analytical vs numerical gradient
check_gradient(func, x, analytical_grad)

# Create synthetic datasets
X, y = create_simple_dataset(n_samples=100, n_features=10, n_classes=3)
X, y = create_regression_dataset(n_samples=100, n_features=5)

# Count model parameters
num_params = count_parameters(model)

# Set random seed
set_seed(42)
```

## Continuous Integration

The test suite is designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=numpy_dl --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Debugging Failed Tests

### Using pytest debugger

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb
```

### Using print statements

```python
def test_example(self):
    result = some_function()
    print(f"Result: {result}")  # Will show in pytest output with -s
    self.assertEqual(result, expected)
```

```bash
# Show print statements
pytest -s
```

### Using logging

```python
import logging
logger = logging.getLogger(__name__)

def test_example(self):
    logger.info("Testing something")
    result = some_function()
    logger.debug(f"Result: {result}")
    self.assertEqual(result, expected)
```

```bash
# Show log output
pytest --log-cli-level=DEBUG
```

## Performance Testing

For performance-critical code, use `pytest-benchmark`:

```bash
pip install pytest-benchmark
```

```python
def test_tensor_multiplication_performance(benchmark):
    """Benchmark tensor multiplication."""
    a = Tensor(np.random.randn(100, 100))
    b = Tensor(np.random.randn(100, 100))

    result = benchmark(lambda: a @ b)
```

## Coverage Goals

Target coverage by component:

- Core (Tensor, Module, Parameter): >99%
- Neural Network Layers: >98%
- Optimizers: >98%
- Loss Functions: >98%
- Models: >95%
- Data utilities: >95%
- Overall: >99%

## Known Issues and Limitations

- Some GPU-specific tests are skipped if CUDA is not available
- Very large models may be tested with reduced sizes for speed
- Some random tests may have rare flakiness (use fixed seeds)

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Add docstrings to tests
5. Update this README if needed

## Questions?

For questions about the test suite, please open an issue on GitHub.
