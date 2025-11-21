Development Guide
=================

This guide explains how to contribute to the NumPy Deep Learning framework.

Setting Up Development Environment
-----------------------------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/vivekkr1809/NumpyDeepLearning.git
   cd NumpyDeepLearning

2. Install in development mode:

.. code-block:: bash

   pip install -e ".[dev]"

3. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Code Style
----------

We follow PEP 8 style guidelines. Use these tools:

.. code-block:: bash

   # Format code
   black numpy_dl/

   # Lint
   flake8 numpy_dl/

   # Type checking
   mypy numpy_dl/

Running Tests
-------------

Run all tests:

.. code-block:: bash

   pytest tests/

Run with coverage:

.. code-block:: bash

   pytest --cov=numpy_dl tests/

Building Documentation
----------------------

Build the documentation locally:

.. code-block:: bash

   cd docs
   make html

The documentation will be available at ``docs/build/html/index.html``.

Contributing Guidelines
-----------------------

1. Create a new branch for your feature
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request

Code Structure
--------------

The codebase is organized as follows:

* ``numpy_dl/core/``: Core tensor and autograd functionality
* ``numpy_dl/nn/``: Neural network layers
* ``numpy_dl/models/``: Pre-built model architectures
* ``numpy_dl/optim/``: Optimization algorithms
* ``numpy_dl/loss/``: Loss functions
* ``numpy_dl/data/``: Data loading utilities
* ``numpy_dl/utils/``: General utilities
* ``numpy_dl/tracking/``: Experiment tracking

Adding New Features
-------------------

When adding new features:

1. Follow existing code patterns
2. Add docstrings (Google style)
3. Include type hints
4. Write unit tests
5. Update relevant documentation
6. Add examples if applicable
