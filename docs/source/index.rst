NumPy Deep Learning Framework Documentation
==========================================

A deep learning framework built from scratch using NumPy, supporting MLP, CNN, RNN, U-Net, and ResNet architectures with CPU/GPU acceleration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api
   examples
   development

Features
--------

* Automatic differentiation with computational graph
* CPU and GPU support (via CuPy)
* Modern architectures: MLP, CNN, RNN, U-Net, ResNet
* Comprehensive utilities for training and evaluation
* Experiment tracking and visualization
* Configuration management

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install numpy-dl

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy_dl as ndl
   from numpy_dl.models import MLP

   # Create a model
   model = MLP(input_size=784, hidden_sizes=[256, 128], output_size=10)

   # Train your model
   optimizer = ndl.optim.Adam(model.parameters(), lr=0.001)
   criterion = ndl.loss.CrossEntropyLoss()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
