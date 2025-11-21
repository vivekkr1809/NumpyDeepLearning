Quick Start Guide
=================

This guide will help you get started with the NumPy Deep Learning framework.

Installation
------------

Basic installation:

.. code-block:: bash

   git clone https://github.com/vivekkr1809/NumpyDeepLearning.git
   cd NumpyDeepLearning
   pip install -e .

With GPU support:

.. code-block:: bash

   pip install -e ".[gpu]"

Your First Model
----------------

Here's a simple example of training a neural network:

.. code-block:: python

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

Using GPU
---------

To use GPU acceleration with CuPy:

.. code-block:: python

   import numpy_dl as ndl

   # Set default device to CUDA
   ndl.utils.set_default_device('cuda')

   # Or move specific tensors/models
   model = model.to('cuda')
   x = x.cuda()

Next Steps
----------

* Check out the :doc:`examples` for complete training scripts
* Read the :doc:`api` documentation for detailed API reference
* Learn about :doc:`development` to contribute to the project
