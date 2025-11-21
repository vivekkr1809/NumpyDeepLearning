"""Base Module class for neural network layers."""

from typing import Iterator, Tuple, List, Dict, Any, Optional, Union
from numpy_dl.core.parameter import Parameter
from numpy_dl.core.tensor import Tensor
from numpy_dl.utils.device import Device


class Module:
    """
    Base class for all neural network modules.

    Your models should subclass this class and implement the forward() method.
    """

    def __init__(self):
        """Initialize module."""
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, 'Module'] = {}
        self.training = True

    def forward(self, *args, **kwargs):
        """
        Forward pass. Must be implemented by subclasses.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            Output tensor(s)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        """Call forward pass."""
        return self.forward(*args, **kwargs)

    def add_parameter(self, name: str, param: Parameter):
        """
        Add a parameter to the module.

        Args:
            name: Parameter name
            param: Parameter tensor
        """
        self._parameters[name] = param
        setattr(self, name, param)

    def add_module(self, name: str, module: 'Module'):
        """
        Add a child module.

        Args:
            name: Module name
            module: Child module
        """
        self._modules[name] = module
        setattr(self, name, module)

    def parameters(self) -> Iterator[Parameter]:
        """
        Iterate over module parameters.

        Yields:
            Module parameters
        """
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = '') -> Iterator[Tuple[str, Parameter]]:
        """
        Iterate over module parameters with names.

        Args:
            prefix: Prefix for parameter names

        Yields:
            (name, parameter) tuples
        """
        for name, param in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name, param)
        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(module_prefix)

    def modules(self) -> Iterator['Module']:
        """
        Iterate over all modules.

        Yields:
            Child modules
        """
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """
        Iterate over all modules with names.

        Args:
            prefix: Prefix for module names

        Yields:
            (name, module) tuples
        """
        yield (prefix, self)
        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(module_prefix)

    def train(self, mode: bool = True):
        """
        Set module in training mode.

        Args:
            mode: Training mode flag
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Set module in evaluation mode."""
        return self.train(False)

    def zero_grad(self):
        """Zero out all parameter gradients."""
        for param in self.parameters():
            param.zero_grad()

    def to(self, device: Union[str, Device]) -> 'Module':
        """
        Move module to device.

        Args:
            device: Target device

        Returns:
            Self
        """
        if isinstance(device, str):
            device = Device(device)

        for param in self.parameters():
            param.data = param.to(device).data
            if param.grad is not None:
                param.grad = param.to(device).grad

        for module in self._modules.values():
            module.to(device)

        return self

    def cpu(self) -> 'Module':
        """Move module to CPU."""
        return self.to('cpu')

    def cuda(self) -> 'Module':
        """Move module to CUDA."""
        return self.to('cuda')

    def __setattr__(self, name: str, value: Any):
        """Override setattr to track parameters and modules."""
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, '_modules'):
                object.__setattr__(self, '_modules', {})
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __repr__(self):
        """String representation."""
        lines = [self.__class__.__name__ + '(']
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = '\n  '.join(mod_str.split('\n'))
            lines.append(f'  ({name}): {mod_str}')
        lines.append(')')
        return '\n'.join(lines)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get module state dictionary.

        Returns:
            Dictionary of parameter names to values
        """
        state = {}
        for name, param in self.named_parameters():
            state[name] = param.numpy()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load module state from dictionary.

        Args:
            state_dict: Dictionary of parameter names to values
        """
        for name, value in state_dict.items():
            parts = name.split('.')
            module = self
            for part in parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, parts[-1])
            param.data = param.device.xp.asarray(value)


class Sequential(Module):
    """
    Sequential container for modules.

    Modules are executed in the order they are added.
    """

    def __init__(self, *modules):
        """
        Initialize Sequential.

        Args:
            *modules: Modules to add
        """
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, x):
        """
        Forward pass through all modules.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx):
        """Get module by index."""
        return self._modules[str(idx)]

    def __len__(self):
        """Get number of modules."""
        return len(self._modules)


class ModuleList(Module):
    """
    List container for modules.

    Holds modules in a list and registers them properly.
    """

    def __init__(self, modules: Optional[List[Module]] = None):
        """
        Initialize ModuleList.

        Args:
            modules: List of modules
        """
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def append(self, module: Module):
        """
        Append module to list.

        Args:
            module: Module to append
        """
        idx = len(self._modules)
        self.add_module(str(idx), module)

    def forward(self, x):
        """Not implemented for ModuleList."""
        raise NotImplementedError("ModuleList has no forward method")

    def __getitem__(self, idx):
        """Get module by index."""
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        return self._modules[str(idx)]

    def __len__(self):
        """Get number of modules."""
        return len(self._modules)

    def __iter__(self):
        """Iterate over modules."""
        return iter(self._modules.values())
