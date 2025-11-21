"""Multi-Task Learning model architectures.

This module implements state-of-the-art multi-task learning architectures:
1. Hard Parameter Sharing: Shared backbone with task-specific heads
2. Soft Parameter Sharing: Separate networks with cross-task regularization
3. Multi-Head Architecture: Flexible multi-task model wrapper

References:
    - Caruana, R. "Multitask Learning" (Machine Learning, 1997)
    - Ruder, S. "An Overview of Multi-Task Learning in Deep Neural Networks" (2017)
    - Crawshaw, M. "Multi-Task Learning with Deep Neural Networks: A Survey" (2020)
"""

from typing import Dict, List, Optional, Union, Callable
import numpy as np
from numpy_dl.core.module import Module, ModuleList, Sequential
from numpy_dl.core.tensor import Tensor
from numpy_dl.nn import Linear


class TaskHead(Module):
    """
    Task-specific head for multi-task learning.

    A task head is a small network that processes shared features and produces
    task-specific outputs. Typically consists of a few fully-connected layers.

    Args:
        input_dim: Input feature dimension from shared backbone
        output_dim: Output dimension for the task
        hidden_dims: List of hidden layer dimensions (default: [])
        activation: Activation function to use (default: None)
        dropout: Dropout rate (default: 0.0)

    Example:
        >>> head = TaskHead(input_dim=512, output_dim=10, hidden_dims=[256, 128])
        >>> output = head(shared_features)  # (batch_size, 10)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[Callable] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = []

        # Build task-specific network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(Linear(prev_dim, hidden_dim))
            if activation is not None:
                layers.append(activation)
            if dropout > 0:
                from numpy_dl.nn import Dropout
                layers.append(Dropout(dropout))
            prev_dim = hidden_dim

        # Final output layer
        layers.append(Linear(prev_dim, output_dim))

        self.network = Sequential(*layers) if layers else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through task head.

        Args:
            x: Shared features from backbone (batch_size, input_dim)

        Returns:
            Task-specific output (batch_size, output_dim)
        """
        if self.network is not None:
            return self.network(x)
        return x


class HardParameterSharing(Module):
    """
    Hard parameter sharing architecture for multi-task learning.

    The most common multi-task learning architecture where all tasks share
    a common backbone network, with task-specific heads for each task.

    Architecture:
        Input -> Shared Backbone -> Task Head 1 -> Output 1
                                  -> Task Head 2 -> Output 2
                                  -> Task Head 3 -> Output 3
                                  -> ...

    Args:
        backbone: Shared backbone network
        task_heads: Dictionary mapping task names to TaskHead modules
        task_output_dims: Dictionary mapping task names to output dimensions
        shared_head_dim: Optional shared head dimension before task-specific heads

    Example:
        >>> from numpy_dl.models import MLP
        >>> backbone = MLP(input_dim=784, hidden_dims=[512, 256], output_dim=128)
        >>> task_heads = {
        ...     'digit': TaskHead(128, 10),
        ...     'parity': TaskHead(128, 2)
        ... }
        >>> model = HardParameterSharing(backbone, task_heads)
        >>> outputs = model({'input': x})  # {'digit': out1, 'parity': out2}
    """

    def __init__(
        self,
        backbone: Module,
        task_heads: Dict[str, TaskHead],
        shared_head_dim: Optional[int] = None
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = list(task_heads.keys())
        self.shared_head_dim = shared_head_dim

        # Register backbone
        self.add_module('backbone', backbone)

        # Optional shared head
        if shared_head_dim is not None:
            self.shared_head = Linear(backbone.output_dim, shared_head_dim)
            self.add_module('shared_head', self.shared_head)
        else:
            self.shared_head = None

        # Register task heads
        for task_name, head in task_heads.items():
            self.add_module(f'head_{task_name}', head)

    def get_task_head(self, task_name: str) -> TaskHead:
        """Get the head for a specific task."""
        return getattr(self, f'head_{task_name}')

    def forward(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        return_shared: bool = False
    ) -> Union[Dict[str, Tensor], tuple]:
        """
        Forward pass through shared backbone and task-specific heads.

        Args:
            x: Input tensor or dict with 'input' key
            return_shared: If True, return (task_outputs, shared_features)

        Returns:
            Dictionary mapping task names to outputs,
            or (task_outputs, shared_features) if return_shared=True
        """
        # Handle dict input
        if isinstance(x, dict):
            x = x['input']

        # Shared backbone
        shared_features = self.backbone(x)

        # Optional shared head
        if self.shared_head is not None:
            shared_features = self.shared_head(shared_features)

        # Task-specific heads
        outputs = {}
        for task_name in self.task_names:
            head = self.get_task_head(task_name)
            outputs[task_name] = head(shared_features)

        if return_shared:
            return outputs, shared_features
        return outputs

    def get_shared_parameters(self):
        """Get parameters from the shared backbone."""
        return self.backbone.parameters()

    def get_task_parameters(self, task_name: str):
        """Get parameters for a specific task head."""
        head = self.get_task_head(task_name)
        return head.parameters()


class SoftParameterSharing(Module):
    """
    Soft parameter sharing architecture for multi-task learning.

    Each task has its own network, but networks are encouraged to share
    similar parameters through regularization (e.g., L2 distance between parameters).

    This architecture provides more flexibility than hard sharing but requires
    more parameters and computation.

    Args:
        task_networks: Dictionary mapping task names to task-specific networks
        sharing_strength: Regularization strength for parameter sharing (default: 0.01)
        sharing_groups: Optional list of task groups that should share more strongly

    Example:
        >>> from numpy_dl.models import MLP
        >>> task_networks = {
        ...     'task1': MLP(784, [256, 128], 10),
        ...     'task2': MLP(784, [256, 128], 2)
        ... }
        >>> model = SoftParameterSharing(task_networks, sharing_strength=0.01)
        >>> outputs = model({'input': x})  # {'task1': out1, 'task2': out2}
    """

    def __init__(
        self,
        task_networks: Dict[str, Module],
        sharing_strength: float = 0.01,
        sharing_groups: Optional[List[List[str]]] = None
    ):
        super().__init__()
        self.task_names = list(task_networks.keys())
        self.sharing_strength = sharing_strength
        self.sharing_groups = sharing_groups

        # Register task networks
        for task_name, network in task_networks.items():
            self.add_module(f'network_{task_name}', network)

    def get_task_network(self, task_name: str) -> Module:
        """Get the network for a specific task."""
        return getattr(self, f'network_{task_name}')

    def forward(
        self,
        x: Union[Tensor, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """
        Forward pass through task-specific networks.

        Args:
            x: Input tensor or dict with 'input' key

        Returns:
            Dictionary mapping task names to outputs
        """
        # Handle dict input
        if isinstance(x, dict):
            x = x['input']

        # Pass through each task network
        outputs = {}
        for task_name in self.task_names:
            network = self.get_task_network(task_name)
            outputs[task_name] = network(x)

        return outputs

    def compute_sharing_loss(self) -> Tensor:
        """
        Compute soft parameter sharing regularization loss.

        Encourages parameters across task networks to be similar by computing
        L2 distance between corresponding parameters.

        Returns:
            Sharing regularization loss
        """
        if len(self.task_names) < 2:
            return Tensor(np.array(0.0))

        sharing_loss = Tensor(np.array(0.0))
        num_pairs = 0

        # If sharing groups specified, only regularize within groups
        if self.sharing_groups is not None:
            task_pairs = []
            for group in self.sharing_groups:
                for i, task1 in enumerate(group):
                    for task2 in group[i + 1:]:
                        task_pairs.append((task1, task2))
        else:
            # Regularize all pairs
            task_pairs = []
            for i, task1 in enumerate(self.task_names):
                for task2 in self.task_names[i + 1:]:
                    task_pairs.append((task1, task2))

        # Compute pairwise parameter distances
        for task1, task2 in task_pairs:
            net1_params = list(self.get_task_network(task1).parameters())
            net2_params = list(self.get_task_network(task2).parameters())

            # Only regularize if networks have same structure
            if len(net1_params) == len(net2_params):
                for p1, p2 in zip(net1_params, net2_params):
                    if p1.data.shape == p2.data.shape:
                        diff = p1 - p2
                        sharing_loss = sharing_loss + (diff * diff).sum()
                        num_pairs += 1

        if num_pairs > 0:
            sharing_loss = sharing_loss * (self.sharing_strength / num_pairs)

        return sharing_loss


class MultiTaskModel(Module):
    """
    Flexible multi-task model wrapper with support for various architectures.

    This wrapper provides a unified interface for multi-task learning models,
    supporting both hard and soft parameter sharing, as well as custom architectures.

    Args:
        architecture: Base multi-task architecture (HardParameterSharing or SoftParameterSharing)
        task_names: List of task names
        use_soft_sharing: If True, add soft parameter sharing regularization

    Example:
        >>> backbone = MLP(784, [512, 256], 128)
        >>> task_heads = {
        ...     'digit': TaskHead(128, 10),
        ...     'parity': TaskHead(128, 2)
        ... }
        >>> architecture = HardParameterSharing(backbone, task_heads)
        >>> model = MultiTaskModel(architecture, ['digit', 'parity'])
        >>> outputs = model(x)  # {'digit': out1, 'parity': out2}
    """

    def __init__(
        self,
        architecture: Module,
        task_names: List[str],
        use_soft_sharing: bool = False
    ):
        super().__init__()
        self.architecture = architecture
        self.task_names = task_names
        self.use_soft_sharing = use_soft_sharing

        self.add_module('architecture', architecture)

    def forward(self, x: Union[Tensor, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Forward pass through multi-task architecture.

        Args:
            x: Input tensor or dict

        Returns:
            Dictionary mapping task names to outputs
        """
        return self.architecture(x)

    def compute_auxiliary_loss(self) -> Optional[Tensor]:
        """
        Compute auxiliary losses (e.g., soft parameter sharing regularization).

        Returns:
            Auxiliary loss or None
        """
        if self.use_soft_sharing and isinstance(self.architecture, SoftParameterSharing):
            return self.architecture.compute_sharing_loss()
        return None


def create_hard_sharing_model(
    backbone: Module,
    task_configs: Dict[str, Dict],
    shared_head_dim: Optional[int] = None
) -> HardParameterSharing:
    """
    Factory function to create a hard parameter sharing model.

    Args:
        backbone: Shared backbone network
        task_configs: Dictionary mapping task names to config dicts with keys:
            - 'output_dim': Output dimension for the task
            - 'hidden_dims': Optional list of hidden dimensions for task head
            - 'activation': Optional activation function
            - 'dropout': Optional dropout rate
        shared_head_dim: Optional dimension for shared head layer

    Returns:
        HardParameterSharing model

    Example:
        >>> from numpy_dl.models import MLP
        >>> from numpy_dl.nn import ReLU
        >>> backbone = MLP(784, [512, 256], 128)
        >>> task_configs = {
        ...     'digit': {'output_dim': 10, 'hidden_dims': [64]},
        ...     'parity': {'output_dim': 2, 'activation': ReLU()}
        ... }
        >>> model = create_hard_sharing_model(backbone, task_configs)
    """
    # Get backbone output dimension
    # Try to infer from last layer
    backbone_dim = None
    if hasattr(backbone, 'output_dim'):
        backbone_dim = backbone.output_dim
    elif hasattr(backbone, 'fc') and hasattr(backbone.fc, 'out_features'):
        backbone_dim = backbone.fc.out_features
    else:
        # Try to find last Linear layer
        for module in reversed(list(backbone.modules())):
            if hasattr(module, 'out_features'):
                backbone_dim = module.out_features
                break

    if backbone_dim is None:
        raise ValueError("Could not infer backbone output dimension. "
                         "Please specify output_dim attribute on backbone.")

    # Create task heads
    task_heads = {}
    input_dim = shared_head_dim if shared_head_dim is not None else backbone_dim

    for task_name, config in task_configs.items():
        output_dim = config['output_dim']
        hidden_dims = config.get('hidden_dims', [])
        activation = config.get('activation', None)
        dropout = config.get('dropout', 0.0)

        task_heads[task_name] = TaskHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout
        )

    return HardParameterSharing(backbone, task_heads, shared_head_dim)


def create_soft_sharing_model(
    input_dim: int,
    task_configs: Dict[str, Dict],
    sharing_strength: float = 0.01,
    sharing_groups: Optional[List[List[str]]] = None
) -> SoftParameterSharing:
    """
    Factory function to create a soft parameter sharing model.

    Args:
        input_dim: Input dimension
        task_configs: Dictionary mapping task names to config dicts with keys:
            - 'network': Pre-built network for the task, OR
            - 'hidden_dims': List of hidden dimensions
            - 'output_dim': Output dimension
        sharing_strength: Regularization strength for parameter sharing
        sharing_groups: Optional list of task groups

    Returns:
        SoftParameterSharing model

    Example:
        >>> task_configs = {
        ...     'task1': {'hidden_dims': [256, 128], 'output_dim': 10},
        ...     'task2': {'hidden_dims': [256, 128], 'output_dim': 2}
        ... }
        >>> model = create_soft_sharing_model(784, task_configs)
    """
    from numpy_dl.models import MLP

    task_networks = {}
    for task_name, config in task_configs.items():
        if 'network' in config:
            task_networks[task_name] = config['network']
        else:
            hidden_dims = config.get('hidden_dims', [256, 128])
            output_dim = config['output_dim']
            task_networks[task_name] = MLP(input_dim, hidden_dims, output_dim)

    return SoftParameterSharing(task_networks, sharing_strength, sharing_groups)
