import torch.nn as nn

from parameters import ExperimentConfig


class MLP(nn.Module):
    """Multi-layer perceptron for MNIST classification.

    Hidden layers are stored in a ModuleList where each entry is a
    Sequential block. The order depends on bn_after_activation:
      - False (default): Linear -> BatchNorm1d -> Activation -> Dropout
      - True           : Linear -> Activation -> BatchNorm1d -> Dropout

    Args:
        experimentConfig: ExperimentConfig holding architecture and regularization settings.
        input_size: Number of input features (784 for MNIST). This is default and cannot be changed.
        num_classes: Number of output classes (10 for MNIST). This is default and cannot be changed.
    """

    def __init__(self, experimentConfig: ExperimentConfig, input_size: int, num_classes: int) -> None:
        super().__init__()

        activation_map = {"relu": nn.ReLU, "gelu": nn.GELU}
        act_cls = activation_map[experimentConfig.activation]

        self.flatten = nn.Flatten()

        # Build hidden layers with ModuleList; each block uses Sequential
        self.hidden_layers = nn.ModuleList()
        in_dim = input_size
        for h in experimentConfig.hidden_sizes:
            block = [nn.Linear(in_dim, h)]
            if experimentConfig.use_bn and not experimentConfig.bn_after_activation:
                block.append(nn.BatchNorm1d(h))
            block.append(act_cls())
            if experimentConfig.use_bn and experimentConfig.bn_after_activation:
                block.append(nn.BatchNorm1d(h))
            if experimentConfig.dropout > 0.0:
                block.append(nn.Dropout(experimentConfig.dropout))
            self.hidden_layers.append(nn.Sequential(*block))
            in_dim = h

        self.output_layer = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
