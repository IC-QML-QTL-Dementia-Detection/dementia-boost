"""Classical fully connected classification heads for dementia detection.

This module provides the classical heads of the transfer-learning paradigm and
their registry:

- `ClassicalClassifierHead` (`paper`): the base paper's head, a dense layer with
  dropout and ReLU followed by a single output unit.
- `TanhBottleneckHead` (`tanh_bottleneck`): the quantum head's classical layers
  without the circuit (pre-net, `tanh` angle scaling, post-net), a control that
  tells the effect of the circuit apart from the effect of the layers around it.

`build_classical_head` builds a head by its registry name, and
`classical_head_for` builds the head a run specification describes.
"""

import math

import torch.nn as nn
from torch import Tensor, tanh

from dementia_boost.core.identity import DEFAULT_CLASSICAL_HEAD, Paradigm, RunSpec


class ClassicalClassifierHead(nn.Module):
    """Classical dense classification head for binary prediction.

    Flattens spatial feature maps, passes them through a dense bottleneck
    (2304 -> 5) with dropout regularization (p=0.5) and ReLU activation,
    and projects down to a single output unit (5 -> 1).

    Attributes:
        DEFAULT_IN_FEATURES: Default number of flattened input features (2304).
        DEFAULT_HIDDEN_FEATURES: Default number of hidden units (5).
        DEFAULT_OUT_FEATURES: Default number of classification outputs (1).
        DEFAULT_DROPOUT_RATE: Default dropout probability (0.5).
        classifier: Sequential container of flattening, linear, dropout, and
            activation layers.
    """

    DEFAULT_IN_FEATURES: int = 2304
    DEFAULT_HIDDEN_FEATURES: int = 5
    DEFAULT_OUT_FEATURES: int = 1
    DEFAULT_DROPOUT_RATE: float = 0.5

    def __init__(
        self,
        in_features: int = DEFAULT_IN_FEATURES,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        out_features: int = DEFAULT_OUT_FEATURES,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        use_sigmoid: bool = True,
    ) -> None:
        """Initializes the classical classification head.

        Args:
            in_features: Number of flattened features from the extractor backbone.
                Defaults to 2304.
            hidden_features: Number of intermediate dense features. Defaults to 5.
            out_features: Number of output classification units. Defaults to 1.
            dropout_rate: Dropout probability during training. Defaults to 0.5.
            use_sigmoid: Whether to append a Sigmoid activation function to the
                final linear layer. Defaults to True.
        """
        super().__init__()

        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass through the dense classifier head.

        Args:
            x: Spatial feature map tensor or flattened embedding tensor of shape
                `(Batch, Channels, Height, Width)` or `(Batch, Features)`.

        Returns:
            Output prediction tensor of shape `(Batch, 1)`.
        """
        return self.classifier(x)

    @staticmethod
    def apply_glorot_init(module: nn.Module) -> None:
        """Applies Glorot (Xavier) Uniform weight initialization.

        Initializes weights of linear submodules with Xavier uniform
        distribution and resets biases to zero. Designed for use with
        PyTorch's `model.apply()`.

        Args:
            module: The PyTorch submodule being visited during recursive
                initialization traversal.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class TanhBottleneckHead(nn.Module):
    """The quantum head's classical layers, with nothing in place of the circuit.

    Flattens the feature maps, projects them to a narrow hidden layer
    (`Linear(2304 -> 6)`), bounds it with `tanh(x) * (pi / 2)` as the quantum
    head scales its circuit angles, and maps the hidden values to one logit
    (`Linear(6 -> 1)`). It has no dropout. Where the quantum head's post-net
    reads the circuit's expectation values, this head's post-net reads the
    scaled hidden values directly.

    Attributes:
        DEFAULT_IN_FEATURES: Default number of flattened input features (2304).
        DEFAULT_HIDDEN_FEATURES: Default hidden width (6, the quantum head's
            default number of qubits).
        DEFAULT_OUT_FEATURES: Default number of classification outputs (1).
        ANGLE_SCALING_FACTOR: Multiplier of the `tanh` output, the quantum
            head's angle scaling.
        flatten: PyTorch Flatten layer.
        pre_net: Linear projection from `in_features` to the hidden width.
        post_net: Linear layer from the hidden width to the outputs.
        sigmoid: Sigmoid applied to the outputs, or `None` for raw logits.
    """

    DEFAULT_IN_FEATURES: int = 2304
    DEFAULT_HIDDEN_FEATURES: int = 6
    DEFAULT_OUT_FEATURES: int = 1
    ANGLE_SCALING_FACTOR: float = math.pi / 2.0

    def __init__(
        self,
        in_features: int = DEFAULT_IN_FEATURES,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        out_features: int = DEFAULT_OUT_FEATURES,
        use_sigmoid: bool = True,
    ) -> None:
        """Initializes the bottleneck head.

        Args:
            in_features: Number of flattened features from the extractor backbone.
                Defaults to 2304.
            hidden_features: Width of the bounded hidden layer. Defaults to 6.
            out_features: Number of output classification units. Defaults to 1.
            use_sigmoid: Whether to apply a Sigmoid to the outputs. Defaults to
                True.
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.pre_net = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.post_net = nn.Linear(
            in_features=hidden_features, out_features=out_features
        )
        self.sigmoid = nn.Sigmoid() if use_sigmoid else None

    def bottleneck(self, x: Tensor) -> Tensor:
        """Computes the scaled hidden values, each in `[-pi/2, pi/2]`.

        Args:
            x: Spatial feature map tensor or flattened embedding tensor of shape
                `(Batch, Channels, Height, Width)` or `(Batch, Features)`.

        Returns:
            Hidden tensor of shape `(Batch, hidden_features)`.
        """
        return tanh(self.pre_net(self.flatten(x))) * self.ANGLE_SCALING_FACTOR

    def forward(self, x: Tensor) -> Tensor:
        """Computes the forward pass through the bottleneck head.

        Args:
            x: Spatial feature map tensor or flattened embedding tensor of shape
                `(Batch, Channels, Height, Width)` or `(Batch, Features)`.

        Returns:
            Output tensor of shape `(Batch, 1)`: logits, or probabilities when
            the head applies a Sigmoid.
        """
        x = self.post_net(self.bottleneck(x))
        return self.sigmoid(x) if self.sigmoid is not None else x

    @staticmethod
    def apply_glorot_init(module: nn.Module) -> None:
        """Applies Glorot (Xavier) Uniform weight initialization.

        The same initialization as the quantum head's pre-net and post-net.

        Args:
            module: The PyTorch submodule being visited during recursive
                initialization traversal.
        """
        ClassicalClassifierHead.apply_glorot_init(module)


CLASSICAL_HEADS: dict[str, type[ClassicalClassifierHead | TanhBottleneckHead]] = {
    DEFAULT_CLASSICAL_HEAD: ClassicalClassifierHead,
    "tanh_bottleneck": TanhBottleneckHead,
}


def _head_class(name: str) -> type[ClassicalClassifierHead | TanhBottleneckHead]:
    """Looks a classical head up by its registry name.

    Args:
        name: Registry name, such as `"paper"` or `"tanh_bottleneck"`.

    Returns:
        The registered head class.

    Raises:
        ValueError: If no head has that name.
    """
    try:
        return CLASSICAL_HEADS[name]
    except KeyError:
        raise ValueError(
            f"Unknown classical head {name!r}; available: {sorted(CLASSICAL_HEADS)}."
        ) from None


def build_classical_head(
    name: str = DEFAULT_CLASSICAL_HEAD,
    in_features: int = ClassicalClassifierHead.DEFAULT_IN_FEATURES,
    hidden_features: int | None = None,
    use_sigmoid: bool = False,
) -> nn.Module:
    """Builds a registered classical head.

    Args:
        name: Registry name of the head. Defaults to the base paper's head.
        in_features: Number of flattened features from the extractor backbone.
        hidden_features: Hidden width, or `None` for the head's own default.
        use_sigmoid: Whether the head applies a Sigmoid to its output. Defaults
            to False (raw logits, for `BCEWithLogitsLoss`).

    Returns:
        The untrained head with PyTorch's default initialization.

    Raises:
        ValueError: If no head has that name.
    """
    head_class = _head_class(name)
    width = hidden_features or head_class.DEFAULT_HIDDEN_FEATURES
    return head_class(
        in_features=in_features, hidden_features=width, use_sigmoid=use_sigmoid
    )


def stored_hidden_features(name: str, hidden_features: int | None) -> int | None:
    """Returns the width a run specification stores for a head.

    A head's own default width is stored as `None`, so asking for it
    explicitly gives the same configuration as not asking.

    Args:
        name: Registry name of the head.
        hidden_features: Requested width, or `None` for the default.

    Returns:
        `None` for the default width, otherwise the requested width.

    Raises:
        ValueError: If no head has that name.
    """
    if hidden_features == _head_class(name).DEFAULT_HIDDEN_FEATURES:
        return None
    return hidden_features


def classical_head_for(spec: RunSpec, use_sigmoid: bool = False) -> nn.Module:
    """Builds the classical head a run specification describes.

    Args:
        spec: Spec of a classical transfer-learning run.
        use_sigmoid: Whether the head applies a Sigmoid to its output. Defaults
            to False (raw logits, as the heads are trained and evaluated).

    Returns:
        The untrained head, whose weights a checkpoint of that run fits.

    Raises:
        ValueError: If the spec is not a classical transfer-learning spec, or
            names an unknown head.
    """
    if spec.paradigm is not Paradigm.CTL:
        raise ValueError(
            f"Only a {Paradigm.CTL.value} spec has a classical head, got "
            f"{spec.paradigm.value}."
        )
    return build_classical_head(
        spec.head or DEFAULT_CLASSICAL_HEAD,
        hidden_features=spec.hidden_features,
        use_sigmoid=use_sigmoid,
    )
