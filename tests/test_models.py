"""Tests for model implementations."""

from __future__ import annotations

from myproject.models.baseline import BaselineModel
from myproject.models.transformer import TransformerModel


def test_baseline_model_init() -> None:
    """Test BaselineModel initialization with default parameters."""
    model = BaselineModel()
    assert model.hidden_size == 64
    assert model.num_layers == 2


def test_baseline_model_custom_params() -> None:
    """Test BaselineModel initialization with custom parameters."""
    model = BaselineModel(hidden_size=128, num_layers=4)
    assert model.hidden_size == 128
    assert model.num_layers == 4


def test_baseline_model_repr() -> None:
    """Test BaselineModel string representation."""
    model = BaselineModel(hidden_size=32, num_layers=1)
    repr_str = repr(model)
    assert "BaselineModel" in repr_str
    assert "32" in repr_str
    assert "1" in repr_str


def test_baseline_model_forward() -> None:
    """Test BaselineModel forward pass returns input unchanged."""
    model = BaselineModel()
    x = [1, 2, 3]
    result = model.forward(x)
    assert result == x


def test_transformer_model_init() -> None:
    """Test TransformerModel defaults match configs/model/transformer.yaml."""
    model = TransformerModel()
    assert (model.d_model, model.nhead, model.num_layers) == (256, 8, 6)


def test_transformer_model_repr_and_forward() -> None:
    """Test TransformerModel representation and placeholder forward pass."""
    model = TransformerModel(d_model=32, nhead=4, num_layers=1)
    assert repr(model) == "TransformerModel(d_model=32, nhead=4, num_layers=1)"
    assert model.forward([1, 2, 3]) == [1, 2, 3]
