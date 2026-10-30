"""Input validation and aggregation use the shared training/serving implementation."""

from forecast.contracts import InputError
from forecast.data import load_inputs

__all__ = ["InputError", "load_inputs"]
