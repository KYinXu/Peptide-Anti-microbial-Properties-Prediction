"""Variant generation for length-extension analysis."""

from .common import ParentRecord, SubsetRecord, VariantRecord
from .generation import generate_variants

__all__ = [
    "ParentRecord",
    "SubsetRecord",
    "VariantRecord",
    "generate_variants",
]
