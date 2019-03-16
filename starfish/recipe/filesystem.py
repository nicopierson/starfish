from typing import Any, Type

from starfish.codebook.codebook import Codebook
from starfish.expression_matrix.expression_matrix import ExpressionMatrix
from starfish.imagestack.imagestack import ImageStack
from starfish.intensity_table.intensity_table import IntensityTable


def load(file_path_or_uri: str, object_type: Type) -> Any:
    if object_type == ImageStack:
        return ImageStack.from_path_or_url(file_path_or_uri)
    elif object_type == IntensityTable:
        return IntensityTable.load(file_path_or_uri)
    elif object_type == ExpressionMatrix:
        return ExpressionMatrix.load(file_path_or_uri)
    elif object_type == Codebook:
        return Codebook.from_json(file_path_or_uri)

    raise NotImplementedError(f"No loader implemented for file type {object_type}")

def save(file_path: str, object: Any) -> None:
    