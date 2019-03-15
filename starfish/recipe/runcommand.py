import inspect
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from starfish.pipeline.algorithmbase import AlgorithmBase
from starfish.pipeline.pipelinecomponent import PipelineComponent, PipelineComponentType
from .filesystem import load


class Runnable:
    def __init__(
            self,
            pipeline_component_name: str,
            algorithm_name: str,
            *inputs,
            **algorithm_options,
    ) -> None:
        self._pipeline_component_name = pipeline_component_name
        self._algorithm_name = algorithm_name
        self._inputs: Sequence[Union[Runnable, Tuple[str, Type]]] = inputs
        self._algorithm_options = algorithm_options

        self._pipeline_component_cls: Optional[Type[PipelineComponent]] = None
        self._algorithm_cls: Optional[Type[AlgorithmBase]] = None
        self._algorithm_instance: Optional[AlgorithmBase] = None

        self._pipeline_component_cls = PipelineComponentType.get_pipeline_component_type_by_name(
            self._pipeline_component_name)
        self._algorithm_cls = getattr(self._pipeline_component_cls, self._algorithm_name)
        self._algorithm_instance = self._algorithm_cls(**self._algorithm_options)

        # retrieve the actual run method
        signature = Runnable._get_actual_run_method_signature(self._algorithm_instance.run)
        keys = list(signature.parameters.keys())

        assert keys[0] == "self"
        keys = keys[1:]  # ignore the "self" parameter
        assert len(self._inputs) <= len(keys)

        reformatted_inputs: List[Union[Runnable, Tuple[str, Type]]] = []
        for _input, key in zip(self._inputs, keys):
            if isinstance(_input, Runnable):
                reformatted_inputs.append(_input)
            elif isinstance(_input, str):
                # it's a path.
                annotation = signature.parameters[key].annotation
                reformatted_inputs.append((_input, annotation))
        self._inputs = reformatted_inputs

    @staticmethod
    def _get_actual_run_method_signature(run_method: Callable) -> inspect.Signature:
        if hasattr(run_method, "__closure__"):
            # it's a closure, probably because of AlgorithmBaseType.run_with_logging.  Unwrap to
            # find the underlying method.
            closure = run_method.__closure__  # type: ignore
            run_method = closure[0].cell_contents

        return inspect.signature(run_method)

    def get_file_input_types(self) -> Mapping[str, Any]:
        """Retrieve a list of the parameter types for the run method."""
        result: MutableMapping[str, Any] = dict()
        for _input in self._inputs:
            if isinstance(_input, tuple):
                result[_input[0]] = _input[1]

        return result

    def run(self):
        """Do the heavy computation involved in this runnable."""
        inputs = list()
        for _input in self._inputs:
            if isinstance(_input, Runnable):
                inputs.append(_input.run())
            else:
                inputs.append(load(_input[0], _input[1]))
        return self._algorithm_instance.run(*inputs)
