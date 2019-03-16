from typing import Dict, Sequence

from .runcommand import Runnable


class Recipe:
    def __init__(
            self,
            recipe_str: str,
            input_paths_or_urls: Sequence[str],
            output_paths: Sequence[str],
    ):
        self.outputs: Dict[int, Runnable] = {}
        self.output_paths = output_paths
        vars = {
            "file_inputs": input_paths_or_urls,
            "compute": Runnable,
            "file_outputs": self.outputs,
        }
        ast = compile(recipe_str, "<string>", "exec")
        exec(ast, vars)

        assert len(self.outputs) == len(self.output_paths), \
            "Recipe generates more outputs than output paths provided!"

        # verify that the outputs are sequential.
        for ix in range(len(self.outputs)):
            assert ix in self.outputs, \
                f"file_outputs[{ix}] is not set"
            assert isinstance(self.outputs[ix], Runnable), \
                f"file_outputs[{ix}] is not the result of a compute(..)"

    def run(self):
        # find each of the outputs and run them.
        for ix in range(len(self.outputs)):
            runnable: Runnable = self.outputs[ix]
            result = runnable.run()
