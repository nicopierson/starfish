from functools import partial
from typing import Optional

import numpy as np
import xarray as xr

from starfish.codebook.codebook import Codebook
from starfish.imagestack.imagestack import ImageStack
from starfish.util import click
from ._base import FilterAlgorithmBase
from .util import determine_axes_to_group_by


class Decode(FilterAlgorithmBase):

    def __init__(self, metric: Optional[str] = "euclidean", min_only=True) -> None:
        """Decode an image in round/channel space to target space

        The resulting ImageStack will be of shape (1, n_targets, z, y, x) where n_targets are the
        number of targets in the codebook. The values of the decoded images are the distance of
        each pixel from the given target, calculated using the provided distance.


        Parameters
        ----------
        min_only : bool
            If True, return a (1, 1, z, y, x) label image whose values represent the integer target
            that has the smallest distance for each pixel in (z, y, x). In this case, a dictionary
            is also returned that contains a mapping between targets and their integer IDs, and a
            second (1, 1, z, y, x) volume whose values contain the distances. The distance image
            can be interpreted as a probability space reflecting the closeness of each pixel to any
            code in the codebook.
        metric : str
            the metric used to calculate pixel intensity distance from codes in codebook

        """

    _DEFAULT_TESTING_PARAMETERS = {}

    @staticmethod
    def _decode(
        image: xr.DataArray,
        codebook: Codebook,
        metric: Optional[str] = "euclidean"
    ) -> np.ndarray:
        """
        """
        traces = image.stack(traces=(Axes.ROUND.value, Axes.CH.value))
        linearized_traces = traces.stack(pixels=(Axes.Z.value, Axes.Y.value, Axes.X.value))

    def run(self) -> ImageStack:
        """Perform filtering of an image stack

        Parameters
        ----------
        stack : ImageStack
            Stack to be filtered.
        in_place : bool
            if True, process ImageStack in-place, otherwise return a new stack
        verbose : bool
            If True, report on the percentage completed (default = False) during processing
        n_processes : Optional[int]
            Number of parallel processes to devote to calculating the filter

        Returns
        -------
        ImageStack :
            If in-place is False, return the results of filter as a new stack.  Otherwise return the
            original stack.

        """
        group_by = determine_axes_to_group_by(self.is_volume)
        clip = partial(self._scale, p=self.p)
        result = stack.apply(
            clip,
            group_by=group_by, verbose=verbose, in_place=in_place, n_processes=n_processes
        )
        return result

    @staticmethod
    @click.command("Decode")
    @click.option(
        "--metric", default="euclidean", type=str, help="distance metric")
    @click.option(
        "--max-only", is_flag=True, help="if True, return only closest target for each pixel")
    @click.pass_context
    def _cli(ctx, metric, max_only):
        ctx.obj["component"]._cli_run(ctx, ScaleByPercentile(p, is_volume))
