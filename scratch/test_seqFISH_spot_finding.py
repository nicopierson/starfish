#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file experiments with spot finding approaches. In doing so, it attempts to categorize the data
characteristics that could cause a user to leverage a particular spot finder. So far, these
characteristics include:

1. How big are your spots? Are they diffraction-limited, e.g. single-pixel? do they vary in size?
# TODO move this down, but larger spots can use DoG instead of LoG, invariant spots can leverage
# smaller gaussian blurring windows, both of these will tune performance.
# fundamentally, the pixel-spot detector is the limit of "find small spots" when laplacians are
# unnecessary
2. Are your spots perfectly aligned, or do they require alignment?
3. What kind of filtering do you want to run on your spots? (size, brightness, separation, ...)

Globally, this explores what users can do, given prior information about their spot characteristics.

# NOTE if you just execute the whole script, it will open a LOT of napari windows; it is designed
#      to be stepped through.

"""
import os
from typing import Tuple
import napari_gui
import numpy as np
import starfish.data
import starfish.display
from starfish.image import Filter, Registration, Segmentation
from starfish.spots import SpotFinder, PixelSpotDecoder, TargetAssignment
from starfish.types import Axes, Features
from skimage.morphology import ball, disk, opening, white_tophat
from functools import partial
from copy import deepcopy
from starfish.spots import SpotFinder
from starfish.spots import PixelSpotDecoder
from skimage.feature import blob_dog, blob_log
import pandas as pd
from scipy.spatial import cKDTree
from starfish.spots import SpotFinder
from itertools import product
from collections import defaultdict
from starfish import IntensityTable
from starfish.types import SpotAttributes

%gui qt5

###################################################################################################
# Pilot on ISS data
#
# ISS data is relatively sparse and has high signal-to-noise for detected spots. It makes it an
# easy substrate to test the approaches.

# this is slow, load locally
# experiment = starfish.data.SeqFISH(use_test_data=False)

from starfish.imagestack.parser.crop import CropParameters  # noqa
experiment = starfish.Experiment.from_json(os.path.expanduser('~/scratch/seqfish/experiment.json'))
fov = experiment['fov_000']

crop_params = CropParameters(x_slice=slice(0, 1024), y_slice=slice(1024, 2048))
image = fov.get_image('primary', crop_params)


###################################################################################################
# Register the data, if it isn't.

# SeqFISH data does not appear to be fully registered, but there's some complexity around sub-pixel
# registration for more complex registrations that I don't have time to understand yet.

# for now, we can register using translation only

from skimage.feature import register_translation  # noqa
from skimage.transform import warp  # noqa
from skimage.transform import SimilarityTransform  # noqa

def _register_imagestack(target_image, reference_image, upsample_factor=5):
    target_image = np.squeeze(target_image)
    reference_image = np.squeeze(reference_image)
    shift, error, phasediff = register_translation(target_image, reference_image, upsample_factor=1)
    return SimilarityTransform(translation=shift)

projection = image.max_proj(Axes.CH, Axes.ZPLANE)
reference_image = projection.sel({Axes.ROUND: 1}).xarray

register_imagestack = partial(
    _register_imagestack, reference_image=reference_image, upsample_factor=5
)
transforms = projection.transform(register_imagestack, group_by={Axes.ROUND}, n_processes=1)
round_to_tf = {axes[Axes.ROUND]: tf for tf, axes in transforms}

# starfish doesn't have a great ability to apply different functions to different parts of an
# imagestack based on their Axes. It would be great if one could iterate over the ImageStack
# and apply a set of transformations to each tile based on the tile's coordinates.

def _warp_tile(tile, transform):
    round_ = int(tile.coords[Axes.ROUND.value])
    return warp(np.squeeze(tile), transforms[round_])

def warp_imagestack(imagestack, transform_set, chunk_by: Tuple):

    new = starfish.ImageStack.from_numpy_array(np.zeros_like(imagestack.xarray.values))

    selectors = product(*(list(int(i) for i in imagestack.xarray.coords[c]) for c in chunk_by))
    for s in selectors:
        selector = dict(zip(chunk_by, s))
        data = np.squeeze(imagestack.xarray.sel(selector))
        res = warp(data, transform_set[selector[Axes.ROUND.value]])
        new.set_slice(selector, res.astype(np.float32))

    return new


image = warp_imagestack(image, round_to_tf, chunk_by=("r", "c", "z"))


###################################################################################################
# remove background from the data and equalize the channels.

def create_weighted_disk(selem_radius):
    s = ball(selem_radius)
    h = int((s.shape[1] + 1) / 2)  # Take only the upper half of the ball
    s = s[:h, :, :].sum(axis=0)  # Flatten the 3D ball to a weighted 2D disc
    weighted_disk = (255 * (s - s.min())) / (s.max() - s.min())  # Rescale weights into 0-255
    return weighted_disk


# # look at the background constructed from these methods
# selem = create_weighted_disk(7)
# opening = partial(opening, selem=selem)

# background = image.apply(
#     opening,
#     group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False, n_processes=8
# )

# starfish.display.stack(background)
# starfish.display.stack(image)

# subtract the background in-place
tophat = partial(white_tophat, selem=create_weighted_disk(7))

image.apply(
    tophat,
    group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=True, n_processes=16
)

###################################################################################################
# INITIAL SPOT FINDING
from starfish.compat import blob_dog, blob_log  # noqa

# run an initial spot finding to identify (1) the PSF and (2) the correct intensity values
def call_spots(spot_detector_function, threshold_percentile, **kwargs):

    volumes = list(product(
        range(image.shape[Axes.ROUND]),
        range(image.shape[Axes.CH]),
    ))

    spot_results = []
    thresholds = []
    for i, (r, c) in enumerate(volumes):
        substack = np.squeeze(
            image.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values
        )
        threshold = np.percentile(np.ravel(substack), threshold_percentile)
        res = spot_detector_function(
            substack,
            threshold=threshold,
            **kwargs
        )
        spot_results.append(res)
        break

    ##############################################################################################
    # MATCH SPOTS ACROSS ROUNDS

    # in which round do you want to seed the local search
    ANCHOR_ROUND = 0
    # what is the radius of the disk (2d) or sphere (3d) of the space that a spot can be found in?
    SEARCH_RADIUS = 3

    # with radius = 3px, only 4% of spots found more than one spot inside the radius and in the ISS
    # data, those were almost always bonafide spots. For this reason, the logic below simply
    # leverages the closest spot, and I'm confident, on this un-crowded, high SnR data, that this
    # finds optimal spots.

    # TODO delete me
    # round_data = defaultdict(list)
    # for arr, (r, c) in zip(spot_results, volumes):
    #     sub_arr = arr.loc[:, ['z', 'y', 'x', 'size_x']]
    #     sub_arr.columns = ["z", "y", "x", "r"]
    #     sub_arr[Axes.CH.value] = np.full(sub_arr.shape[0], c)
    #     round_data[r].append(sub_arr)

    round_data = defaultdict(list)
    for arr, (r, c) in zip(spot_results, volumes):
        arr = pd.DataFrame(
            data=np.hstack([arr[:, :-3], np.mean(arr[:, -3:], axis=1)[:, None]]),
            columns=['z', 'y', 'x', 'r']
        )
        arr[Axes.CH.value] = np.full(arr.shape[0], c)
        round_data[r].append(arr)

    # this method is nice because we can assess round imbalance!
    # we could go back and increase the number of spots to search for in the other rounds...
    # and now I understand why they're doing this the way they are...
    # possible objective function: num_spots - penalty for spots that don't decode
    round_dataframes = {
        k: pd.concat(v, axis=0).reset_index().drop('index', axis=1) for k, v in round_data.items()
    }

    # this will need to be generalized to volumetric data by adding Axes.Z.value, note that Z should
    # potentially have a great penalty, as pixels are potentially not isomorphic.
    traces = []
    template = cKDTree(
        round_dataframes[ANCHOR_ROUND][[Axes.ZPLANE.value, Axes.Y.value, Axes.X.value]])
    for r in sorted(set(round_dataframes.keys()) - {ANCHOR_ROUND, }):
        query = cKDTree(round_dataframes[r][[Axes.ZPLANE.value, Axes.Y.value, Axes.X.value]])
        traces.append(template.query_ball_tree(query, SEARCH_RADIUS, p=2))

    ##############################################################################################
    # CREATE INTENSITY TABLE


    # TODO build a hamming tree from the codebook (is there a way to generalize?)
    # use this to refine the local search space in cases of crowded data. here it's not necessary,
    # and so this isn't built yet.

    # set some constants here
    template_data = round_dataframes[ANCHOR_ROUND]
    query_spot_indices = zip(*traces)
    N_CH=4  # noqa
    N_ROUND=6  # noqa

    # build SpotAttributes from anchor round
    attrs = round_dataframes[ANCHOR_ROUND].drop(Axes.CH, axis=1)
    attrs.columns = ['z', 'y', 'x', 'radius']
    attrs['radius'] = np.full(attrs.shape[0], fill_value=4)
    spot_attributes = SpotAttributes(attrs)


    # build an IntensityTable
    # this is a slow, loop based way to measure the intensities of each of the spots that are found
    # across the blob_dog results for each image. This should be vectorized, but is adequate to
    # prototype a spot detector that functions across rounds. This code is only necessary because
    # the SpotFinder does not return the maximum intensity of the spot by default, so we need to
    # go measure it from the returned spot locations.

    # these arrays will hold indices into the eventual IntensityTable
    feature_indices = []
    round_indices = []
    channel_indices = []

    # this array will contain values to be set in the IntensityTable.
    values = []

    def add_spot_information_to_indexers(
        round_, spot_results, imagestack, spot_index, curr_feature
    ):
        # this function finds the maximum intensities of a spot in a given location
        # this function has side-effects on the above arrays
        selector = {
            Axes.ROUND: round_,
            Axes.CH: spot_results.loc[spot_index, Axes.CH],
            Axes.X: int(spot_results.loc[spot_index, Axes.X]),
            Axes.Y: int(spot_results.loc[spot_index, Axes.Y]),
            Axes.ZPLANE: int(spot_results.loc[spot_index, Axes.ZPLANE]),
        }

        values.append(imagestack.xarray.sel(selector))
        round_indices.append(selector[Axes.ROUND])
        channel_indices.append(selector[Axes.CH])
        feature_indices.append(curr_feature)

    # we need to keep track of which feature we're on, since each one can have up to four spots.
    feature_index = 0

    # starting with the anchor channel, measure the maximum intensity of each spot in each pixel
    # trace
    from itertools import zip_longest
    for anchor_index, indices in zip_longest(template_data.index, query_spot_indices):

        # add anchor spot information
        add_spot_information_to_indexers(
            ANCHOR_ROUND,
            template_data,
            image,
            anchor_index,
            feature_index,
        )

        # iterate through each of the other rounds
        try:
            for round_, spot_indices in enumerate(indices, 1):

                # we allow for more than one spot to occur within the search radius. For now, take
                # only the closest spot. Later, we can evaluate if filtering based on hamming codes
                # is needed here
                for spot_index in spot_indices:
                    add_spot_information_to_indexers(
                        round_,
                        round_dataframes[round_],
                        image,
                        spot_index,
                        feature_index,
                    )
                    break  # skip spots after the first one
        except:
            pass

        # the next iteration will examine the following spot in the anchor round, so we increment
        # the feature index
        feature_index += 1

    # fill the intensity table
    intensities = IntensityTable.empty_intensity_table(spot_attributes, n_ch=N_CH, n_round=N_ROUND)
    intensities.values = np.full_like(intensities.values, np.nan)
    intensities.values[
        feature_indices,
        np.array(channel_indices, dtype=int),
        np.array(round_indices, dtype=int)
    ] = values

    return intensities

spot_kwargs = dict(
    min_sigma=np.array([0.5, 1, 1]),
    max_sigma=np.array([2, 4, 4]),
    # sigma_ratio=1.2,
    num_sigma=10,
    exclude_border=2
)

intensities = call_spots(blob_log, threshold_percentile=96, **spot_kwargs)

starfish.display.stack(image, intensities)

###################################################################################################
# normalize the channels so that they have approximately equal intensity

def scale_by_percentile(array: np.ndarray, p: int) -> np.ndarray:
    maxval = np.percentile(array, p)
    scaled = array / maxval

    # truncate any above-value intensities
    scaled[scaled > 1] = 1
    return scaled

scale_by_99 = partial(scale_by_percentile, p=99.7)

image.apply(
    scale_by_99,
    group_by={Axes.CH},
    in_place=True,
)

starfish.display.stack(image)

###################################################################################################
# DECODE THE IMAGES

decoded_dog_intensities = experiment.codebook.decode_per_round_max(dog_intensities)

# what fraction of spots decode?
fail_decoding = np.sum(decoded_dog_intensities['target'] == 'nan')
obs = (decoded_dog_intensities.shape[0] - fail_decoding) / decoded_dog_intensities.shape[0]

# what fraction of spots would match by chance?
# each spot has 4 rounds each with 4 options. (no spot is not an option here by design)
exp = 31 / (4 ** 4)

# odds ratio ~ 7.27
obs / exp

###################################################################################################
# SEGMENT THE CELLS

seg = Segmentation.Watershed(
    nuclei_threshold=.16,
    input_threshold=.22,
    min_distance=57,
)
label_image = seg.run(image, experiment['fov_001']['nuclei'])

# assign spots to cells; this doesn't really work.
ta = TargetAssignment.Label()
assigned = ta.run(label_image, decoded_dog_intensities)

# this sucks. Let's try Ilastik. Dump a max projection of the nuclei image and dots images.
# import skimage.io
# to_ilastik = np.squeeze(np.maximum(
#     experiment['fov_001']['nuclei'].xarray.values,
#     experiment['fov_001']['dots'].xarray.values
# ))
# skimage.io.imsave(os.path.expanduser('~/Downloads/nuclei.tiff'), to_ilastik)
# # lame, turns out this is just hard to segment :(

###################################################################################################
# Create a count matrix based on this bad segmentation.
counts = assigned.to_expression_matrix()
