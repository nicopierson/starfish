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
%load_ext autoreload
%autoreload 2
%gui qt

import os
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
import warnings
from scipy.spatial import cKDTree
from starfish.spots import SpotFinder
from itertools import product
from collections import defaultdict
from starfish import IntensityTable
from starfish.types import SpotAttributes

###################################################################################################
# LOAD DATA

experiment = starfish.Experiment.from_json(
    "https://d2nhj9g34unfro.cloudfront.net/browse/formatted/20190201/starmap_test/experiment.json"
)
field_of_view = experiment['fov_000']
primary_image = field_of_view['primary']

# look at the data
starfish.display.stack(primary_image)

###################################################################################################
# Register the data, if it isn't.
# Data is already registered

###################################################################################################
# REMOVE BACKGROUND
# no background removal or filtering necessary of the cleared tissue, according to authors
# but we can walk through what this could look like.

###################################################################################################
# REMOVE Z-SLICES THAT ARE NOT IN FOCUS
# this is important to ensure that scaling doesn't fabricate signal in out-of-focus planes. For this
# experiment, which is z-projected, this step can be omitted.

###################################################################################################
# NORMALIZE INTENSITY
# Apply histogram matching to each z-volume. Assumes that all channels and rounds should have
# approximately similar fluorescence distributions

from skimage.transform import match_histograms  # noqa

# note that we should also match over Z, but that this data is projected
def match_channel_round(array):
    reference=primary_image.xarray.sel({Axes.CH.value: 0, Axes.ROUND.value: 1}).values
    return match_histograms(array, reference=reference)

histogram_normalized = primary_image.apply(match_channel_round, group_by={Axes.ROUND, Axes.CH})

starfish.display.stack(histogram_normalized)

###################################################################################################
# CALL SPOTS IN EACH TILE

# set-up:
volumes = list(product(
    range(histogram_normalized.shape[Axes.ROUND]),
    range(histogram_normalized.shape[Axes.CH]),
))

THRESHOLD = np.percentile(np.ravel(histogram_normalized.xarray.values), 94)
spot_results = []
for i, (r, c) in enumerate(volumes):
    substack = np.squeeze(
        histogram_normalized.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values
    )
    res = blob_dog(
        substack,
        min_sigma=np.array([1, 2, 2]),
        max_sigma=np.array([2, 8, 8]),
        sigma_ratio=1.2,
        threshold=THRESHOLD,
        exclude_border=2
    )
    spot_results.append(res)
    print(f"done calling for volume {i}")

###################################################################################################
# MATCH SPOTS ACROSS ROUNDS

# in which round do you want to seed the local search
ANCHOR_ROUND = 0
# what is the radius of the disk (2d) or sphere (3d) of the space that a spot can be found in?
SEARCH_RADIUS = 7

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

# Match across rounds
traces = []
template = cKDTree(round_dataframes[ANCHOR_ROUND][[Axes.ZPLANE.value, Axes.Y.value, Axes.X.value]])
for r in sorted(set(round_dataframes.keys()) - {ANCHOR_ROUND,}):
    query = cKDTree(round_dataframes[r][[Axes.ZPLANE.value, Axes.Y.value, Axes.X.value]])
    traces.append(template.query_ball_tree(query, SEARCH_RADIUS, p=2))

###################################################################################################
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

def add_spot_information_to_indexers(round_, spot_results, imagestack, spot_index, curr_feature):
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
for anchor_index, indices in zip(template_data.index, query_spot_indices):

    # add anchor spot information
    add_spot_information_to_indexers(
        ANCHOR_ROUND,
        template_data,
        histogram_normalized,
        anchor_index,
        feature_index,
    )

    # iterate through each of the other rounds
    for round_, spot_indices in enumerate(indices, 1):

        # we allow for more than one spot to occur within the search radius. For now, take only
        # the closest spot. Later, we can evaluate if filtering based on hamming codes is needed
        # here
        for spot_index in spot_indices:
            add_spot_information_to_indexers(
                round_,
                round_dataframes[round_],
                histogram_normalized,
                spot_index,
                feature_index,
            )
            break  # skip spots after the first one

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

###################################################################################################
# INTERPRET RESULTS

# evaluate our handywork by projecting across channel (as we've done to fill these arrays).
# the logic should be that in a max-projection across channels, each round subsequent to the
# anchor round should _also_ have spots that are detected.

starfish.display.stack(histogram_normalized, intensities)

# with a fixed threshold...
# what we see instead is that the quantile normalization has failed to fully equalize the rounds,
# and that the anchor round has very bright spots relative, in particular, to rounds 2 and 3. As a
# result, we're not successfully calling spots in those rounds, which will cause decoding to fail.

# this demonstrates that the amount of signal picked up by the peaks in the different rounds is
# very uneven. How to fix?, specifically, the 2nd and 3rd channel are particularly sparse. In the
# fourth round, the 2nd and 4th channel are sparse.
# this might be a simple metric to diagnose an experiment or spot finding approach
intensities.sum(Features.AXIS)

###################################################################################################
# DECODE THE IMAGES

# filter out intensities lacking a spot in each round.
valid = (~intensities.isnull()).sum(("c", "r")) >= 6
valid_intensities = intensities[valid]

# fix the codebook
codebook = experiment.codebook.shift(c=-1)[:, :4, :]
decoded = codebook.decode_per_round_max(valid_intensities)

# check for common problems
auto_channel = Codebook.constitutive_channel_fluorescence_mask(n_channel=4, n_round=6)
auto_panchannel = Codebook.constitutive_fluorescence_mask(n_channel=4, n_round=6)

decoded_auto_ch = auto_channel.metric_decode(
    valid_intensities.fillna(0),
    max_distance=0.52, min_intensity=THRESHOLD, norm_order=2
)

# check how many pan-channel autofluorescent spots there are:
np.histogram(decoded_auto_ch.distance, bins=10)
np.sum(decoded_auto_ch.distance < 0.52)

# check how many all-channel autofluorescent spots there are:
decoded_pan_ch = auto_panchannel.metric_decode(
    valid_intensities.fillna(0),
    max_distance=0.52, min_intensity=THRESHOLD, norm_order=2
)

# get the successfully decoded spots only
pass_decoding = decoded[decoded.target != 'nan']

# get the spots that are detected but do not decode:
fail_decoding = decoded[decoded.target == "nan"]
fail_decoding["radius"] = fail_decoding.radius["radius"] * 1.2  # increase mask size

# look at decoded spots. Mask failing spots.
masks = {"fail_decoding": ("y", fail_decoding)}
starfish.display.stack(histogram_normalized, spots=pass_decoding, extra_spots=masks)

###################################################################################################
# WHAT DOESN'T DECODE?

# what fraction of spots decode?
obs = pass_decoding.shape[0] / decoded.shape[0]

# what fraction of spots would match by chance?
# each spot has 4 rounds each with 4 options. (no spot is not an option here by design)
exp = codebook.shape[0] / (4 ** 6)

# odds ratio ~ 7.27
obs / exp

###################################################################################################
# SEGMENT THE CELLS

seg = Segmentation.Watershed(
    nuclei_threshold=.16,
    input_threshold=.22,
    min_distance=57,
)
label_image = seg.run(image, experiment['fov_000']['nuclei'])

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

###################################################################################################