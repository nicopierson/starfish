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
import napari_gui
import numpy as np
import xarray as xr
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

experiment = starfish.data.ISS()
image = experiment['fov_001']['primary']

###################################################################################################
# Register the data, if it isn't.

registration = Registration.FourierShiftRegistration(
    upsampling=1000,
    reference_stack=experiment['fov_001']['dots']
)
registered = registration.run(image, in_place=False)

# registration produces some bizarre edge effects where the edges get wrapped to maintain the
# image's. original shape This is bad because it interacts with the scaling later, so we need to
# crop these out. This is accomplished by cropping by the size of the shift
cropped = registered.sel({Axes.X: (25, -25), Axes.Y: (10, -10)})
starfish.display.stack(cropped)

###################################################################################################
# remove background from the data and equalize the channels. We're going to test using a weighted
# structuring element in addition to the one built into starfish

selem_radii = (7, 10, 13)

def create_weighted_disk(selem_radius):
    s = ball(selem_radius)
    h = int((s.shape[1] + 1) / 2)  # Take only the upper half of the ball
    s = s[:h, :, :].sum(axis=0)  # Flatten the 3D ball to a weighted 2D disc
    weighted_disk = (255 * (s - s.min())) / (s.max() - s.min())  # Rescale weights into 0-255
    return weighted_disk

selems = [create_weighted_disk(r) for r in selem_radii]

# look at the background constructed from these methods

backgrounds = []
for s in selems:
    opening = partial(opening, selem=s)

    backgrounds.append(cropped.apply(
        opening,
        group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False, n_processes=8
    ))

viewers = [starfish.display.stack(i) for i in backgrounds]
starfish.display.stack(image)

# it looks like the smallest radius (7) is doing the best job, use that.
# weighted vs unweighted didn't make much of a difference.

tophat = partial(white_tophat, selem=create_weighted_disk(7))

background_subtracted = cropped.apply(
    tophat,
    group_by={Axes.ROUND, Axes.CH, Axes.ZPLANE}, verbose=False, in_place=False, n_processes=8
)

starfish.display.stack(background_subtracted)
starfish.display.stack(image)

###################################################################################################
# remove slices in z that are not in focus
# this is important to ensure that scaling doesn't fabricate signal in out-of-focus planes. For this
# experiment, which is z-projected, this step can be omitted.

###################################################################################################
# normalize the channels so that they have approximately equal intensity

# A popular approach is quantile normalization, to equalize two or more images.
# It's also possible to scale based on a near-maximal value that excludes outliers.


def quantile_normalize(xarray):
    stacked = xarray.stack(pixels=(Axes.X.value, Axes.Y.value, Axes.ZPLANE.value))
    inds = stacked.groupby(Axes.CH.value).apply(np.argsort)
    pos = inds.groupby(Axes.CH.value).apply(np.argsort)

    sorted_pixels = deepcopy(stacked)
    for v in sorted_pixels.coords[Axes.CH.value]:
        sorted_pixels[v, :] = sorted_pixels[v, inds[v].values].values

    rank = sorted_pixels.mean(Axes.CH.value)

    output = deepcopy(stacked)
    for v in output.coords[Axes.CH.value]:
        output[v] = rank[pos[v].values].values

    return output.unstack("pixels")


quantile_normalized = background_subtracted.xarray.groupby(
    Axes.ROUND.value
).apply(quantile_normalize)
quantile_normalized = starfish.ImageStack.from_numpy_array(quantile_normalized.values)

starfish.display.stack(quantile_normalized)

# TODO this function replicates the above, using skimage functions.
# TODO this is coming in skimage 0.15.0, but isn't on pypi yet.

# from skimage.transform import match_histograms  # noqa

# # note that we should also match over Z, but that this data is projected
# match_r1c1 = partial(
#     match_histograms,
#     reference=background_subtracted.xarray.sel({Axes.CH.value: 0, Axes.ROUND.value: 1})
# )
#
# histogram_normalized = background_subtracted.apply(match_r1c1, group_by={Axes.ROUND, Axes.CH})

###################################################################################################
# TEST DECODING FILTER

traces = quantile_normalized.xarray.stack(
    features=(Axes.ZPLANE.value, Axes.Y.value, Axes.X.value)
)
traces = traces.transpose(Features.AXIS, Axes.CH.value, Axes.ROUND.value)
CODEBOOK = experiment.codebook
res = CODEBOOK.metric_decode(traces, max_distance=1, min_intensity=0, norm_order=2)

# could use this data to produce a label image in the "max_probability" case -- but it's not ideal,
# would rather have the whole space

# break down the decoder
NORM_ORDER = 2
norm_intensities, norms = CODEBOOK._normalize_features(traces, norm_order=NORM_ORDER)
norm_codes, _ = CODEBOOK._normalize_features(CODEBOOK, norm_order=NORM_ORDER)

# use scipy cdist to generate the resulting distance matrix -- this will take a while!
from scipy.spatial.distance import cdist  # noqa
intensity_traces = norm_intensities.stack(traces=(Axes.CH.value, Axes.ROUND.value))
intensity_codes = norm_codes.stack(traces=(Axes.CH.value, Axes.ROUND.value))
distances = cdist(intensity_traces, intensity_codes)

# reshape into an ImageStack
distances = xr.DataArray(distances, coords=(norm_intensities.features, norm_codes.target))
distances = distances.unstack(Features.AXIS)
distances = distances.rename(target=Axes.CH.value)
distance_image = distances.expand_dims(Axes.ROUND.value)

# normalize into (0, 1) -- probably _don't_ want this, but we require it for ImageStack :( :(
normalized_image = distance_image / distance_image.max()

imagestack = starfish.ImageStack.from_numpy_array(normalized_image.values)


metric_outputs, targets = CODEBOOK._approximate_nearest_code(
    norm_codes, norm_intensities, metric=metric)


###################################################################################################
# CALL SPOTS

# this is the DEFAULT spot finding. Below this

# start with 1/1000 pixels as spots and move up from there as necessary to avoid the situation
# where too many peaks are called and the computation time blows up. eventually settled on 97%,
# which still misses _some_ spots.

# my intuition here is that for coded assays, we can relax stringency and rely upon decoding to
# eliminate spurious spots. There are ways that badly formed codebooks could allow this approach
# to introduce bias which should be carefully evaluated. (e.g. if the channels are not balanced in
# their representation in the codebook.)

# TODO this threshold might not correspond to local maxima directly -- it's related to the
# "scale space" -- what does this mean?
threshold = np.percentile(np.ravel(quantile_normalized.xarray.values), 97)

# start with the BlobDetector; difference of hessians only works in 2d, so we're skipping that.
# TODO think about removing DoH from the BlobDetector
min_sigma = 1  # TODO this appears to pick up _some_ 1-px signal
max_sigma = 4  # TODO can we translate these into pixels to be more intuitive for users?
num_sigma = 9  # how many is enough? #TODO dial it down until things break (fewer = faster)
               # TODO num_sigma is _not used_ by blob_dog or blob_doh, make this clear // remove

bd_dog = SpotFinder.BlobDetector(
    min_sigma, max_sigma, num_sigma, threshold=threshold, detector_method='blob_dog',
    is_volume=False
)
dog_intensities = bd_dog.run(quantile_normalized, reference_image_from_max_projection=True)
starfish.display.stack(quantile_normalized, dog_intensities, mask_intensities=threshold - 1e-5)

# LocalMaxPeakFinder is the same thing as blob_log if run after an inverted LoG filter...
# implies we could find the threshold in the same way. We need to refine our spot finding in the

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

###################################################################################################
# APPENDIX, WORK IN PROGRESS AFTER THIS POINT.
# this appendix explores calling spots in each tile and then linking them back together with a local
# search. it is inspired by the spot finder applied by the SeqFISH pipeline, developed by Sheel Shah
# in Long Cai's group.
# the process is broken up into three parts:
# 1. call spots in each tile (broken up into three different approaches.)
# 2. link spots that correspond
# 3. create an IntensityTable
# 4. analyze and ponder the results

###################################################################################################
# CALL SPOTS IN EACH TILE

# note that there are three options separated by "# %%", only one should be run at a time as they
# will overwrite

# set-up:
tiles = list(product(
    range(quantile_normalized.shape[Axes.ROUND]),
    range(quantile_normalized.shape[Axes.CH])
))

# %%

# OPTION 1: apply a threshold calculated as a percentile from the whole imagestack
# TODO: transform's tooltip is wrong, it lists the old "apply over" wording
# NOTE: we could use transform here, but then later could would not be consistent with use of
#       `tiles` in the subsequent options
# spot_finder = partial(
#     blob_dog, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold
# )
# spot_results = quantile_normalized.transform(
#     spot_finder, group_by={Axes.CH, Axes.ROUND, Axes.ZPLANE}
# )

threshold = np.percentile(np.ravel(quantile_normalized.xarray.values), 95)
spot_results = []
for i, (r, c) in enumerate(tiles):
    substack = np.squeeze(quantile_normalized.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values)
    spot_results.append(
        blob_dog(substack, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    )

# %%

# OPTION 2: use the local max peak finder to find the threshold for each round
# TODO right now this performs worse than the one I found by eye by picking the 97th percentile
# across the ravel'ed imagestack

thresholds = []
for r, c in tiles:
    substack = quantile_normalized.sel({Axes.CH: c, Axes.ROUND: r})
    lmpf = SpotFinder.LocalMaxPeakFinder(
        min_distance=2, stringency=0, min_obj_area=3, max_obj_area=np.inf,
        min_num_spots_detected=50, is_volume=False, verbose=False)
    # TODO this method's contract is wrong, it only takes ndarray, not xarray.
    # TODO this method is not that fast -- ~ 5s per image.
    thresholds.append(lmpf._compute_threshold(substack.xarray.values))

# apply the thresholds to sequential tiles
spot_results = []
for i, (r, c) in enumerate(tiles):
    substack = np.squeeze(quantile_normalized.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values)
    spot_results.append(
        blob_log(substack, min_sigma=min_sigma, max_sigma=max_sigma, threshold=thresholds[i])
    )

# %%

# OPTION 3: apply using a variable threshold calculated on each image

PERCENTILE = 97

thresholds = []
for r, c in tiles:
    substack = quantile_normalized.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values
    linear_substack = np.ravel(substack)
    thresholds.append(np.percentile(linear_substack, 97))

# apply the thresholds to sequential tiles
spot_results = []
for i, (r, c) in enumerate(tiles):
    substack = np.squeeze(quantile_normalized.sel({Axes.CH: c, Axes.ROUND: r}).xarray.values)
    spot_results.append(
        blob_dog(substack, min_sigma=min_sigma, max_sigma=max_sigma, threshold=thresholds[i])
    )

###################################################################################################
# MATCH SPOTS ACROSS ROUNDS

# in which round do you want to seed the local search
ANCHOR_ROUND = 0
# what is the radius of the disk (2d) or sphere (3d) of the space that a spot can be found in?
SEARCH_RADIUS = 3

# with radius = 3px, only 4% of spots found more than one spot inside the radius and in the ISS
# data, those were almost always bonafide spots. For this reason, the logic below simply leverages
# the closest spot, and I'm confident, on this un-crowded, high SnR data, that this finds optimal
# spots.

round_data = defaultdict(list)
for arr, (r, c) in zip(spot_results, tiles):
    arr = pd.DataFrame(arr, columns=['y', 'x', 'r'])
    arr[Axes.CH.value] = np.full(arr.shape[0], c)
    round_data[r].append(arr)

# this method is nice because we can assess round imbalance!
# we could go back and increase the number of spots to search for in the other rounds...
# and now I understand why they're doing this the way they are...
# possible objective function: num_spots - penalty for spots that don't decode
round_dataframes = {
    k: pd.concat(v, axis=0).reset_index().drop('index', axis=1) for k, v in round_data.items()
}

# this will need to be generalized to volumetric data by adding Axes.Z.value
traces = []
template = cKDTree(round_dataframes[ANCHOR_ROUND][[Axes.X.value, Axes.Y.value]])
for r in sorted(set(round_dataframes.keys()) - {ANCHOR_ROUND, }):
    query = cKDTree(round_dataframes[r][[Axes.X.value, Axes.Y.value]])
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
N_ROUND=4  # noqa

# build SpotAttributes from anchor round
attrs = round_dataframes[ANCHOR_ROUND].drop(Axes.CH, axis=1)
attrs.columns = ['y', 'x', 'radius']
attrs['z'] = np.zeros(attrs.shape[0], dtype=int)
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
        Axes.ZPLANE: 0  # TODO generalize
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
        quantile_normalized,
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
                quantile_normalized,
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

starfish.display.stack(quantile_normalized, intensities, project_axes={Axes.CH})

# with a fixed threshold...
# what we see instead is that the quantile normalization has failed to fully equalize the rounds,
# and that the anchor round has very bright spots relative, in particular, to rounds 2 and 3. As a
# result, we're not successfully calling spots in those rounds, which will cause decoding to fail.

# this demonstrates that the amount of signal picked up by the peaks in the different rounds is
# very uneven. How to fix?, specifically, the 2nd and 3rd channel are particularly sparse. In the
# fourth round, the 2nd and 4th channel are sparse.
intensities.sum(Features.AXIS)
# this might be a simple metric to diagnose an experiment or spot finding approach

starfish.display.stack(quantile_normalized, intensities)

# some aspects of this are _biologically_ and _experimentally_ weird, as (r, c)=(0, 0) is
# almost completely tumor-specific...

# with the adaptive threshold, the caller is too picky about "what's a spot"

# I'm going to try an adaptive percentile next, to say the top x% of pixels by intensity value
# likely comprise the peaks. If that fails, I'll go play more with the LoG and DoG preprocessing
# to try to understand better how those work and whether there is room for modification.

# with the adaptive percentile, the problem is even more pronounced; this suggests to me that it's
# really a problem further up the pipeline. It's also worth noting that there are more significant
# issues with the adaptive threshold than with the threshold that was calculated across the whole
# image stack -- this is counter to my intuition, and suggests that while the histograms may have
# been equalized, the threshold for peak/not-peak has not been properly identified yet?

# can we try lowering the threshold? Does this imply something about how to pick it? based on the
# image with the _fewest_ peaks, and then move upwards?

# if we were to do this with the "find once and then measure" approach we'd probably want to start
# with the slice that has the MOST peaks.