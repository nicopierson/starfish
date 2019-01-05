#!/usr/bin/env python
# coding: utf-8

import copy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.filters
import skimage.morphology
from skimage.transform import AffineTransform, warp
from tqdm import tqdm

import starfish.data
import starfish.plot
from starfish.image import Registration
from starfish.spots import SpotFinder
from starfish.types import Indices

# Try to load the data
exp = starfish.data.BaristaSeq()
fov = exp['fov_000']
img = fov['primary']
nissl = fov['nuclei']  # this is actually dots, should change this.

# Max project over Z
z_projected_image = img.max_proj(Indices.Z)
z_projected_nissl = nissl.max_proj(Indices.Z)

# Translate the C channel to account for filter misalignment for the C channel
# This is happening in-place for now.

transform = AffineTransform(translation=(1.9, -0.4))
channels = (0,)
rounds = np.arange(img.num_rounds)
slice_indices = product(channels, rounds)

for ch, round_, in slice_indices:
    indices = {Indices.CH: ch, Indices.ROUND: round_, Indices.Z: 0}
    tile = z_projected_image.get_slice(indices)[0]
    transformed = warp(tile, transform)
    z_projected_image.set_slice(
        indices=indices,
        data=transformed.astype(np.float32),
        axes=[Indices.CH, Indices.ROUND, Indices.Z]
    )


# Median Filter
# this might not be critical, it's designed to remove camera noise. Ask Deep about replacing
# def median_filter(projected_stack):
#     import scipy.signal
#     with warnings.catch_warnings():
#         return scipy.signal.medfilt(projected_stack)
#
#
# median_filtered = z_projected_image.apply(median_filter, in_place=False, verbose=True)

# Instead, Low pass the data to remove camera noise

glp = starfish.image.Filter.GaussianLowPass(sigma=1)
low_passed = glp.run(z_projected_image, in_place=False)

# Do bleed corrections
bleed_correction_factors = pd.DataFrame(
    data=[
        [0, 1, 0.05],
        [0, 2, 0],
        [0, 3, 0],
        [1, 0, 0.35],
        [1, 2, 0],
        [1, 3, 0],
        [2, 0, 0],
        [2, 1, 0.02],
        [2, 3, 0.84],
        [3, 0, 0],
        [3, 1, 0],
        [3, 2, 0.05]
    ],
    columns=(('bleed_from', 'bleed_into', 'factor_bleed_from_into')),
)
# bleed_correction_factors['bleed_from'] = bleed_correction_factors['bleed_from'].astype(int)
# bleed_correction_factors['bleed_into'] = bleed_correction_factors['bleed_into'].astype(int)


def do_bleed_correction(stack):
    bleed_corrected = copy.deepcopy(stack)

    for index, (ch1, ch2, constant) in tqdm(bleed_correction_factors.iterrows()):
        bleed = stack.get_slice({Indices.CH: int(ch1)})[0] * constant
        img_to_correct = stack.get_slice({Indices.CH: int(ch2)})[0]
        corrected = np.maximum(img_to_correct - bleed, 0)
        bleed_corrected.set_slice({Indices.CH: int(ch2)}, corrected)
    return bleed_corrected


bleed_corrected = do_bleed_correction(low_passed)


# Extract the background (morphological opening)
def opening(image):
    # ball is extremely slow, using disk instead
    selem = skimage.morphology.disk(radius=20)
    background = skimage.morphology.opening(image, selem=selem)
    return np.maximum(image - background, 0)


background_subtracted = bleed_corrected.apply(
    opening, in_place=False, verbose=True
)

background_subtracted.show_stack({Indices.ROUND: 0}, rescale=True)

# Registration

# They do some registration pre-processing:
# Channel 3 is translated down 1000 intensity: `np.maximum(0, img - 1000)`
# Channel 1 is multiplied by 5.
# Question for Xiaoyin: What's the point of these modifications?
# Answer: that was for previous experiments and is no longer necessary


# Registration is necessary.
# To do the alignment in the small tiles is challenging because there are potentially large shifts
# in stage alignment across rounds, and so registration benefits from a large image space.
# In this case, registration worked, with shifts of about ~40-50 pixels.
# But that may not always be the case, so they are currently using the pre-stitched images from
# their microscope's software for alignment.

# In the long term, looking at the smaller tiles may give better results (no stitching errors)
# (can result in doubling of spots) -- this would certainly screw up segmentation.

round_1 = background_subtracted.get_slice({Indices.ROUND: 0})[0].reshape(1, 4, 1, 1000, 800)
registration_reference_stack = starfish.ImageStack.from_numpy_array(round_1)

reg = Registration.FourierShiftRegistration(  # type: ignore
    upsampling=1, reference_stack=registration_reference_stack
)

registered = reg.run(background_subtracted)

# To check this, sum the images across channels before and after, look at a small subset of the
# data and check the cell positioning.

# not being done anymore; was being done because the channel was very noisy.
# read each of the channels
# ch3 = np.maximum(ch3 - 1000, 0) * 1.2
# ch1 *= 5
# understand why this is being done?
ch_data = {}
for ch_num in np.arange(registered.num_chs):
    data = registered.get_slice({Indices.CH: ch_num})[0]
    ch_data[ch_num] = pd.Series(np.ravel(data)).describe()

ch_data

# Find Rolonies

# Authors use FIJI rolony finding. This method takes a blob size and an aboslute tolerance which
# defines the (subtractive) difference between the maximum and the (local) background.


test_image = registered.get_slice({Indices.ROUND: 0, Indices.CH: 0})[0]

plt.hist(np.ravel(test_image), log=True, bins=50)

SpotFinder.GaussianSpotDetector()  # type: ignore

lmpf = SpotFinder.LocalMaxPeakFinder(  # type: ignore
    spot_diameter=11, min_mass=0.01, max_size=100, separation=11 * 1.5,
    is_volume=False, preprocess=False, noise_size=[0, 0, 0]
)

# Their spots are about 7 pixels
# Bleed through isn't always completely corrected. In these cases there can be small (1.5px)
# shifts across channels to correct for this, merge rolonies within 1.5px of each other.

# doing the max projection could result in information loss because the per-channel information
# might allow separation of rolonies that are overlaping in the max projection, but separate in
# channel space.

# If the channels are not balanced (one is noisy) then you hurt the weaker channels.

mp = registered.max_proj(Indices.Z, Indices.CH, Indices.ROUND)

psd = SpotFinder.PixelSpotDetector(  # type: ignore
    codebook=exp.codebook,
    metric='euclidean', distance_threshold=0.517, magnitude_threshold=0.0005,
    min_area=4, max_area=50, norm_order=2
)

intensities, ccdr = psd.run(registered)

# non-rigid ICP will be needed for this data.

# Is there a multiplication between the codebook and the image intensity values that could be used
# to visualize a given code?

plt.scatter(intensities['x'], intensities['y'], alpha=0.2, c='r')

lmpf = starfish.spots.SpotFinder.LocalMaxPeakFinder(  # type: ignore
    spot_diameter=7, min_mass=0.0001, max_size=100,
    separation=0.5, noise_size=(1, 1), is_volume=False
)
results = lmpf.run(registered)

intensities[0].shape

results = {}

# write a spot detector
rounds = np.arange(registered.num_rounds)
channels = np.arange(registered.num_chs)
slice_indices = product(channels, rounds)
for ch, round_ in slice_indices:
    results[ch, round_] = skimage.feature.blob_log(
        registered.get_slice({Indices.CH: ch, Indices.ROUND: round_})[0],
        min_sigma=0.2, max_sigma=4, num_sigma=30, threshold=0.0003, overlap=0.5, log_scale=False
    )
    break

# Just loop over channels

channels = np.arange(registered.num_chs)
round_ = 0
for ch in channels:
    results[ch, round_] = skimage.feature.blob_log(
        registered.get_slice({Indices.CH: ch, Indices.ROUND: round_})[0],
        min_sigma=0.2, max_sigma=4, num_sigma=30, threshold=0.0003, overlap=0.5, log_scale=False
    )


def plot_on_image(image, results):
    """results is from blob_log, image should be a 2d-image"""
    showit.image(image, size=20, clim=(0, 0.05))
    plt.scatter(results[:, 2], results[:, 1], alpha=0.4, c='r')


plot_on_image(skimage.exposure.rescale_intensity(registered.get_slice({Indices.ROUND: 0, Indices.CH:0})[0][0, :, :]), results[0, 0])

showit.image(skimage.exposure.rescale_intensity(skimage.filters.gaussian(registered.get_slice({Indices.CH: 1, Indices.ROUND: 0})[0][0, :, :], sigma=1)),
             size=20, clim=(0, 0.05))

showit.image(skimage.exposure.rescale_intensity(registered.get_slice({Indices.CH: 1, Indices.ROUND: 0})[0][0, :, :]), clim=(0, 0.1), size=20)

results[0][0, :]

gsd = SpotFinder.GaussianSpotDetector(
    min_sigma=2, max_sigma=5, num_sigma=10, threshold=0.002, overlap=0.5, reference_image_from_max_projection=True
)

registered.shape

intensities = gsd.run(registered)

codebook = exp.codebook

psd = SpotFinder.PixelSpotDetector(codebook, distance_threshold=0.51, magnitude_threshold=0.0, min_area=0, max_area=np.inf, metric='euclidean')

pixel_results, ccdr = psd.run(background_subtracted)


pd.Series(*np.unique(codebook.decode_per_round_max(peak_results)['target'], return_counts=True)[::-1])

decoded_peak_results = codebook.metric_decode(peak_results, max_distance=0.517, min_intensity=0, norm_order=2)

ccdr.decoded_image.shape

f, ax = plt.subplots(figsize=(10, 10))
starfish.plot.decoded_spots(decoded_image=ccdr.decoded_image[10], ax=ax)

proj_image = background_subtracted.max_proj(Indices.CH, Indices.ROUND, Indices.Z)


plt.hist(np.ravel(proj_image), log=True)

showit.image(proj_image[300:600, :300], clim=(0.0001, 0.002), size=10)
