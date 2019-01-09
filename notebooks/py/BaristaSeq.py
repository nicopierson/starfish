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
    indices = {Indices.ROUND: round_, Indices.CH: ch, Indices.Z: 0}
    tile = z_projected_image.get_slice(indices)[0]
    transformed = warp(tile, transform)
    z_projected_image.set_slice(
        indices=indices,
        data=transformed.astype(np.float32),
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
        bleed_corrected.set_slice(
            {Indices.CH: int(ch2)},
            corrected,
            axes=[Indices.ROUND, Indices.Z]
        )
    return bleed_corrected


bleed_corrected = do_bleed_correction(low_passed)


def opening(image):
    """Extract the background (morphological opening)"""
    # ball is extremely slow, using disk instead
    selem = skimage.morphology.disk(radius=20)
    background = skimage.morphology.opening(image, selem=selem)
    return np.maximum(image - background, 0)


background_subtracted = bleed_corrected.apply(
    opening, in_place=False, verbose=True
)

# Registration

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

# check if translation was enough; max project channels and view rounds as an RGB image
from skimage.exposure import rescale_intensity
rgb = np.squeeze(registered.max_proj(Indices.CH).xarray.values).T
plt.imshow(rescale_intensity(rgb))
plt.show()

# definitely still needs registration. Call spots, use those as anchors, and run registration
# against the first round.

# call spots
for round in range(rgb.shape[-1]):
    pass  # run spot finding to get peaks
    # break into zones, select 25 highest intensity peaks in each quadrant.

# registration here works pretty well, but we need to fit a _rotation_
# it's easy to view the failure state by creating an RGB image of the data
# non-rigid ICP will be needed for this data.


# Find Rolonies

# Authors use FIJI rolony finding. This method takes a blob size and an aboslute tolerance which
# defines the (subtractive) difference between the maximum and the (local) background.
# the closest type of spot finder we have is the Trackpy locate. However, it works badly,
# likely due to the poor registration. This data _really_ needs rotation.

# Their spots are about 7 pixels
# Bleed through isn't always completely corrected. In these cases there can be small (1.5px)
# shifts across channels to correct for this, authors merge rolonies within 1.5px of each other.

# they prefer finding spots in each channel to doing the max projection. They state that the max
# projection could result in information loss because the per-channel information
# might allow separation of rolonies that are overlaping in the max projection, but separate in
# channel space.

# # this errors, I think because no spots are found.
# lmpf = SpotFinder.TrackpyLocalMaxPeakFinder(  # type: ignore
#     spot_diameter=11, min_mass=0.01, max_size=100, separation=11 * 1.5,
#     is_volume=False, preprocess=False, noise_size=[0, 0, 0]
# )

# the pixel-based approach works much better.

psd = SpotFinder.PixelSpotDetector(  # type: ignore
    codebook=exp.codebook,
    metric='euclidean', distance_threshold=0.517, magnitude_threshold=0.0005,
    min_area=4, max_area=50, norm_order=2
)

intensities, ccdr = psd.run(registered)

# try gaussian spot finder
gsd = SpotFinder.GaussianSpotDetector(
    min_sigma=2, max_sigma=5, num_sigma=10, threshold=0.002, overlap=0.5,
    reference_image_from_max_projection=True
)

intensities = gsd.run(registered)
