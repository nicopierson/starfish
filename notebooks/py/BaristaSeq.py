#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from itertools import product

import imreg_dft.imreg as imr
import numpy as np
import pandas as pd
import skimage.filters
import skimage.morphology
from skimage.transform import SimilarityTransform, warp
from tqdm import tqdm

import starfish.data
import starfish.plot
from starfish.spots import SpotFinder
from starfish.types import Indices

# ======================================= LOAD THE DATA ===========================================

exp = starfish.data.BaristaSeq()
fov = exp['fov_000']
img = fov['primary']
nissl = fov['nuclei']  # this is actually dots, should change this.

# ===================================== MAX PROJECT OVER Z ========================================

z_projected_image = img.max_proj(Indices.Z)
z_projected_nissl = nissl.max_proj(Indices.Z)

# ====================================== ALIGN C CHANNEL ==========================================

# Translate the C channel to account for filter misalignment in the imaging system that this data
# was generated on. This should be done with an arbitrary warping that can be applied to the
# ImageStack

transform = SimilarityTransform(translation=(1.9, -0.4))
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

# ======================= HIGH PASS FILTER TO ELIMINATE CAMERA NOISE ==============================

ghp = starfish.image.Filter.GaussianHighPass(sigma=1)
high_passed = ghp.run(z_projected_image, in_place=False)

# ======================= CORRECT FOR BLEED THROUGH FOR ILLUMINA SBS ==============================

# Illumina SBS reagents have significant spectral overlap. This linear unmixing step corrects
# for these problems by creating an out-of-place intermediate which it can optionally overwrite
# atop the input image if in_place=True

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


def do_bleed_correction(stack):
    bleed_corrected = deepcopy(stack)

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


bleed_corrected = do_bleed_correction(high_passed)

# ==================================== REMOVE BACKGROUND ==========================================

# background is removed by subtracting the result of an opening operation run on the data with a
# 20-radius disk


def opening(image):
    """Extract the background (morphological opening)"""
    # ball is extremely slow, using disk instead
    selem = skimage.morphology.disk(radius=20)
    background = skimage.morphology.opening(image, selem=selem)
    return np.maximum(image - background, 0)


background_subtracted = bleed_corrected.apply(
    opening, in_place=False, verbose=True
)

# ============================== EQUALIZE CHANNEL DYNAMIC RANGE ===================================

# scale each channel by the 99.9th percentile, setting each channel's maximum intensity to 1, and
# removing outlier expression values in the top 0.01%.


def scale_by_channel_percentile(data: np.ndarray, percentile: float) -> np.ndarray:
    cval = np.percentile(data, percentile)
    assert cval > 0
    data[data > cval] = cval
    return data / cval


scaled = background_subtracted.apply(
    scale_by_channel_percentile, group_by={Indices.CH}, in_place=False, verbose=True,
    percentile=99.9
)

# ==================================== FINE REGISTRATION ==========================================

# Rigid similarity registration is fit in the fourier domain. BaristaSeq requires somewhat more
# complex registration that can fit shearing or non-rigid warpings. We cannot account for this at
# the current time, but we can get close.

# It may be possible to fit a piecewise rigid affine transformation, however we'll wait for the
# ImageStack Cropping Framework for that.

# TODO ambrosejcarr: Our cropping framework should be applicable to ImageStacks in addition to
# experiments (it should run on xarrays!). This would get us closer to supportting the piecewise
# rigid affine transform.

# apply a transformation to some subset of the array and either (1) return result in-place or
# return a complete array where part has been transformed.
# TODO it would be nice to have a concatenate function so these transformations could be applied
#      separately and then merged.


channel_projected = scaled.max_proj(Indices.CH)
registered = deepcopy(scaled)
target_rounds = np.setdiff1d(scaled.coordinates[Indices.ROUND.value].values, np.array([0]))
for round_ in target_rounds:

    template_selector = {
        Indices.CH.value: 0,
        Indices.ROUND.value: 0,
        Indices.Z.value: 0,
    }

    subject_selector = {
        Indices.CH.value: 0,
        Indices.ROUND.value: round_,
        Indices.Z.value: 0,
    }

    template = channel_projected.get_slice(template_selector)[0]
    subject = channel_projected.get_slice(subject_selector)[0]
    res = imr.similarity(template, subject)

    similarity_matrix = SimilarityTransform(
        scale=res['scale'],
        rotation=res['angle'] * (np.pi / 180),  # convert to radians
        translation=res['tvec'],
    )

    # apply the transform to each (r, c) pair
    for ch in scaled.coordinates[Indices.CH.value].values:
        application_selector = {
            Indices.CH.value: ch,
            Indices.ROUND.value: round_,
            Indices.Z.value: 0,
        }
        template = registered.get_slice(application_selector)[0]
        transformed = warp(template, similarity_matrix)
        registered.set_slice(
            application_selector,
            transformed.astype(np.float32),
        )

# TODO this is producing some weird artifacts, fix them!

# ========================================= BLUR SPOTS ============================================

# Blur the spots to spread information slightly such that pixel-wise decoding is less sensitive to
# locality. May be needed to apply a kernel-maximum to calculate a local search
# (as done with SeqFISH)

glp = Filter.GaussianLowPass(sigma=1)
low_passed = glp.run(registered)

# ========================================= FIND SPOTS ============================================

psd = SpotFinder.PixelSpotDetector()
psd.run(registered)
