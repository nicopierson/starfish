import matplotlib.pyplot as plt
from numpy import array, expand_dims
from skimage.io import imread
from skimage import img_as_float32
import napari_gui as gui
from timeit import timeit

%gui qt

path_to_data = '/Users/nicholassofroniew/Documents/DATA-fish/B.tif'
raw_data = imread(path_to_data)

from starfish import ImageStack

stack = ImageStack.from_numpy_array(img_as_float32(expand_dims(expand_dims(patch,0),0)))

window = gui.imshow(stack.xarray.values.squeeze().transpose(2,1,0))
viewer = window.viewer
viewer.layers[-1].colormap = 'viridis'


from starfish.spots import SpotFinder

detector = SpotFinder.BlobDetector(
    min_sigma=.5,
    max_sigma=10,
    num_sigma=5,
    threshold=0.01,
    measurement_type='mean',
)

spots = detector.run(stack)

centers = array([spots.z, spots.y, spots.x]).T

viewer.add_markers(centers[:,[1,2,0]],
                   face_color='green', edge_color='green', symbol='ring', size=8)

def inside(shape, center, window):
    """
    Returns boolean if a center and its window is fully contained
    within the shape of the image on all three axes
    """
    return all([(center[i]-window[i] >= 0) & (center[i]+window[i] <= shape[i]) for i in range(0,3)])

def volume(im, center, window):
    if inside(im.shape, center, window):
        volume = im[(center[0]-window[0]):(center[0]+window[0]), (center[1]-window[1]):(center[1]+window[1]), (center[2]-window[2]):(center[2]+window[2])]
        volume = volume.astype('float64')
        baseline = volume[[0,-1],[0,-1],[0,-1]].mean()
        volume = volume - baseline
        volume = volume/volume.max()
        return volume

window_bead = [14, 14, 14]
beads = [volume(patch, x, window_bead) for x in centers]
beads = [x for x in beads if x is not None]
beads = array(beads)

avg = beads.mean(0)


plt.figure(figsize=(10,10));
plt.subplot(1,3,1);
plt.imshow(avg.mean(0));
plt.subplot(1,3,2);
plt.imshow(avg.mean(1));
plt.subplot(1,3,3);
plt.imshow(avg.mean(2));

## Do 3D deconvolution with RL using learnt psf

from skimage.restoration import richardson_lucy

decon_raw = richardson_lucy(img_as_float32(patch), avg)


decon = decon_raw
decon[:,:window_bead[1]+1,:] = 0
decon[:,-window_bead[1]-1:,:] = 0
decon[:window_bead[0]+1,:,:] = 0
decon[-window_bead[0]-1:,:,:] = 0
decon[:,:,:window_bead[2]+1] = 0
decon[:,:,-window_bead[2]-1:] = 0


plt.figure(figsize=(15,15));
plt.subplot(1,2,1);
plt.imshow(patch.max(axis=0), clim=(200, 2000));
plt.subplot(1,2,2);
plt.imshow(decon.max(axis=0), clim=(0, .2));


beadsD = [volume(decon, x, window_bead) for x in centers]
beadsD = [x for x in beadsD if x is not None]
beadsD = array(beadsD)

avgD = beadsD.mean(0)

plt.figure(figsize=(10,10));
plt.subplot(1,3,1);
plt.imshow(avgD.mean(0));
plt.subplot(1,3,2);
plt.imshow(avgD.mean(1));
plt.subplot(1,3,3);
plt.imshow(avgD.mean(2));

plt.figure(figsize=(20,5));
plt.subplot(1,3,1);
plt.plot(avg.mean((0,1)));
plt.plot(avgD.mean((0,1)));
plt.subplot(1,3,2);
plt.plot(avg.mean((0,2)));
plt.plot(avgD.mean((0,2)));
plt.subplot(1,3,3);
plt.plot(avg.mean((1,2)));
plt.plot(avgD.mean((1,2)));

## estimate fwhm


from scipy.optimize import curve_fit

def fit(yRaw):
    y = yRaw - (yRaw[0]+yRaw[-1])/2
    x = (array(range(y.shape[0])) - y.shape[0]/2)
    x = (array(range(y.shape[0])) - y.shape[0]/2)
    popt, pcov = curve_fit(gauss, x, y, p0 = [1, 0, 1, 0])
    FWHM = 2.355*popt[2]
    yFit = gauss(x, *popt)
    return x, y, yFit, FWHM


from numpy import exp

def gauss(x, a, mu, sigma, b):
    return a*exp(-(x-mu)**2/(2*sigma**2))+b


profile_lat = (avgD.mean((0,1)) + avgD.mean((0,2)))/2
profile_ax = avgD.mean((1,2))

plt.plot(profile_lat);
plt.plot(profile_ax);

x, y, yFit, FWHM = fit(profile_lat)


plt.plot(x, y, '.');
plt.plot(x, yFit);

FWHM  # (full width at half maximum)

sigma = FWHM/2/3**(.5)

sigma

avgD.max()

## Redo spot detection on deconvolved stack¶

stackD = ImageStack.from_numpy_array(img_as_float32(expand_dims(expand_dims(decon,0),0)))


viewer.add_image(stackD.xarray.values.squeeze().transpose(2,1,0),{})
viewer.layers[-1].colormap = 'viridis'
viewer.layers[-1].clim = (0, 0.2)

detector = SpotFinder.BlobDetector(
    min_sigma=1,
    max_sigma=2,
    num_sigma=4,
    threshold=0.01,
    measurement_type='mean',
)

# blobs = dots; define the spots in the dots image, but then find them again in the stack.
spotsD = detector.run(stackD)

centersD = array([spotsD.z, spotsD.y, spotsD.x]).T

viewer.add_markers(centersD[:,[1,2,0]],
                   face_color='red', edge_color='red', symbol='ring', size=8)