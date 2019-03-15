import numpy as np

from starfish import Codebook, ImageStack, IntensityTable
from starfish.recipe.runcommand import Runnable


def test_get_file_input_types():
    """Verify that we can map from inputs that are meant to be filepaths to the expected object
    type.

    If you're looking at this because this test is failing, ask yourself if you've changed
    PerRoundMaxChannelDecoder (either the name of the algorithm or its parameters).  If so, then
    you can update the expected outcome of the test.  If you did not, then it's probably a
    legitimate failure.
    """
    runnable = Runnable(
        "decode", "PerRoundMaxChannelDecoder",
        "should_be_intensity_table", "should_be_codebook")
    file_input_types = runnable.get_file_input_types()
    assert len(file_input_types) == 2
    assert file_input_types["should_be_intensity_table"] == IntensityTable
    assert file_input_types["should_be_codebook"] == Codebook


def test_get_file_input_types_from_runnable():
    """Verify that we can map from inputs, some of which are meant to be filepaths, to the expected
     object type.

    If you're looking at this because this test is failing, ask yourself if you've changed
    BlobDetector or PerRoundMaxChannelDecoder (either the name of the algorithm or its parameters).
    If so, then you can update the expected outcome of the test.  If you did not, then it's probably
    a legitimate failure.
    """
    intensity_table_runnable = Runnable(
        "detect_spots", "BlobDetector",
        "should_be_image_stack",
        min_sigma=1,
        max_sigma=1,
        num_sigma=1,
        threshold=1,
        overlap=1,
    )
    runnable = Runnable(
        "decode", "PerRoundMaxChannelDecoder",
        intensity_table_runnable, "should_be_codebook")

    file_input_types = intensity_table_runnable.get_file_input_types()
    assert len(file_input_types) == 1
    assert file_input_types["should_be_image_stack"] == ImageStack

    file_input_types = runnable.get_file_input_types()
    assert len(file_input_types) == 1
    assert file_input_types["should_be_codebook"] == Codebook


def test_run():
    filter_runnable = Runnable(
        "filter", "WhiteTophat",
        "https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/fov_001/hybridization.json",
        masking_radius=15,
        is_volume=False,
    )
    result = filter_runnable.run()
    assert isinstance(result, ImageStack)

    # pick a random part of the registered image and assert on it
    expected_filtered_values = np.array(
        [[0.1041123, 0.09968718, 0.09358358, 0.09781034, 0.08943313, 0.08853284,
          0.08714428, 0.07518119, 0.07139697, 0.0585336, ],
         [0.09318685, 0.09127947, 0.0890364, 0.094728, 0.08799877, 0.08693064,
          0.08230717, 0.06738383, 0.05857938, 0.04223698],
         [0.08331426, 0.0812543, 0.08534371, 0.0894789, 0.09184404, 0.08274967,
          0.0732433, 0.05564965, 0.04577706, 0.03411917],
         [0.06741435, 0.07370108, 0.06511024, 0.07193103, 0.07333485, 0.07672236,
          0.06019684, 0.04415961, 0.03649958, 0.02737468],
         [0.05780118, 0.06402685, 0.05947966, 0.05598535, 0.05192646, 0.04870679,
          0.04164187, 0.03291371, 0.03030441, 0.02694743],
         [0.04649424, 0.06117342, 0.05899138, 0.05101091, 0.03639277, 0.03379873,
          0.03382925, 0.0282597, 0.02383459, 0.01651026],
         [0.0414435, 0.04603647, 0.05458152, 0.04969863, 0.03799496, 0.0325475,
          0.02928206, 0.02685588, 0.02172885, 0.01722743],
         [0.04107728, 0.04161135, 0.04798963, 0.05156023, 0.03952087, 0.02899214,
          0.02589456, 0.02824444, 0.01815823, 0.01557945],
         [0.03901731, 0.03302052, 0.03498893, 0.03929199, 0.03695735, 0.02943466,
          0.01945525, 0.01869231, 0.01666284, 0.01240558],
         [0.02664226, 0.02386511, 0.02206454, 0.02978561, 0.03265431, 0.0265507,
          0.02214084, 0.01844815, 0.01542687, 0.01353475]],
        dtype=np.float32
    )

    assert result.xarray.dtype == np.float32

    assert np.allclose(
        expected_filtered_values,
        result.xarray[2, 2, 0, 40:50, 40:50]
    )
