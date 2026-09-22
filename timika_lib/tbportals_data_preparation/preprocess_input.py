# This code generates numpy array extracted from segmented lungs of the original image.
from tbportals_data_preparation.preprocess_and_postprocess import (
    _predict_mask,
    _cropped_lung,
)
import warnings
import SimpleITK as sitk

warnings.simplefilter(action="ignore")


def _gen_preprocessed_lung_image(
    temp_filename,
    model,
    device,
    model_input_size,
    seg_threshold,
):
    """
    Generate a preprocessed lung image from original image.This process happens in 3
    stages:
    1.) Reading and Resampling of original image
    2.) Predict mask from the array of resampled image array
    3.) Segment lungs from the original image using the predicted mask of origina size
    Args:
        original_img (SimpleITK.Image):  Original SimpleITK image
        model (torch.nn.Module): Lung Segmentation model.
        device(torch.device): Device to test the model on
        model_input_size(tuple): size required as lung segmentation model input
        gaussian_sigma(scalar or tuple with image dimension length): If given,
               blur the image with a Gaussian with the given standard
               deviation(s) before resampling.
        seg_threshold(float): Threshold to binarize the mask.

    Returns:
        cropped_lung_image(SimpleITK.Image): Cropped lung image
    """

    # Predict Image
    original_img = sitk.ReadImage(temp_filename)
    pred_mask_original_size = _predict_mask(
        temp_filename,
        model,
        device,
        model_input_size,
        threshold=seg_threshold,
    )

    # Segment Lungs from Original Images and preprocess the CXR to only contain lung regions or bounding box'd
    # cropped lung regions CXR.
    cropped_lung_image = _cropped_lung(original_img, pred_mask_original_size)

    return cropped_lung_image
