import numpy as np
import SimpleITK as sitk
import pydicom
import torch
from monai.transforms import Compose, AsDiscrete, Activations
from monai.transforms import (
    LoadImaged,
    Resized,
    NormalizeIntensityd,
    RepeatChanneld,
    EnsureChannelFirstd,
    ScaleIntensityd,
)
import monai
from monai.data import list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
import random
import string

"""
This module contains functions related to preprocessing of input images, prediction of lung segmentation mask on th
preprocessed image and postprocessing of the predicted mask.
"""


def _load_model(lung_segment_model_path):
    """
    Load model from the pretrained lung segmentation model path
    Args:
        lung_segment_model_path(pathlib.Path): Pretrained Lung segmentation
                                               model path
    Returns:
       model(torch.nn.Module): Model architecture
       device(torch.device): Device to test the model on
    """

    device = torch.device("cuda")
    model = torch.load(str(lung_segment_model_path))
    model.eval()

    return model, device


def _srgb2gray(image):
    # Convert sRGB image to gray scale and rescale results to [0,255]
    channels = [
        sitk.VectorIndexSelectionCast(image, i, sitk.sitkFloat32)
        for i in range(image.GetNumberOfComponentsPerPixel())
    ]
    # linear mapping
    gray_image = (
        1 / 255.0 * (0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])
    )
    # nonlinear gamma correction
    gray_image = (
        gray_image * sitk.Cast(gray_image <= 0.0031308, sitk.sitkFloat32) * 12.92
        + gray_image ** (1 / 2.4)
        * sitk.Cast(gray_image > 0.0031308, sitk.sitkFloat32)
        * 1.055
        - 0.055
    )
    return sitk.Cast(sitk.RescaleIntensity(gray_image), sitk.sitkUInt8)


def _read_image(file):
    try:
        org_img = sitk.ReadImage(file)
    except:  # noqa E722
        ds = pydicom.dcmread(file)
        org_img = sitk.GetImageFromArray(
            ds.pixel_array, isVector=(len(ds.pixel_array.shape) == 3)
        )
    # Some images have a 3rd dimension of size 1, get rid of it.
    if org_img.GetDimension() != 2 and org_img.GetSize()[2] == 1:
        org_img = org_img[:, :, 0]
    # Some images are grayscale but the channel is repeated three times
    # (gray RGB image).
    if org_img.GetNumberOfComponentsPerPixel() > 1:
        org_img = _srgb2gray(org_img)

    return org_img


def _resample_cxr(new_size, gaussian_sigma, org_img):
    """
    Downsample the input image to the given new_size. To avoid aliasing
    artifacts you may want to blur the image before the downsampling operation.
    This is important if your image contains high frequency data.
    Args:
        new_size: The size of the resampled image in pixels.
        gaussian_sigma(scalar or tuple with image dimension length): If given,
               blur the image with a Gaussian with the given standard
               deviation(s) before resampling.
        file (str): File path to image we want to resample.
    Returns:
        Tuple (SimpleITK.Image, SimpleITK.Image): Original image and it's
                                                normalized resampled image.
    """

    new_spacing = [
        sz * spc / nsz
        for nsz, sz, spc in zip(new_size, org_img.GetSize(), org_img.GetSpacing())
    ]
    smoothed_image = sitk.SmoothingRecursiveGaussian(org_img, gaussian_sigma)
    resampled_for_seg = sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        org_img.GetOrigin(),
        new_spacing,
        org_img.GetDirection(),
        0,
        sitk.sitkFloat32,
    )
    return resampled_for_seg


def _get_channels(model):
    """
    Get number of channels from the trained model.
    Args:
       model(torch.nn.Module): Model architecture
    Returns:
       num_channels(int): Number of channels that the model was trained with.
    """
    first_parameter = next(model.parameters())
    input_shape = first_parameter.size()
    num_channels = input_shape[1]
    return num_channels


def generate_random_string():
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(8))


def _predict_mask(temp_filename, model, device, model_input_size, threshold=0.5):
    """
    Predict the lung mask from the resampled image array. This function uses
    trained segmentation models(torch) to segment lungs for a given image with
    size equal to model input size.
    Args:
        original_img (SimpleITK.Image): Original image
        model (torch.nn.Module): Lung Segmentation model.
        device(torch.device): Device to test the model on.
        model_input_size(int):  Segmentation model input size.
        threshold(float): Threshold to binarize the mask.
    Returns:
        pred_mask_original_size(numpy array): Prediction masks of segmented lungs with same
                                  size as input array.
    """

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=threshold)])
    test_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Resized(keys=["img"], spatial_size=model_input_size, mode=("bilinear")),
            RepeatChanneld(keys=["img"], repeats=3),
            ScaleIntensityd(keys=["img"]),
            NormalizeIntensityd(
                keys=["img"],
                subtrahend=[0.485, 0.456, 0.406],
                divisor=[0.229, 0.224, 0.225],
                channel_wise=True,
            ),
        ]
    )

    test_files = [{"img": temp_filename}]
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    with torch.no_grad():
        test_data = next(iter(test_loader))
        test_image = test_data["img"].to(device)
        roi_size = (96, 96)
        sw_batch_size = 4
        pred_mask = sliding_window_inference(test_image, roi_size, sw_batch_size, model)
        pred = post_trans(decollate_batch(pred_mask)[0])
        pred_mask = np.transpose(pred[1].cpu().numpy(), [1, 0]).astype(np.int32)

    original_img = sitk.ReadImage(temp_filename)
    pred_mask_original_size = _postprocess_lung(pred_mask, original_img)

    return pred_mask_original_size


def _postprocess_lung(np_lung_segmentation_mask, original_img):
    """
    Resample a subregion of a given image based on a mask so as to contain bounding box cropped region of lungs.
    Args:
       original_image (SimpleITK.Image): Original Image
       resampled_image (SimpleITK.Image): Image that was resampled to create
                                           sitk_lung_segmentation_image.
                                         Both images occupy the same physical
                                         space, but they differ in
                                         size (pixel count) and spacing.
       np_lung_segmentation_mask (numpy.array): Array denoting segmentation
                                                mask.
                                                Array size matches the
                                                sitk_lung_segmentation_image
                                                image size.
       new_size (list like): The region in the original_image defined by the
                               segmentation mask is resampled
                             to this size.
       gaussian_sigma(scalar or tuple with image dimension length): If given,
               blur the image with a Gaussian with the given standard
               deviation(s) before resampling.
    Returns:
        SimpleITK.Image and Confidence: Returns the lung segmented region in
                                        the  original image which is resampled
                                        to the given size and returns segmentation
                                        confidence if the segmentation process
                                        of the CXR was done well.
    """
    lung_labels = [1, 2]
    segmentation_image = sitk.GetImageFromArray(np_lung_segmentation_mask)

    # segmentation_image.CopyInformation(resampled_image)
    # relabel the segmentation so that the labels 1, and 2 correspond to the largest
    # components, assumed to be the lungs
    disjointed_segmentation_image = sitk.RelabelComponent(
        sitk.ConnectedComponent(segmentation_image), sortByObjectSize=True
    )
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(disjointed_segmentation_image)

    pred_mask_disjointed = sitk.GetArrayFromImage(disjointed_segmentation_image)
    #  'Left' and 'Right' terms used in this function refers to the side
    # corresponding to the clinical interpretation. Number of pixels are thresholded
    # based on the minimum computed value from the dataset used to train the
    # lung segmentation model.(https://github.com/v7labs/covid-19-xray-dataset)

    threshold_for_num_pixels_left_lung = 2266
    threshold_for_num_pixels_right_lung = 2390
    try:
        # Some of the images which are successfully segmented do not contain
        # lungs in the images.So we threshold the perimeter of the 2 lung
        # regions.Area  is not taken as a measure because some of the
        # successfully segmented images  have a major area  difference. E.g:
        # 'CHNCXR_0361_1.png'(Due to the existence  of large airspaces in the
        # lungs). So area is not taken as a measure for  filtering out the
        # "non-lung" containing images.

        # Check for left and right lung labels by grabbing the x-coordinates.
        left_lung_x_coordinate = label_shape_filter.GetCentroid(lung_labels[0])[0]
        right_lung_x_coordinate = label_shape_filter.GetCentroid(lung_labels[1])[0]

        if left_lung_x_coordinate < right_lung_x_coordinate:
            lung_labels = [2, 1]
        num_pixels_left_lung = label_shape_filter.GetNumberOfPixels(lung_labels[0])
        num_pixels_right_lung = label_shape_filter.GetNumberOfPixels(lung_labels[1])
        if not (
            num_pixels_left_lung >= threshold_for_num_pixels_left_lung
            and num_pixels_right_lung >= threshold_for_num_pixels_right_lung
        ):  # Threshold
            np_lung_segmentation_mask = np.ones(
                segmentation_image.GetSize(), dtype=np.uint8
            )
        else:
            # Remove noise regions that are not from predicted lung regions label
            np_lung_segmentation_mask = (
                (pred_mask_disjointed == lung_labels[0])
                + (pred_mask_disjointed == lung_labels[1])
            ).astype(np.uint8)
    except:  # noqa E722
        # Some predicted masks contain no lung masks(no '1's in
        # 'np_lung_segmentation_mask' array)
        # Assigning lung predicted masks to all ones would essentially select
        # all the
        # area of the original image in the end
        np_lung_segmentation_mask = np.ones(
            segmentation_image.GetSize(), dtype=np.uint8
        )

    segmentation_image = sitk.GetImageFromArray(np_lung_segmentation_mask)

    new_spacing = [
        sz * spc / nsz
        for nsz, sz, spc in zip(
            original_img.GetSize(),
            segmentation_image.GetSize(),
            segmentation_image.GetSpacing(),
        )
    ]

    predicted_mask_original_size = sitk.Resample(
        segmentation_image,
        original_img.GetSize(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        segmentation_image.GetOrigin(),
        new_spacing,
        segmentation_image.GetDirection(),
        0,
        original_img.GetPixelIDValue(),
    )

    return predicted_mask_original_size


def _cropped_lung(original_image, predicted_binary_mask):
    """

    Args:
       original_image (SimpleITK.Image): Original Image
       predicted_binary_mask (SimpleITK.Image): Predicted binary mask image
    Returns:
        cropped_lung: Returns the cropped lung segmented region.
    """
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(predicted_binary_mask)
    bounding_box = label_shape_filter.GetBoundingBox(1)
    cropped_lung = original_image[
        bounding_box[0] : bounding_box[0] + bounding_box[2],
        bounding_box[1] : bounding_box[1] + bounding_box[3],
    ]
    return cropped_lung


def _only_lung(original_image, predicted_binary_mask):
    """

    Args:
       original_image (SimpleITK.Image): Original Image
       predicted_binary_mask (SimpleITK.Image): Predicted binary mask image
    Returns:
        only_lung_img: Returns the image which contain only lung regions.
    """
    only_lung_img = original_image * predicted_binary_mask
    return only_lung_img
