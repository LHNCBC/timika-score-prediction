import os
import torch
import argparse
import pathlib
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
from monai.transforms import Compose, AsDiscrete, Activations
from monai.transforms import (
    LoadImaged,
    Resized,
    NormalizeIntensityd,
    RepeatChanneld,
    EnsureChannelFirstd,
    ScaleIntensityd,
)
import tempfile
import monai
from monai.data import list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference

"""
This script is used to predict the binary lung masks on the test Chest X Rays using the
pretrained lung segmentation model.User can provide the output prediction directory
name to save the binary lung masks in the given folder.
"""


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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(str(lung_segment_model_path), map_location=device)
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


def _predict_mask(file_path, model, device, model_input_size, threshold=0.5):
    """
    Predict the lung mask from the resampled image array. This function uses
    trained segmentation models(torch) to segment lungs for a given image with
    size equal to model input size.
    Args:
        resampled_image_arr (numpy array): Numpy array obtained from resampled
                                           images to provide input for
                                           segmentation network in the
                                           shape of (num_images,
                                                     segmentation_input_size_x,
                                                     segmentation_input_size_y)
        model (torch.nn.Module): Lung Segmentation model.
        device(torch.device): Device to test the model on.
        model_input_size(int): Model input
        batch_size: Batch size for the model. Performing inference in batch
                    mode is faster than image by image.
    Returns:
        pred_masks(numpy array): Prediction masks of segmented lungs with same
                                  size as input array.
    """
    original_img = _read_image(file_path)

    temp_file = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
    temp_filename = temp_file.name

    sitk.WriteImage(original_img, temp_filename)

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
    pred_mask = sitk.GetImageFromArray(pred_mask)

    # Post process the mask to remove small connected components
    disk = [2, 2, 2]
    pred_mask = sitk.BinaryMorphologicalOpening(pred_mask, disk)
    pred_mask = sitk.BinaryFillhole(pred_mask)
    pred_mask = sitk.BinaryMorphologicalClosing(pred_mask, disk)

    # Retain the two largest connected component
    pred_mask = sitk.ConnectedComponent(pred_mask)
    pred_mask = sitk.RelabelComponent(pred_mask)
    pred_mask = sitk.BinaryThreshold(pred_mask, 1, 2)

    new_spacing = [sz * spc / nsz
        for nsz, sz, spc in zip(
            original_img.GetSize(), pred_mask.GetSize(), pred_mask.GetSpacing())]

    pred_mask_original_size = sitk.Resample(
        pred_mask,
        original_img.GetSize(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        original_img.GetOrigin(),
        new_spacing,
        original_img.GetDirection(),
        0,
        sitk.sitkUInt8,
    )

    temp_file.close()
    return pred_mask_original_size


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lung Segmentation Model in Chest X Rays."
    )

    parser.add_argument(
        "input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing \
                                column names as 'input file name' which represent \
                                paths of  CXRs and their corresponding binary masks respectively",
    )

    parser.add_argument(
        "lung_segmentation_model_path",
        type=pathlib.Path,
        help="Model path \
                        for pretrained lung segmentation",
    )
    parser.add_argument(
        "output_pred_dir",
        type=str,
        help="Output Directory to \
                        save the prediction images in their original images",
    )

    parser.add_argument(
        "--img_size",
        default=[224, 224],
        help="Image size to \
                        resample the input test images for the model to predict on.",
        nargs=2,
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size to test the model"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to convert the predicted scores to binary lung mask. \
                                            If prob. > threshold --> lung voxel,else --> background",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="No. of sub processes \
                                                    that tells the data loader instance to use\
                                                     for data loading",
    )
    args = parser.parse_args()

    model, device = _load_model(args.lung_segmentation_model_path)

    df = pd.read_csv(args.input_csv_path)
    if not os.path.exists(args.output_pred_dir):
        os.makedirs(args.output_pred_dir)

    for file in df["input file name"]:
        mask = _predict_mask(file, model, device, args.img_size)

        filename = os.path.splitext(os.path.basename(file))[0]
        seg_folder = os.path.join(args.output_pred_dir, (file.split('../data/')[1]).split(filename)[0])
        os.makedirs(seg_folder, exist_ok=True)

        """output_seg_file = seg_folder + filename + "_lung_pred_seg.nii.gz"
        mask.SetSpacing([1.0, 1.0])
        sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), output_seg_file)"""

        # Read file using _read_image function
        original_img = _read_image(file)
        # Find bounding box of mask and crop the original image
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array[mask_array > 0] = 1
        bbox = np.argwhere(mask_array)
        y_min, x_min = bbox.min(axis=0)
        y_max, x_max = bbox.max(axis=0) + 1
        original_img = original_img[x_min:x_max, y_min:y_max]
        # Save the cropped image
        cropped_img_file = seg_folder + filename + ".png"
        print(cropped_img_file)
        sitk.WriteImage(original_img, cropped_img_file)

if __name__ == "__main__":
    main()
