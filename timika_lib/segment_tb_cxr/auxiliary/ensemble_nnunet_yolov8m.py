import os
import argparse
import pandas as pd
import SimpleITK as sitk
import pydicom
from ultralytics import YOLO
import torch
import numpy as np
import sys
import contextlib
import io
import warnings


# ---------------------------------------------------------------------------
# Vendored inline (from segment_tb_cxr/unet_resnet18/inference/inference_tb_segment.py:103-140)
# so this module has no intra-repo dependency on inference_tb_segment.py /
# train_tb_segment.py (which would otherwise drag in tensorboard + monai.visualize).
# Only `_read_image` is needed by the ensemble functions below; the CLI-only
# argparse validators `file_path`/`csv_path` are not used by this notebook.
# ---------------------------------------------------------------------------
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

"""
This script generates probability maps from YOLOv8 and nnU-Net models,
computes their mean to create an ensemble, and then applies a threshold
to generate a binary segmentation mask. The resulting mask is saved
to the specified output folder. It takes in a csv with column name
'filename', weight file paths of YOLOv8 and nnUNet, output segmentation
folder to save the generated ensemble predictions and output csv filename with
an extra column name 'ensemble_pred_tb_seg_file' corresponding to the original
filenames. The prediction in the output folder are generated with
{filename}_ensemble_pred_seg.nrrd format.
"""


@contextlib.contextmanager
def suppress_stdout():
    new_target = io.StringIO()
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


# Use the context manager to suppress output from nnunet library when its
# variables are not exported.
with suppress_stdout():
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

warnings.filterwarnings("ignore")


def gen_yolov8_prob_map(file, yolov8_model):
    """
    Generates a probability map for the presence of specific regions (e.g., TB) in an input image using a YOLOv8 model.

    This function processes an input image, applies a YOLOv8 model to predict
    probability masks, and resamples the results back to the original image
    size. If no regions are detected by the model, the function returns `None`.

    Parameters:
    ----------
    file : str
        Path to the input image file.
    yolov8_model : YOLOv8 model object
        Pre-trained YOLOv8 model used for predicting probability masks.

    Returns:
    -------
    np.ndarray or None
        A 2D NumPy array representing the resampled probability map in the original image dimensions.
        If no regions are detected by the YOLOv8 model, returns `None`.
    """
    original_img = _read_image(file)

    # Yolov8 expects inputs to be in uint8 format scaled to [0-255].
    # Different intensity ranges result in different results.
    rescaled_img = sitk.Cast(
        sitk.RescaleIntensity(original_img, 0, 255), sitk.sitkUInt8
    )

    img_arr = sitk.GetArrayViewFromImage(rescaled_img)
    img_arr = np.expand_dims(img_arr, -1)
    img_arr = np.repeat(img_arr, 3, 2)
    results = yolov8_model.predict(
        source=img_arr, save=False, save_txt=False, verbose=False
    )

    if (
        results[0].prob_masks is not None
    ):  # If the yolov8 predictions does not contain any regions of "TB"
        cropped_prob_masks = results[0].prob_masks.data.cpu().numpy()
        combined_mask = cropped_prob_masks.sum(axis=0)

        combined_mask = np.clip(combined_mask, 0, 1)

        result_image = sitk.GetImageFromArray(combined_mask)

        new_spacing = [
            sz * spc / nsz
            for nsz, sz, spc in zip(
                original_img.GetSize(),
                result_image.GetSize(),
                result_image.GetSpacing(),
            )
        ]
        pred_mask_original_size = sitk.Resample(
            result_image,
            original_img.GetSize(),
            sitk.Transform(),
            sitk.sitkLinear,
            original_img.GetOrigin(),
            new_spacing,
            original_img.GetDirection(),
            0,
            sitk.sitkFloat32,
        )

        yolov8_prob = sitk.GetArrayFromImage(pred_mask_original_size)

    else:

        yolov8_prob = None

    return yolov8_prob


def gen_nnunet_prob_map(file, predictor):
    """
    Generates a probability map for an input image using a pre-trained nnU-Net model.

    This function reads an input image, initializes an nnU-Net predictor with specified settings,
    and performs inference using the provided nnU-Net weights. It returns the predicted probability map.

    Parameters:
    ----------
    file : str
        Path to the input image file.
    predictor : str
        Initialized and loaded nnUNet model.

    Returns:
    -------
    np.ndarray
        A NumPy array representing the predicted probability map from nnU-Net for the input image.

    """
    original_img = _read_image(file)

    npy_image = sitk.GetArrayFromImage(original_img)
    npy_image = npy_image[None, None]
    max_spacing = max(original_img.GetSpacing())
    spacings_for_nnunet = [max_spacing * 999, *list(original_img.GetSpacing())[::-1]]

    image_props = {"spacing": spacings_for_nnunet}

    # Perform prediction and get the probabilities as a numpy array
    nnunet_mask, nnunet_prob = predictor.predict_single_npy_array(
        npy_image, image_props, save_or_return_probabilities=True
    )

    return nnunet_prob[1][0]


def gen_ensembled_yolov8_nnunet_segmentation(
    file, yolov8_prob_map, nnunet_prob_map, output_seg_folder, binary_threshold=0.5
):
    """
    This function takes probability maps from YOLOv8 and nnU-Net, computes their mean to create an ensemble,
    and then applies a threshold to generate a binary segmentation mask. The resulting mask is saved
    to the specified output folder.

    Parameters:
    ----------
    file : str
        Path to the input image file. Used to copy metadata and determine the output filename.
    yolov8_prob_map : np.ndarray
        Probability map generated by the YOLOv8 model.
    nnunet_prob_map : np.ndarray
        Probability map generated by the nnU-Net model.
    output_seg_folder : str
        Directory where the resulting binary segmentation mask will be saved.

    Returns:
    -------
    None
        The binary segmentation mask is written to the output folder as a `_seg.nrrd` file.

    """
    if yolov8_prob_map is not None:
        ensembled_prob = np.mean([yolov8_prob_map, nnunet_prob_map], axis=0)
    else:
        ensembled_prob = nnunet_prob_map / 2

    """ensemble_nnunet_yolov8_prob_map_img = sitk.GetImageFromArray(ensembled_prob)
    original_img = _read_image(file)
    ensemble_nnunet_yolov8_prob_map_img.CopyInformation(original_img)"""

    filename = os.path.splitext(os.path.basename(file))[0]
    # Robustness patch (vendored copy): the original derived a nested output
    # sub-path by splitting on the literal '../data/' in the input path, which
    # crashes for any path that does not contain '../data/'. Preserve that
    # behavior when the marker is present, else write flat into output_seg_folder.
    if "../data/" in file:
        seg_folder = os.path.join(
            output_seg_folder, (file.split("../data/")[1]).split(filename)[0]
        )
    else:
        seg_folder = os.path.join(output_seg_folder, "")
    os.makedirs(seg_folder, exist_ok=True)
    output_seg_file = os.path.join(seg_folder, filename + "_ensemble_seg.nii.gz")
    print(output_seg_file)

    sitk.WriteImage(
        sitk.Cast(sitk.GetImageFromArray((ensembled_prob > binary_threshold).astype('uint8')), sitk.sitkUInt8),
        output_seg_file,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv_path",
        type=csv_path,
        help="Input CSV path with column filename",
    )
    parser.add_argument(
        "yolov8_weights", type=file_path, help="Weights path for yolov8"
    )
    parser.add_argument(
        "nnunet_weights", type=file_path, help="Weights path for nnunet"
    )
    parser.add_argument(
        "output_seg_dir", type=str, help="output directory to save the images"
    )
    parser.add_argument(
        "--binary_mask_threshold", type=float, default=0.5, help="Binary mask threshold"
    )
    parser.add_argument(
        "output_csv_path",
        type=str,
        help="Output CSV path with column filename and ensemble_pred_tb_seg_file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_seg_dir):
        os.makedirs(args.output_seg_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inititalize YOLOv8 model
    yolov8_model = YOLO(args.yolov8_weights, verbose=False)
    yolov8_model.to(device)

    # Initialize nnUNet model and silence the progress bar to be consistent with YOLO behavior.
    predictor = nnUNetPredictor(device=torch.device(device), allow_tqdm=False)

    # Directory in nnunet/weights has a sub folder named fold_X where X is arbitrary. Here '0' is used.
    predictor.initialize_from_trained_model_folder(
        os.path.dirname(os.path.dirname(args.nnunet_weights)),
        checkpoint_name=os.path.basename(args.nnunet_weights),
        use_folds=(0,),
    )

    df = pd.read_csv(args.input_csv_path)

    for file in df["filename"].tolist():
        yolov8_prob_map = gen_yolov8_prob_map(file, yolov8_model)
        nnunet_prob_map = gen_nnunet_prob_map(file, predictor)

        gen_ensembled_yolov8_nnunet_segmentation(
            file,
            yolov8_prob_map,
            nnunet_prob_map,
            args.output_seg_dir,
            args.binary_mask_threshold,
        )

    df["ensemble_pred_tb_seg_file"] = df["filename"].apply(
        lambda x: os.path.join(
            args.output_seg_dir,
            os.path.splitext(os.path.basename(x))[0] + "_ensemble_seg.nii.gz",
        )
    )
    df.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    main()
