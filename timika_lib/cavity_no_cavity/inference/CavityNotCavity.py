import torch
import monai
from tbportals_data_preparation.preprocess_input import _gen_preprocessed_lung_image
import numpy as np
from torch.nn import Module
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RepeatChanneld,
    Resized,
    ScaleIntensityd,
    NormalizeIntensityd,)
from torch.utils.data import DataLoader
import string
import random
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


class CavityNotCavity(object):
    def __init__(
        self,
        lung_seg_model: Module,
        device: torch.device,
        cavity_not_cavity_models: list = None,
        cavity_not_cavity_threshold: float = 0.5,
        lung_seg_model_input_size: tuple = (224, 224),
        cavity_not_cavity_model_input_size: tuple = (512, 512),
        seg_threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
            lung_seg_model(torch.nn.Module): UNet segmentation model path with weights containing to segment CXRs
            cavity_not_cavity_models(torch.nn.Module): List of Cavity/Not-Cavity models
            cavity_not_cavity_model_input_size (tuple): required output image size for building cavity/not-cavity model
            device(torch.device): Device to test the model on
            cavity_not_cavity_threshold: Threshold to binarize 1/0 labels representing Cavity/Not-Cavity
            lung_seg_model_input_size (tuple): size required to resample the input chest x ray
                                          files for input to pre-trained segmentation
                                          algorithm. Default is based on the size used
                                          for the training set.
            gaussian_blur(scalar or tuple with image dimension length): If given,
                   blur the image with a Gaussian with the given standard deviation(s)
                   before resampling.
            seg_threshold(float): Threshold to binarize masks.
        """
        self.lung_seg_model = lung_seg_model
        self.cavity_not_cavity_models = cavity_not_cavity_models
        self.device = device
        self.lung_seg_model_input_size = lung_seg_model_input_size
        self.cavity_not_cavity_model_input_size = cavity_not_cavity_model_input_size
        self.seg_threshold = seg_threshold
        self.cavity_not_cavity_threshold = cavity_not_cavity_threshold

    def _predict_single_image_for_cavity_not_cavity(self, temp_filename):
        """
        Predictions from the numpy test data using the Cavity/Not-Cavity model

        Parameters
        ----------

        cavity_not_cavity_model (torch.nn.Module): Cavity/Not-Cavity model.
        X_test (numpy array): Numpy test array to be used for model prediction
        device(torch.device): Device to test the model on

        Returns
        -------
        y_pred (numpy array): probability for Cavity
        """
        print(self.cavity_not_cavity_model_input_size)
        pre_transforms = [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Resized(
                keys=["img"],
                spatial_size=self.cavity_not_cavity_model_input_size,
                mode="bilinear",
            ),
            RepeatChanneld(keys=["img"], repeats=3),
            ScaleIntensityd(keys=["img"], minv=0, maxv=1),
            NormalizeIntensityd(
                keys=["img"],
                subtrahend=[0.485, 0.456, 0.406],
                divisor=[0.229, 0.224, 0.225],
                channel_wise=True,
            ),
        ]
        transforms = Compose(pre_transforms)
        file = [{"img": temp_filename}]
        train_ds = monai.data.Dataset(data=file, transform=transforms)
        loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),)
        data = next(iter(loader))
        X_test = data["img"].to(self.device)
        softmax = torch.nn.Softmax(dim=1)
        predictions = []
        with torch.no_grad():
            for model in self.cavity_not_cavity_models:
                y_pred = softmax(model(X_test))
            predictions.append(y_pred[:, 1])

        return torch.mean(torch.stack(predictions), dim=0).cpu().numpy()[0]

    def generate_random_string(self):
        characters = string.ascii_letters + string.digits
        return "".join(random.choice(characters) for _ in range(8))

    def predict_cavity(self, temp_filename):
        """
        This function preprocesses the simpleitk image and predicts on the
        preprocessed array using the loaded models.
        Parameters
        ----------
        sitk_img SimpleITK.Image) : Original SimpleITK Image.
        Returns
        -------
        probability_for_cavity(float) : Probability for Cavity.
        """
        # Preprocess the Image
        preprocessed_img = _gen_preprocessed_lung_image(
            temp_filename,
            self.lung_seg_model,
            self.device,
            self.lung_seg_model_input_size,
            self.seg_threshold,)

        temp_filename = self.generate_random_string()
        temp_filename = "../output/" + temp_filename + "_cropped.nrrd"

        sitk.WriteImage(preprocessed_img, temp_filename)

        # Predict the image.
        probability_for_cavity = self._predict_single_image_for_cavity_not_cavity(
            temp_filename
        )

        # Binarize cavity label
        predicted_label_for_cavity_no_cavity = (
            probability_for_cavity > self.cavity_not_cavity_threshold
        )
        return float(
            probability_for_cavity
        ), predicted_label_for_cavity_no_cavity.astype(np.uint8)
