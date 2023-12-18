from os import PathLike

import cv2
import numpy as np
import scipy.fft as sfft
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_gradient_magnitude

import darkblur.calc as calc
import darkblur.pio as pio


def is_image_dark(mrc_path: PathLike, xml_path: PathLike = None,
                  intensity_threshold: float = 0.2,
                  dark_threshold: float = 50) -> bool:
    """
    Determines if an MRC format image is dark based on specified intensity thresholds.

    This function reads an image in MRC format, normalizes the pixel intensity values, and calculates the histogram
    of these normalized values. It then assesses the percentage of pixels that fall below a specified intensity
    threshold and determines if the image is dark based on this percentage exceeding a certain threshold.

    Parameters:
    mrc_path (Union[str, Path]): The file path to the MRC image.
    xml_path (Union[str, Path], optional): The file path to the corresponding XML file.
    intensity_threshold (float, optional): The threshold for pixel intensity, normalized between 0 and 1.
                                           Default is 0.2.
    dark_threshold (float, optional): The percentage threshold of low-intensity pixels used to classify
                                      an image as dark. Default is 50.

    NOTE:
            - The function assumes pixel intensity values are normalized between 0 and 1.
    Returns:
    bool: True if the image is considered dark, False otherwise.
    Example:
    is_dark = is_image_dark("/path/to/image.mrc")
    print(is_dark)
    """

    im_data = pio.read_mrc(mrc_path)
    normalised_im_data = calc.normalise(im_data=im_data)
    hist = np.histogram(normalised_im_data, bins=np.arange(0, 1.1, 0.1))[0]
    low_intensity_percentage = (np.sum(hist[:int(intensity_threshold * 10)]) / hist.sum()) * 100
    if low_intensity_percentage > dark_threshold:
        pio.modify_unselect_filter(xml_path, condition="True")
        return True
    return False


def is_image_blurry(mrc_path: PathLike, xml_path: PathLike = None, crop_size=256) -> bool:
    """
        Analyzes electron microscopy images to determine if an image is blurry and modifies an associated XML file if it is.

        This function processes electron microscopy images in MRC format. It applies a series of image processing techniques
        to each image to assess its sharpness. If an image is determined to be blurry (based on the ratio of the axes of
        the fitted ellipse to its largest contour), it marks the corresponding XML file to indicate that the image is blurry.

        Parameters:
        path_to_im (str): Path to the directory containing MRC files.
        path_to_xml (str, optional): Path to the directory containing XML files. Defaults to None.
        crop_size (int, optional): Size to which the Fourier-transformed image is cropped. Defaults to 256.

        Returns:
        Path: Returns the path of the MRC file if it is determined to be blurry.

        Notes:
        - The function assumes the presence of XML files corresponding to MRC files. If an MRC file is found to be blurry,
          the corresponding XML file is modified.
        - Blurriness determination is based on the elliptical fit of the largest contour in a processed version of the image.
          The ratio of the minor to major axis of this ellipse is compared against predetermined thresholds.
        - If the image is blurry, the function modifies the corresponding XML file and returns the path of the blurry MRC file.
        - The function processes all MRC files in the given directory.
        """
    t = 0.8
    T = 1.2

    m = pio.read_mrc(mrc_path)
    gradient_change_map = transform_im_data(m, crop_size=crop_size, threshold_for_peak_finding=0.1)
    edt = distance_transform_edt(gradient_change_map)
    threshold_value = np.percentile(edt, 95)
    _, binary_mask = cv2.threshold(edt, threshold_value, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, orientation) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        ratio = minor_axis / major_axis
        if not t <= ratio <= T:
            pio.modify_unselect_filter(xml_path, condition="True")
            return True
        return False


def transform_im_data(im_data, crop_size=256, threshold_for_peak_finding=0.1):
    ft = calc.normalise(np.abs(np.log(sfft.fftshift(sfft.fft2(im_data.data)))))
    rows, cols = ft.shape
    crow, ccol = rows // 2, cols // 2  # Center of the image

    # Cropping and histogram equalization
    f_shift_crop = ft[crow - crop_size:crow + crop_size, ccol - crop_size:ccol + crop_size]
    b = gaussian_gradient_magnitude(f_shift_crop, sigma=10)
    c = calc.normalise(b)
    thresholded_im_ft = np.where(c <= threshold_for_peak_finding, 0, c)
    return thresholded_im_ft


def match_image_fname_to_xml(im_name, xml_name):
    if im_name.stem in xml_name.stem:
        return xml_name
