from os import PathLike

import cv2
import numpy as np
import scipy.fft as sfft
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_gradient_magnitude

import darkblur.calc as calc
import darkblur.pio as pio


def is_image_dark(mrc_path: PathLike, darkness_threshold: float = 0.65) -> bool:
    img_data = pio.read_mrc(mrc_path)
    norm = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    hist = cv2.calcHist([norm], [0], None, [256], [0, 256]).flatten()
    dark_threshold_bins = 20
    dark_pixels = sum(hist[:dark_threshold_bins])
    total_pixels = img_data.size
    if (dark_pixels / total_pixels) > darkness_threshold:
        return True
    return False


def is_image_obstructed(mrc_path: PathLike, coverage_percentage_threshold: float = 0.15) -> bool:
    image = pio.read_mrc(mrc_path)
    mask, largest_contour = detect_obstructions(image)
    # Calculate the area of the largest black region
    largest_area = cv2.contourArea(largest_contour)
    # Calculate the total image area
    total_area = image.shape[0] * image.shape[1]
    # Calculate the percentage of the image that the largest black region covers
    coverage_percentage = largest_area / total_area
    if coverage_percentage > coverage_percentage_threshold:
        return True
    return False


def is_image_blurry(mrc_path: PathLike, crop_size=256) -> bool:
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
    t = 0.85
    T = 1.10

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
            #pio.modify_unselect_filter(xml_path, condition="True")
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


def detect_obstructions(image):
    # Normalize the image to 8-bit
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # Apply a Gaussian blur
    image_blur = cv2.GaussianBlur(image_normalized, (5, 5), sigmaX=5, sigmaY=5)
    # Apply OTSU thresholding to get the binary image
    _, binary_image = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological closing to close small holes or gaps
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area - keeping the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask for the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask, largest_contour
