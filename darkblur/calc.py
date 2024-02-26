from pathlib import Path

import cv2
import numpy as np
import pylab as plt
from pio import read_mrc
from skimage.transform import hough_line, hough_line_peaks
from scipy.ndimage import gaussian_gradient_magnitude





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




def analyze_motion_blur_fft(image_path):
    image = read_mrc(image_path)
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # Load image and convert to grayscale

    # Apply FFT, shift the DC component to the center, and crop to 256x256
    f_transform = np.fft.fft2(image_normalized)
    f_shift = np.fft.fftshift(f_transform)
    center = np.array(f_shift.shape) // 2
    cropped_f_shift = f_shift[center[0] - 128:center[0] + 128, center[1] - 128:center[1] + 128]

    # Calculate the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(cropped_f_shift) + 1)  # Added 1 to avoid log(0)

    # Detect lines using Hough Transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    ggm = gaussian_gradient_magnitude(magnitude_spectrum, sigma=5)
    h, theta, d = hough_line(ggm, theta=tested_angles)
    _, angles, dists = hough_line_peaks(h, theta, d)


    # Visualize results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Cropped Magnitude Spectrum')
    for angle, dist in zip(angles, dists):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.subplot(1, 3, 2)
    plt.imshow(ggm, cmap='gray')
    plt.title('Gaussian Gradient Magnitude')

    plt.subplot(1, 3, 3)
    plt.plot(np.sum(magnitude_spectrum, axis=0), label='Horizontal Energy Profile')
    plt.plot(np.sum(magnitude_spectrum, axis=1), label='Vertical Energy Profile')
    plt.legend()
    plt.title('Energy Spread Estimation')

    plt.tight_layout()
    plt.show()

    # Interpretation of results needed here
    #print("Detected angles (radians):", angles)
    #print("Energy spread can be inferred from the profiles.")
#
# pth = Path("/Users/ps/data/wip/darkblur/images/dark/").glob("*.mrc")
# files = list(pth)
# f = files[0]
# print(f)
# analyze_motion_blur_fft(f)
# """
# # This is for the dark-blurry test
# pth = Path("/Users/ps/data/wip/darkblur/images/dark/").glob("*.mrc")
# files = list(pth)
# not_dark = [f for f in files if not is_frame_dark(f)]
# not_obstructed = [f for f in not_dark if not is_frame_obstructed(f)]
# total = len(files)
# dark_count = total - len(not_dark)
# obstructed_count = len(not_dark) - len(not_obstructed)
#
# # Printing the file names that are neither dark nor obstructed
# for f in not_obstructed:
#     print(f.name)
#
# # Print the results
# print(f"{dark_count}/{total} are dark")
# print(f"{obstructed_count}/{len(not_dark)} are obstructed")
