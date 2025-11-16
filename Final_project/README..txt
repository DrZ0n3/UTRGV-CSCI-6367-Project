CSCI 6367 Final Project – Problem I: Lung CT Image Processing

Overview
This project performs a series of image processing tasks on a grayscale lung CT image using Python. The tasks include frequency analysis using FFT,
image downsampling, interpolation in both frequency and spatial domains, and error analysis.

Files
lung_ct.jpg: Input grayscale lung CT image.

main.py: Python script containing all processing steps.


Requirements
Python 3.x

Libraries:
numpy
Pillow
matplotlib
scipy

Install dependencies using:

bash
pip install numpy pillow matplotlib scipy


Image dimensions (M × N) are printed.

Displayed using matplotlib.

2. 2D FFT and Spectrum Visualization
2D Fast Fourier Transform (FFT) is computed.

Magnitude and phase spectra are displayed:

Before and after applying fftshift.

Magnitude is log-scaled for better visibility.

3. Downsampling
The image is downsampled to half its size by discarding odd rows and columns.

Both original and downsampled images are displayed.

4. Frequency-Domain Interpolation
FFT of the downsampled image is computed.

Zero-padding is applied in the frequency domain to restore original size.

Inverse FFT is used to reconstruct the image.

Conjugate symmetry is preserved to ensure a real-valued result.

5. Spatial-Domain Interpolation
Bilinear interpolation is performed using scipy.ndimage.zoom.

The result is cropped to match the original image size.

6. Error Analysis
Mean Squared Error (MSE) is computed between:

Original vs. frequency-domain interpolated image.

Original vs. spatial-domain interpolated image.

Interpolated images are rescaled to match the original intensity range before comparison.

Output
Visualizations:

Original image

FFT magnitude and phase (shifted and unshifted)

Downsampled image

Frequency-domain and spatial-domain interpolated images

Printed MSE values for both interpolation methods.