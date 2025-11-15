from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# Open and convert to grayscale
img = Image.open("lung_ct.jpg").convert("L")

# Resize to MxN
# M, N = 200, 300

M, N = img.size
# showI = image.resize((N, M))  # Note: PIL takes (width, height)

print(f"Image size: {M} x {N} ")

# --- Compute 2D FFT ---
f = np.fft.fft2(img)            # 2D FFT
fshift = np.fft.fftshift(f)     # Shift low freq to center

# --- Magnitude and Phase ---
magnitude = np.abs(fshift)             # magnitude
phase = np.angle(fshift)               # phase

# For visualization: use log scale on magnitude
magnitude_spectrum = 20 * np.log(magnitude + 1)

# --- Display ---
plt.figure(figsize=(12,4))

# Original
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Magnitude
plt.subplot(1,3,2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis("off")

# Phase
plt.subplot(1,3,3)
plt.imshow(phase, cmap='gray')
plt.title("Phase Spectrum")
plt.axis("off")

plt.show()



# Down-sample by taking every 2nd pixel (discard odd rows/cols)
img_np = np.array(img)
downsampled = img_np[::2, ::2]

# Display
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(downsampled, cmap='gray')
plt.title("Down-sampled Image (M/2 × N/2)")
plt.axis("off")

plt.show()

# Show image
img.show()



# -----------------------------
# (4) Frequency-domain interpolation (zero-padding)
# -----------------------------
# Compute FFT of downsampled image
Fd = np.fft.fft2(downsampled)
Fd_shift = np.fft.fftshift(Fd)

# Prepare zero-padded array of original size
M_full, N_full = img_np.shape
M2, N2 = downsampled.shape
Fpad_shift = np.zeros((M_full, N_full), dtype=complex)

# Compute where to insert small FFT into center of big one
start_r = M_full//2 - M2//2
start_c = N_full//2 - N2//2
Fpad_shift[start_r:start_r+M2, start_c:start_c+N2] = Fd_shift

# Inverse shift and inverse FFT to get interpolated image
Fpad = np.fft.ifftshift(Fpad_shift)
img_freq_interp = np.fft.ifft2(Fpad)
img_freq_interp = np.real(img_freq_interp)

# -----------------------------
# (5) Spatial-domain interpolation (bilinear)
# -----------------------------
# Upsample using linear interpolation
img_spatial_interp = ndimage.zoom(downsampled, (2, 2), order=1)
img_spatial_interp = img_spatial_interp[:M_full, :N_full]  # crop if needed

# -----------------------------
# (6) Compute mean squared errors (MSE)
# -----------------------------
def mse(a, b):
    return np.mean((a.astype(float) - b.astype(float)) ** 2)

# Normalize both interpolations to match original range
def rescale(recon, ref):
    rmin, rmax = recon.min(), recon.max()
    omin, omax = ref.min(), ref.max()
    return (recon - rmin) / (rmax - rmin + 1e-9) * (omax - omin) + omin

img_freq_interp_rescaled = rescale(img_freq_interp, img_np)
img_spatial_interp_rescaled = rescale(img_spatial_interp, img_np)

mse_freq = mse(img_np, img_freq_interp_rescaled)
mse_spatial = mse(img_np, img_spatial_interp_rescaled)

print(f"MSE (Frequency-domain interpolation): {mse_freq:.4f}")
print(f"MSE (Spatial linear interpolation): {mse_spatial:.4f}")

# Display interpolated images
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img_np, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_freq_interp_rescaled, cmap='gray')
plt.title("Freq-domain interpolation")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_spatial_interp_rescaled, cmap='gray')
plt.title("Spatial linear interpolation")
plt.axis("off")
plt.show()