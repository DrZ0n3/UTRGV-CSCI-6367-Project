from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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