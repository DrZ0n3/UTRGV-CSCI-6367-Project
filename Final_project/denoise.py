# PROBLEM 2: Denoising and Edge Detection
# In this problem, you will denoise a noisy image using a median filter
# and then perform edge detection on the denoised image using the Canny edge detector.
import cv2
import matplotlib.pyplot as plt

# --- Step 1: Load the noisy image ---
# Load the image in grayscale
img_noisy = cv2.imread('noisy-img.jpg', cv2.IMREAD_GRAYSCALE)

if img_noisy is None:
    print("Error: Could not read image. Make sure 'noisy-img.jpg' is in the same directory.")
else:
    # --- Step 2: Apply a Median Filter to denoise ---
    # We use a 5x5 kernel. A 3x3 might be too small, and 7x7 might be too blurry.
    # 5x5 is a good balance for this level of noise.
    img_denoised = cv2.medianBlur(img_noisy, 5)

    # --- Step 3: Perform edge detection on the denoised image ---
    # We use the Canny edge detector.
    # The 100 and 200 are the minVal and maxVal for hysteresis thresholding.
    img_edges = cv2.Canny(img_denoised, 100, 200)

    # --- Step 4: Display and compare the images ---
    plt.figure(figsize=(15, 5))

    # Plot 1: Original Noisy Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_noisy, cmap='gray')
    plt.title('Original Noisy Image')
    plt.axis('off')

    # Plot 2: Denoised Image
    plt.subplot(1, 3, 2)
    plt.imshow(img_denoised, cmap='gray')
    plt.title('Denoised (Median Filter)')
    plt.axis('off')

    # Plot 3: Edge-Detected Image
    plt.subplot(1, 3, 3)
    plt.imshow(img_edges, cmap='gray')
    plt.title('Edge Detection (Canny)')
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()