import numpy as np
import cv2
import matplotlib.pyplot as plt


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Mapping from orientation to derivative orders
    grad_dir = {'x': (1, 0), 'y': (0, 1)}

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator
    try:
        sobel = cv2.Sobel(gray, cv2.CV_64F, *grad_dir[orient], ksize=sobel_kernel)
    except:
        raise ValueError("orient must be 'x' or 'y'")

    # Take the absolute value of the derivative
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create mask
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel < thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the magnitude
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    # Scale to 8-bit (0 - 255)
    scaled = np.uint8(255 * grad_mag / np.max(grad_mag))

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= thresh[0]) & (scaled < thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    abs_x = np.absolute(grad_x)
    abs_y = np.absolute(grad_y)

    # Use np.arctan2(abs_y, abs_x) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_y, abs_x)

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir).astype(np.uint8)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir < thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output


if __name__ == "__main__":
    image = cv2.imread("./test_images/test4.jpg")

    ksize = 3

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 130))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 130))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(30, 100))
    dir_binary = dir_thresh(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    plt.imshow(combined, cmap="gray")
    plt.show()
