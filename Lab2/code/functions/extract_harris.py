import cv2
import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    # gradient kernel
    x_grad_kernel = np.array([[1, 0, -1]]) / 2
    y_grad_kernel = np.array([[1], [0], [-1]]) / 2

    # gradient calculation
    gradient_Ix = signal.convolve2d(img, x_grad_kernel, mode='same', boundary='symm')
    gradient_Iy = signal.convolve2d(img, y_grad_kernel, mode='same', boundary='symm')
    gradient_Ixy = gradient_Ix * gradient_Iy
    gradient_Ix_square = gradient_Ix**2
    gradient_Iy_square = gradient_Iy**2

    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    blurred_gradient_Ix_square = cv2.GaussianBlur(gradient_Ix_square, ksize=[3, 3], sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    blurred_gradient_Iy_square = cv2.GaussianBlur(gradient_Iy_square, ksize=[3, 3], sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    blurred_gradient_Ixy = cv2.GaussianBlur(gradient_Ixy, ksize=[3, 3], sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    # 2D gaussian kernel
    window_func_1D = cv2.getGaussianKernel(ksize=3, sigma=sigma)
    window_func_2D = window_func_1D * window_func_1D.transpose()
    # sum to build matrix "M"
    M_Ix_square = signal.convolve2d(blurred_gradient_Ix_square, window_func_2D, mode='same', boundary='symm')
    M_Iy_square = signal.convolve2d(blurred_gradient_Iy_square, window_func_2D, mode='same', boundary='symm')
    M_Ixy = signal.convolve2d(blurred_gradient_Ixy, window_func_2D, mode='same', boundary='symm')

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    det_M = M_Ix_square * M_Iy_square - M_Ixy**2
    trace_M = M_Ix_square + M_Iy_square
    C = det_M - k * trace_M**2

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    max_filtered_C = ndimage.maximum_filter(C, size=3, mode='reflect')
    #(element > thresh) & element is max
    coordinate_y, coordinate_x = np.where((C > thresh) & (C == max_filtered_C))

    corners = np.stack((coordinate_x, coordinate_y), axis=-1)

    return corners, C

