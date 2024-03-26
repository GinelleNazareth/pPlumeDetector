import numpy as np
from skimage import measure
cimport numpy as cnp

cnp.import_array()

DTYPE = np.uint8

ctypedef cnp.uint8_t DTYPE_t

def block_cluster_fn(cnp.ndarray[DTYPE_t, ndim=2] seg_img, int window_width_pixels, int row_padding, int col_padding, cluster_min_pixels):
    window_rows = window_width_pixels
    window_cols = window_width_pixels

   # Initialize image matrices
    cdef int rows = seg_img.shape[0]
    cdef int cols = seg_img.shape[1]
    cdef cnp.ndarray clustered_cores_img = np.zeros((rows, cols), dtype=DTYPE)
    cdef cnp.ndarray cluster_regions_img = np.zeros((rows, cols), dtype=DTYPE)
    cdef cnp.ndarray labelled_clustered_img = np.zeros((rows, cols), dtype=DTYPE)
    cdef cnp.ndarray window_ones = np.ones((int(window_rows), int(window_cols)), dtype=DTYPE)

    cdef int start_row, end_row, start_col, end_col

    # Create clustered_cores_img, identifying pixels at the center of high density windows.
    # Also create cluster_regions_img, identifying high density regions. Note: output matrices are zero padded
    for row in range(row_padding, rows-row_padding, 1):
        for col in range(col_padding, cols-col_padding, 1):
            if seg_img[row, col]:
                start_row = row - row_padding
                end_row   = row + row_padding + 1
                start_col = col - col_padding
                end_col   = col + col_padding + 1
                filled = (seg_img[start_row:end_row, start_col:end_col]).sum()
                if filled > cluster_min_pixels:
                    clustered_cores_img[row,col] = 1
                    cluster_regions_img[start_row:end_row, start_col:end_col] = window_ones

    # Identify and label separate regions
    labelled_regions_img, num_clusters = measure.label(cluster_regions_img, return_num=True, connectivity=2)

    # Mask input image with the labelled regions image to create the labelled clustered image
    labelled_clustered_img = labelled_regions_img * seg_img

    return clustered_cores_img, cluster_regions_img, labelled_regions_img, labelled_clustered_img, num_clusters


