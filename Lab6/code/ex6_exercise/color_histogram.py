import numpy as np
def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    '''
    hist_bin:number of bins
    '''
    #get the region of interest
    roi = frame[ymin:ymax + 1, xmin:xmax + 1]
    bin_size = (255 + 1) // hist_bin
    r_hist = np.zeros(hist_bin)
    g_hist = np.zeros(hist_bin)
    b_hist = np.zeros(hist_bin)
    #put the pixel into bins
    for y in range(roi.shape[0]):
        for x in range(roi.shape[1]):
            r_pixel, g_pixel, b_pixel = roi[y, x]
            r_hist[r_pixel // bin_size] += 1
            g_hist[g_pixel // bin_size] += 1
            b_hist[b_pixel // bin_size] += 1
    rgb_hist = np.vstack([r_hist, g_hist, b_hist])
    #normalization
    total = np.sum(rgb_hist)
    #prevent 0 divisor
    assert total != 0
    rgb_hist /= total
    return rgb_hist
