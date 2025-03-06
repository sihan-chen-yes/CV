import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost
import math
def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    '''
    hist:initial histograms
    hist_bin:number of bins
    '''
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    num = particles.shape[0]
    particles_w = np.zeros((num, 1))
    for i in range(num):
        center_x = particles[i, 0]
        center_y = particles[i, 1]
        x_min = round(center_x - bbox_width / 2)
        x_max = round(center_x + bbox_width / 2)
        y_min = round(center_y - bbox_height / 2)
        y_max = round(center_y + bbox_height / 2)
        #clip to the frame size
        x_min, x_max, y_min, y_max = np.clip([x_min, x_max, y_min, y_max], a_min=[0, 0, 0, 0], a_max=[frame_width, frame_width, frame_height, frame_height])
        hist_cur = color_histogram(x_min, y_min, x_max, y_max, frame, hist_bin)
        chi_squared_distance = chi2_cost(hist_cur, hist)
        #Chi-Square distance
        particles_w[i] = 1. / (math.sqrt(2. * np.pi) * sigma_observe) * np.exp(-0.5 * np.square(chi_squared_distance / sigma_observe))

    #normalization
    #prevent 0 divisor
    eps = 1e-60
    particles_w /= (np.sum(particles_w) + eps)
    return particles_w
