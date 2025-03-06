import numpy as np
def resample(particles, particles_w):
    #CDF
    cumulative_sum = np.cumsum(particles_w)
    cumulative_sum[-1] = 1.0
    #get the index randomly
    indices = np.searchsorted(cumulative_sum, np.random.rand(len(particles_w)))
    resampled_particles = particles[indices]
    #same weight after resampling
    resampled_weights = np.ones_like(particles_w) / len(particles_w)

    return resampled_particles, resampled_weights

