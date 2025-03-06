import numpy as np
def propagate(particles, frame_height, frame_width, params):
    # time / frames
    delta_t = 1 / 30
    sigma_position = params["sigma_position"]
    sigma_velocity = params["sigma_velocity"]
    noise_position = np.random.normal(0, sigma_position, size=(particles.shape[0], 2))
    noise_velocity = np.random.normal(0, sigma_velocity, size=(particles.shape[0], 2))
    #no motion
    if params["model"] == 0:
        A_matrix = np.array([[1, 0], [0, 1]])
        noise = noise_position
    #constant velocity
    else:
        A_matrix = np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])
        noise = np.hstack([noise_position, noise_velocity])
    particles_new = particles @ A_matrix.T + noise
    #position clip
    particles_new[:, 0:2] = np.clip(particles_new[:, 0:2], a_min=[0, 0], a_max=[frame_width, frame_height])
    return particles_new
