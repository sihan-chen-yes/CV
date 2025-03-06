import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the second image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    # distance calculation
    desc1 = np.expand_dims(desc1, axis=1)
    desc2 = np.expand_dims(desc2, axis=0)
    distances = ((desc1 - desc2)**2).sum(axis=2)
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        indices_y = np.argmin(distances, axis=1)
        indices_x = np.arange(q1)
        matches = np.stack((indices_x, indices_y), axis=-1)
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        indices_y = np.argmin(distances, axis=1)
        indices_x = np.arange(q1)
        #matches of q1 to q2
        matches = np.stack((indices_x, indices_y), axis=-1)
        #q1 to q2 values
        q1_to_q2_values = np.min(distances, axis=1)
        q2_to_q1_values = np.min(distances, axis=0)
        # to check if the value is the minimum
        mutual_indices = q1_to_q2_values == q2_to_q1_values[matches[:, 1]]
        matches = matches[mutual_indices]
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        indices_y = np.argmin(distances, axis=1)
        indices_x = np.arange(q1)
        matches = np.stack((indices_x, indices_y), axis=-1)
        # find the smallest two
        smallest = np.partition(distances, 2, axis=1)
        ratio_indices = (smallest[:, 0] / smallest[:, 1]) < ratio_thresh
        matches = matches[ratio_indices]
    else:
        raise NotImplementedError
    return matches

