import stim
import numpy as np
from typing import List, Tuple
import scipy

def get_decode_info_from_circuit(
        circ, decompose_errors=False):
    dem = circ.detector_error_model(
        decompose_errors=decompose_errors,
        approximate_disjoint_errors=True,
        flatten_loops=True, allow_gauge_detectors=True)
    det_mat, obs_mat, w, gauge_logical_index = dem_to_matrix(
        dem, gauge_label=[-1,-1,-1])
    return det_mat.toarray(), obs_mat.toarray().flatten(), w, gauge_logical_index

def dem_to_matrix(dem: stim.DetectorErrorModel,
                  gauge_label: List[int] = [-1, -1, -1],
                  return_weight_type: str = 'lin'
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a detector error model to a detector matrix, observable
        matrix, weight array, and a full matrix containing all information. Assumes that loops only have one level. Allows both 
    Args:
        dem: detector error model.
        gauge_label: detector index for logical observables that are written as
            gauge detectors here. Here, we assume all logical observables are
            specified in this format. We also assume that they appear at the
            end.
        return_weight_type: 'lin' returns the error probabilities directly,
            while 'log' returns the log likelihood log(p/(1-p))
    Returns:
        A tuple of four elements:
            1. A numpy array of shape (num_errors, num_detectors) where each
                row is a detector error event.
            2. A numpy array of shape (num_errors, num_observables) where each
                row is an observable error event.
            3. A numpy array of shape (num_errors,) where each element is the
                weight of the corresponding error.
            4. A numpy array containing the indices of the gauge logicals in
                the detector error model.
        Here, num_errors always removes error events with 0 probability.
    """ 
    gauge_logical_indices = []
    
    # previously lil matrix
    parsed_det_matrix = scipy.sparse.csc_matrix(
        (dem.num_errors, dem.num_detectors), dtype=bool)
    parsed_obs_matrix = scipy.sparse.csc_matrix(
        (dem.num_errors, dem.num_observables), dtype=bool)
    weights = np.zeros(dem.num_errors)
    
    detector_shift = 0 # keep track of cumulative detector shift
    err_count = 0 # index of current error event
    # list of error lines to remove due to argument being 0
    err_lines_keep = []
    for line in dem:
        if line.type == 'error':
            probability = line.args_copy()[0]
            if probability != 0:
                err_lines_keep.append(err_count)
                if return_weight_type == 'lin':
                    weights[err_count] = probability
                else:
                    weights[err_count] = np.log(probability / (1 - probability))
                for detector in line.targets_copy():
                    if stim.DemTarget.is_relative_detector_id(detector):
                        parsed_det_matrix[err_count, detector.val + detector_shift] = True
                    elif stim.DemTarget.is_logical_observable_id(detector):
                        parsed_obs_matrix[err_count, detector.val] = True
            err_count += 1
        if (line.type == 'detector' and
                line.args_copy()[:len(gauge_label)] == gauge_label):
            gauge_logical_indices.append(
                line.targets_copy()[0].val + detector_shift)
        if line.type == 'shift_detectors':
            detector_shift += line.targets_copy()[0]
        if line.type == 'repeat':
            # unroll inner loop body, assume there are no nested loops
            raise ValueError('Loops are not supported, please use flatten_loops when constructing the dem!')
    # Remove rows with 0 probability
    parsed_det_matrix = parsed_det_matrix[np.array(err_lines_keep), :]
    parsed_obs_matrix = parsed_obs_matrix[np.array(err_lines_keep), :]
    weights = weights[np.array(err_lines_keep)]
    
    det_mat = parsed_det_matrix[:, np.array(
        [i for i in range(dem.num_detectors)
         if i not in gauge_logical_indices])]
    obs_mat = scipy.sparse.hstack([parsed_obs_matrix,
                         parsed_det_matrix[:, np.array(gauge_logical_indices)]])
    return det_mat, obs_mat, np.array(weights), np.array(gauge_logical_indices)