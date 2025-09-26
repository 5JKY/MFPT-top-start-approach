# Following the same logic as two-region method, this version can have comparable performance.
import numpy as np
import matplotlib.pyplot as plt


def simulate_AbInAb_regular(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    """
    b1: position of the lower absorbing boundary 
    b2: position of the upper absorbing boundary
    """
    np.random.seed(10)
    position_arr = init_point * np.ones(num_particles, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=8)
    init_id = np.where(n_arr == init_point)
    # count the number of particles fall on the left region amd right region seperately
    count_n_left = np.zeros(n_arr.size)
    count_n_right = np.zeros(n_arr.size)
    # count the number of particles fall on initial position, to obtain Pst(x0)
    count_n_left[init_id] = num_particles
    count_n_right[init_id] = num_particles
    # 2d array of first passage time: each raw is for one experiment, each column is for one accounted position
    ti_n = np.zeros((num_particles, n_arr.size))
    abort_ti_n = np.array([])

    # iteration numbers, can be converted to time
    left_iter_count_arr = np.zeros(num_particles)
    right_iter_count_arr = np.zeros(num_particles)
    # flag_escape_arr = np.zeros(num_particles)
    # Keep generating random steps until each particle explores the entire region
    while position_arr.size > 0:
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice(
            [-step_size, step_size], size=position_arr.size
            )
        # Calculate beta * delta_U (potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            # Find indices where the obtained position fall on the n_arr
            idx_n = np.where(n_arr == position_arr[i])
            if idx_n[0].size == 1:
                if position_arr[i] < init_point-step_size/2:
                    left_iter_count_arr[i] += 1
                    if ti_n[i, idx_n] == 0:
                        ti_n[i, idx_n] = left_iter_count_arr[i]
                    # only count for the particle hasn't visited left absorbing boundary
                    # if flag_escape_arr[i] > -1:
                    if ti_n[i, 0] == 0:
                        # count_n_left[0] will always equal to 0, since if idx_n=0, ti_n[i, 0]!=0
                        count_n_left[idx_n] += 1  
                elif position_arr[i] > init_point+step_size/2:
                    right_iter_count_arr[i] += 1
                    if ti_n[i, idx_n] == 0:
                        ti_n[i, idx_n] = right_iter_count_arr[i]
                    # only count for the particle hasn't visited right absorbing boundary
                    # if flag_escape_arr[i] < 1:
                    if ti_n[i, -1] == 0:
                        count_n_right[idx_n] += 1
                else:
                    left_iter_count_arr[i] += 1
                    right_iter_count_arr[i] += 1
                    # only count for the particle hasn't visited left absorbing boundary
                    # if flag_escape_arr[i] > -1:
                    if ti_n[i, 0] == 0:
                        count_n_left[idx_n] += 1
                    # only count for the particle hasn't visited right absorbing boundary
                    # if flag_escape_arr[i] < 1:
                    if ti_n[i, -1] == 0:
                        # count_n_right[-1] will always equal to 0, since if idx_n is the last state, ti_n[i, -1]!=0
                        count_n_right[idx_n] += 1
            
        # Use mask to find indices of walkers (reach the left absorbing boundar) & (need to be recycled)
        # left_return_mask = (position_arr < b1+step_size/4) & (flag_escape_arr<1)
        left_return_mask = (position_arr < b1+step_size/4) & (ti_n[:, -1] == 0)
        # flag_escape_arr[left_return_mask] -= 1
        # counting the contribution of left recycling step 
        position_arr[left_return_mask] = init_point
        left_iter_count_arr[left_return_mask] += 1
        right_iter_count_arr[left_return_mask] += 1
        # count only for the right region that haven't been fully explored
        count_n_right[init_id] += np.sum(left_return_mask)
        # Each simulation is aborted when the walker escape from left after visiting right absoring boundary
        indices = np.where(position_arr < b1+step_size/4)
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)
        # flag_escape_arr = np.delete(flag_escape_arr, indices)
        
        # Use mask to find indices of walkers (reach the right absorbing boundar) & (need to be recycled)
        # right_return_mask = (position_arr > b2-step_size/4) & (flag_escape_arr>-1)
        right_return_mask = (position_arr > b2-step_size/4) & (ti_n[:, 0] == 0)
        # flag_escape_arr[right_return_mask] += 1
        # Counting the contribution of right recycling step 
        position_arr[right_return_mask] = init_point
        left_iter_count_arr[right_return_mask] += 1
        right_iter_count_arr[right_return_mask] += 1
        # count only for the left region that haven't been fully explored
        count_n_left[init_id] += np.sum(right_return_mask)
        # Each simulation is aborted when the walker escape from right after visiting left absoring boundary
        indices = np.where(position_arr > b2-step_size/4)
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)
        # flag_escape_arr = np.delete(flag_escape_arr, indices)

    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n_left, count_n_right, abort_ti_n

