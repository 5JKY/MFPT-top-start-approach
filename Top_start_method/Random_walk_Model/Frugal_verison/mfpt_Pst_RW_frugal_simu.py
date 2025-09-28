# The frugal version of AbInAb_regular simulation, 
# the data from the completed trajectories on either said won't be wasted
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
    # slice require integer
    init_id = np.where(n_arr == init_point)[0][0]
    n1_arr = n_arr[ :init_id+1]
    n2_arr = n_arr[init_id: ]
    # count the number of particles fall on the left region and right region seperately
    # 2d array of histogram: each raw is for one experiment, each column is for one accounted position
    count_n1 = np.zeros((num_particles, n1_arr.size))
    count_n2 = np.zeros((num_particles, n2_arr.size))
    # count the number of particles fall on initial position, to obtain Pst(x0)
    count_n1[:, -1] += 1
    count_n2[:, 0] += 1
    # For each particle, the following list can store the historgram of distribution
    # obtained from trajectories explored the entire region either on the left or on the right
    count_list_n1 = [[] for _ in range(num_particles)]
    count_list_n2 = [[] for _ in range(num_particles)]
    # store the histogram (or averaged histogram) after exploring the entire region [b1, b2]
    abort_count_n1 = np.array([])
    abort_count_n2 = np.array([])
    # 2d array of first passage time: each raw is for one experiment, each column is for one accounted position
    fpt_n1 = np.zeros((num_particles, n1_arr.size))
    fpt_n2 = np.zeros((num_particles, n2_arr.size))
    # For each particle, the following list can store the fpt obtained from
    # trajectories explored the entire region either on the left or on the right
    fpt_list_n1 = [[] for _ in range(num_particles)]
    fpt_list_n2 = [[] for _ in range(num_particles)]
    # store the fpt (or averaged fpt) after exploring the entire region [b1, b2]
    abort_fpt_n1 = np.array([])
    abort_fpt_n2 = np.array([])
    # iteration numbers, can be converted to time
    left_iter_count_arr = np.zeros(num_particles)
    right_iter_count_arr = np.zeros(num_particles)

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
            if position_arr[i] < init_point-step_size/2:
                # Find indices where the obtained position fall on the n1_arr
                idx_n1 = np.where(n1_arr == position_arr[i])
                if idx_n1[0].size == 1:
                    left_iter_count_arr[i] += 1
                    count_n1[i, idx_n1] += 1
                    if fpt_n1[i, idx_n1] == 0:
                        fpt_n1[i, idx_n1] = left_iter_count_arr[i]
                if position_arr[i] < b1+step_size/4:
                    if len(count_list_n2[i]) == 0:  # the walker hasn't visited b2
                        position_arr[i] = init_point
                        count_list_n1[i].append(count_n1[i].copy())
                        fpt_list_n1[i].append(fpt_n1[i].copy())
                        count_n1[i] = 0
                        fpt_n1[i] = 0
                        left_iter_count_arr[i] = -1
                    else:  # the walker has visited b2
                        abort_count_n1 = np.append(abort_count_n1, count_n1[i])
                        abort_fpt_n1 = np.append(abort_fpt_n1, fpt_n1[i])
                        # collect the averaged data aborted on the right 
                        abort_count_n2 = np.append(abort_count_n2, np.mean(count_list_n2[i], axis=0))
                        abort_fpt_n2 = np.append(abort_fpt_n2, np.mean(fpt_list_n2[i], axis=0))
                        # abort_count_n2 = np.append(abort_count_n2, np.array(count_list_n2[i]))
                        # abort_fpt_n2 = np.append(abort_fpt_n2, np.array(fpt_list_n2[i]))
                        
            elif position_arr[i] > init_point+step_size/2:
                # Find indices where the obtained position fall on the n2_arr
                idx_n2 = np.where(n2_arr == position_arr[i])
                if idx_n2[0].size == 1:
                    right_iter_count_arr[i] += 1
                    count_n2[i, idx_n2] += 1
                    if fpt_n2[i, idx_n2] == 0:
                        fpt_n2[i, idx_n2] = right_iter_count_arr[i]
                if position_arr[i] > b2-step_size/4:
                    if len(count_list_n1[i]) == 0:  # the walker hasn't visited b1
                        position_arr[i] = init_point
                        count_list_n2[i].append(count_n2[i].copy())
                        fpt_list_n2[i].append(fpt_n2[i].copy())
                        count_n2[i] = 0
                        fpt_n2[i] = 0
                        right_iter_count_arr[i] = -1
                    else:  # the walker has visited b1
                        abort_count_n2 = np.append(abort_count_n2, count_n2[i])
                        abort_fpt_n2 = np.append(abort_fpt_n2, fpt_n2[i])
                        # collect the averaged data aborted on the left
                        abort_count_n1 = np.append(abort_count_n1, np.mean(count_list_n1[i], axis=0))
                        abort_fpt_n1 = np.append(abort_fpt_n1, np.mean(fpt_list_n1[i], axis=0))
                        # abort_count_n1 = np.append(abort_count_n1, np.array(count_list_n1[i]))
                        # abort_fpt_n1 = np.append(abort_fpt_n1, np.array(fpt_list_n1[i]))
                            
            if position_arr[i] == init_point:
                left_iter_count_arr[i] += 1
                count_n1[i, -1] += 1
                right_iter_count_arr[i] += 1
                count_n2[i, 0] += 1
    
        # Simulations are aborted when the walkers had visited both absorbing boundaries
        indices = np.where((position_arr < b1+step_size/4)|(position_arr > b2-step_size/4))[0]
        position_arr = np.delete(position_arr, indices)
        count_n1 = np.delete(count_n1, indices, axis=0)
        fpt_n1 = np.delete(fpt_n1, indices, axis=0)
        count_n2 = np.delete(count_n2, indices, axis=0)
        fpt_n2 = np.delete(fpt_n2, indices, axis=0)
        # Sort indices in descending order to avoid index shifting
        indices = sorted(indices, reverse=True)
        for idx in indices:
            count_list_n1.pop(idx)
            fpt_list_n1.pop(idx)
            count_list_n2.pop(idx)
            fpt_list_n2.pop(idx)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)

    abort_count_n1 = abort_count_n1.reshape((num_particles, n1_arr.size))
    abort_fpt_n1 = abort_fpt_n1.reshape((num_particles, n1_arr.size))
    abort_count_n2 = abort_count_n2.reshape((num_particles, n2_arr.size))
    abort_fpt_n2 = abort_fpt_n2.reshape((num_particles, n2_arr.size))
    # abort_count_n1 = abort_count_n1.reshape((-1, n1_arr.size))
    # abort_fpt_n1 = abort_fpt_n1.reshape((-1, n1_arr.size))
    # abort_count_n2 = abort_count_n2.reshape((-1, n2_arr.size))
    # abort_fpt_n2 = abort_fpt_n2.reshape((-1, n2_arr.size))

    return abort_count_n1, abort_count_n2, abort_fpt_n1, abort_fpt_n2

