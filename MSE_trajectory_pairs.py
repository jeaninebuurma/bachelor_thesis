import os
import numpy as np
import itertools
import random

from APReL.aprel.basics import TrajectorySet


def calculate_mse_per_step(trajectory1, trajectory2):
    num_steps = len(trajectory1)
    mse_per_step = []

    for step in range(num_steps):
        mse = np.mean((trajectory1[step][1] - trajectory2[step][1]) ** 2)
        mse_per_step.append(mse)

    average_mse = np.mean(mse_per_step)
    return average_mse


def rank_trajectory_pairs(trajectories):
    mse_matrix = np.zeros((len(trajectories), len(trajectories)))

    # Calculate MSE per step for each pair of trajectories based on actions
    for i, j in itertools.combinations(range(len(trajectories)), 2):
        mse = calculate_mse_per_step(trajectories[i], trajectories[j])
        mse_matrix[i, j] = mse
        mse_matrix[j, i] = mse

    ranked_indices = np.unravel_index(np.argsort(mse_matrix, axis=None), mse_matrix.shape)

    ranked_pairs = [(ranked_indices[0][k], ranked_indices[1][k]) for k in range(len(ranked_indices[0]))]

    mse_values = [mse_matrix[ranked_indices[0][k], ranked_indices[1][k]] for k in range(len(ranked_indices[0]))]

    return ranked_pairs, mse_values


def select_pairs(ranked_pairs, mse_values, sorting_order):
    """Selects every eighth pair from the ranked pairs list where the MSE is not zero and applies the given sorting order.

    Args:
        ranked_pairs: List of trajectory pairs ranked based on MSE values (from lowest to highest)
        mse_values: List of MSE values corresponding to the ranked trajectory pairs
        sorting_order: The sorting order to apply. One of 'ascending', 'descending', or 'random'.

    Returns:
        selected_pairs: List of selected pairs where every eighth pair has non-zero MSE and sorted based on the order.
    """
    selected_pairs = [(pair, mse) for pair, mse in zip(ranked_pairs, mse_values) if mse != 0]
    selected_pairs = selected_pairs[::8]  # Select every eighth pair

    if sorting_order == 'ascending':
        selected_pairs.sort(key=lambda x: x[1])  # Sort in ascending order based on MSE

    elif sorting_order == 'descending':
        selected_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order based on MSE

    elif sorting_order == 'random':
        random.shuffle(selected_pairs)  # Shuffle the selected pairs randomly

    # Print selected pairs and their MSE values
    print("Selected Pairs:")
    for i, (pair, mse) in enumerate(selected_pairs, start=1):
        print("{}: Pair {}, MSE {}".format(i, pair, mse))

    return selected_pairs


interaction_type = input("Please choose the interaction type ('high_five' or 'handshake'): ")
sorting_order = input("Please choose the sorting order: (ascending/descending/random): ")

data_folder = f"data_rec/{interaction_type}_rec"
state_filename = f"{interaction_type}_state_traj_rec.csv"
action_filename = f"{interaction_type}_action_traj_rec.csv"

rec_states = np.genfromtxt(os.path.join(data_folder, state_filename), delimiter=' ')
rec_actions = np.genfromtxt(os.path.join(data_folder, action_filename), delimiter=' ')

print("rec_states shape:", rec_states.shape)
print("rec_actions shape:", rec_actions.shape)


def load_trajectory_set(state_traj_path, action_traj_path, steps_per_traj=60):
    # first load np arrays from recorded trajectories
    state_trajs = state_traj_path
    action_trajs = action_traj_path

    total_trajs_num = int(np.shape(state_trajs)[0] / steps_per_traj)
    trajectory_set = []

    for i in range(total_trajs_num):
        traj = []
        states = state_trajs[i * steps_per_traj: ((i + 1) * steps_per_traj), :]
        actions = action_trajs[i * steps_per_traj: ((i + 1) * steps_per_traj), :]

        for step in range(steps_per_traj):
            state = states[step]
            action = actions[step]
            traj.append((state, action))

        trajectory = traj
        trajectory_set.append(trajectory)

    return trajectory_set


trajectories = load_trajectory_set(rec_states, rec_actions)


# Rank trajectory pairs based on MSE of actions
ranked_pairs, mse_values = rank_trajectory_pairs(trajectories)

# Select every eighth pair with non-zero MSE and apply the chosen sorting order
selected_pairs = select_pairs(ranked_pairs, mse_values, sorting_order)

# Create a new folder for the interaction type in data_order if it doesn't exist
os.makedirs(f"data_order/{interaction_type}", exist_ok=True)

state_filename = f"data_order/{interaction_type}/states_{sorting_order}_rec.csv"
action_filename = f"data_order/{interaction_type}/actions_{sorting_order}_rec.csv"

states = []
actions = []

for pair, _ in selected_pairs:
    trajectory_indices = pair

    for index in trajectory_indices:
        trajectory = trajectories[index]
        states.extend([step[0] for step in trajectory])
        actions.extend([step[1] for step in trajectory])

np.savetxt(state_filename, states, delimiter=' ', fmt='%f')
np.savetxt(action_filename, actions, delimiter=' ', fmt='%f')

# Print all trajectory pairs and their MSE values (excluding pairs with MSE=0)
print("\nAll Pairs:")
print("----------------------")
print("Pair\t\t\t\tMSE")
print("----------------------")
non_zero_pairs = [(pair, mse) for pair, mse in zip(ranked_pairs, mse_values) if mse != 0]
for pair, mse in non_zero_pairs:
    print("({}, {})\t\t{}".format(pair[0], pair[1], mse))
