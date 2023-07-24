import numpy as np
import os
import pandas as pd
from gym import spaces
from qibullet import SimulationManager
import pybullet as p
from time import sleep
from stable_baselines3 import PPO

from greet_env import GreetEnv

def calculate_mse_error(learner_action, teacher_action):
    error = np.linalg.norm(learner_action - teacher_action)
    return error

def run_test_trial(greeting_type, participant_number, order_type):
    greet_env = GreetEnv(using_gui=False,
                         auto_step=True,
                         greeting_type=greeting_type,
                         participant_number=participant_number,
                         order_type=order_type)

    model_file = f"models/ppo_{participant_number}_{greeting_type}_{order_type}.zip"
    if not os.path.exists(model_file):
        greet_env.close()
        print(f"Data files for participant do not exist. Skipping...")
        return
    policy = PPO.load(f"models/ppo_{participant_number}_{greeting_type}_{order_type}", env=greet_env)

    path_to_natural_state_traj = f"data_participant_{participant_number}/{greeting_type}_rec/{greeting_type}_state_traj_rec.csv"
    path_to_natural_action_traj = f"data_participant_{participant_number}/{greeting_type}_rec/{greeting_type}_action_traj_rec.csv"

    if not os.path.exists(path_to_natural_state_traj) or not os.path.exists(path_to_natural_action_traj):
        greet_env.close()
        print(f"Data files for participant do not exist. Skipping...")
        return

    natural_state_traj = np.genfromtxt(path_to_natural_state_traj)
    natural_action_traj = np.genfromtxt(path_to_natural_action_traj)

    total_steps_per_episode = 60
    total_test_episodes = 1

    # Load previously saved MSE results or create an empty array if it's the first test
    data_folder = "results_mse"
    state_filename = f"{greeting_type}_mse.csv"
    previous_mse_results = np.array([])
    if os.path.exists(os.path.join(data_folder, state_filename)):
        previous_mse_results = np.genfromtxt(os.path.join(data_folder, state_filename), delimiter=' ')

    mse_data = []  # List to store MSE results for all trials

    for episode in range(total_test_episodes):
        step = 0

        state = greet_env.reset()
        natural_state = natural_state_traj[step, :]
        sleep(1.0)
        done = False
        total_mse_error = 0.0

        for step in range(total_steps_per_episode):
            learner_action, _ = policy.predict(natural_state, deterministic=True)
            teacher_action = natural_action_traj[step, :]
            mse_error = calculate_mse_error(learner_action, teacher_action)
            total_mse_error += mse_error
            state, reward, done, _ = greet_env.step(action=learner_action)
            print("[Test trial: {}]: step {} finished".format(episode + 1, step + 1))
            natural_state = natural_state_traj[step, :]

            sleep(0.01)

        mean_mse_error = total_mse_error / total_steps_per_episode
        print(mean_mse_error)

        mse_data.append(mean_mse_error)  # Append the MSE result for this trial to the data list

        # Convert mean_mse_error to a 1D array
        mean_mse_array = np.atleast_1d(mean_mse_error)
        print(previous_mse_results)

        # Concatenate or stack the new MSE result with the previous ones
        if previous_mse_results.size == 0:
            updated_mse_results = np.expand_dims(mean_mse_array, axis=0)
        else:
            updated_mse_results = np.append(previous_mse_results, mean_mse_array)

        # Save the updated MSE results
        os.makedirs(data_folder, exist_ok=True)
        np.savetxt(os.path.join(data_folder, state_filename), updated_mse_results, delimiter=' ')

    greet_env.close()
    print('All test trials finished')

    teaching_score = input(f"What was the teaching score for participant {participant_number} with the order {order_type} for the greeting {greeting_type}?: ")
    # Create a new DataFrame for the participant's information
    df_info = pd.DataFrame({
        "Participant_number": [participant_number],
        "Order_type": [order_type],
        "Teaching_score": [teaching_score],
        "Mean_MSE": [mean_mse_error]
    })

    info_filename = f'{greeting_type}_after_training_info.csv'
    if os.path.exists(os.path.join(data_folder, info_filename)):
        # If the participant file exists, read it
        df_participant = pd.read_csv(os.path.join(data_folder, info_filename))
        # Add the new participant's information to the DataFrame
        df_participant = pd.concat([df_participant, df_info], ignore_index=True)
    else:
        df_participant = df_info

    # Save the participant information to a CSV file
    df_participant.to_csv(os.path.join(data_folder, info_filename), index=False)

    print('Participant information saved.')

    # Example of how to analyze the data using mixed ANOVA with Pingouin
    # import pingouin as pg
    # aov = pg.mixed_anova(data=df, dv='MSE', within='Episode', subject='Trial', correction=True)
    # print(aov)

if __name__ == '__main__':
    greeting_types = ['handshake', 'high_five']
    participant_numbers = list(range(1, 11))
    order_types = ['ascending', 'descending', 'random']

    for greeting_type in greeting_types:
        for participant_number in participant_numbers:
            for order_type in order_types:
                run_test_trial(greeting_type, participant_number, order_type)
