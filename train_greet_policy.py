import numpy as np
from datetime import datetime
from stable_baselines3 import PPO

from greet_env import GreetEnv

# used to log when the training starts to distinguish models
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def main():
    # create an instance of greeting environment
    # recommend to set using_gui to False and auto_step to True to save training time
    greeting_type = input("Enter the greeting type (handshake or high_five): ")
    participant_number = input("Please enter the participant number: ")
    order_type = input("Please choose the order type ('ascending', 'descending', or 'random'): ")
    greet_env = GreetEnv(using_gui=False,
                         auto_step=True,
                         greeting_type=greeting_type,
                         participant_number=participant_number,
                         order_type=order_type)

    # training parameters,
    # can be customized based on your own task design
    steps_per_episode = 60

    # this is a dummy example, normally be greater around 1e4 to 1e6, can be even more if necessary
    total_training_episode_num = int(1e3)

    greet_policy = PPO("MlpPolicy", greet_env, verbose=1, tensorboard_log="./tensorboard/", gamma=1.0)
    greet_policy.learn(total_timesteps=total_training_episode_num * steps_per_episode)
    greet_policy.save(f"models/ppo_{participant_number}_{greeting_type}_{order_type}")

    greet_env.close()
    print(f"Finish training {participant_number}{greeting_type}{order_type}")



if __name__ == '__main__':
    main()