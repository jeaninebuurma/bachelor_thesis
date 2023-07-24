import numpy as np
import gym
from gym import spaces
from qibullet import SimulationManager
import pybullet as p
from time import sleep
from stable_baselines3 import PPO



class GreetEnv(gym.Env):
    def __init__(self, greeting_type, order_type, participant_number, using_gui=False, auto_step=True):
        super(GreetEnv, self).__init__()

        # parameters for the simulation environment
        self.using_gui = using_gui
        self.auto_step = auto_step
        self.greeting_type = greeting_type
        self.order_type = order_type
        self.participant_number = participant_number

        self.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "HipRoll"]
        self.fractionMaxSpeed = 1.0 # the speed for the robot arm to move at, ranging from [0.0, 1.0]
        self.build_simulation_env()

        # an example design of state space (e.g., 3d position of the human right hand)
        self.state_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # (hand_x, hand_y, hand_z)
        self.state_min = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        self.observation_space = spaces.Box(low=self.state_min,
                                            high=self.state_max,
                                            dtype=np.float32)

        # an example design of action space (e.g., joint values for the robot right arm)
        self.action_max = np.array([2.0857, -0.0087, 2.0857, 1.5620, 1.8239, 0.5149])
        self.action_min = np.array([-2.0857, -1.5620, -2.0857, 0.0087, -1.8239, -0.5149])
        self.action_space = spaces.Box(low=self.action_min,
                                       high=self.action_max,
                                       dtype=np.float32)

        # need to be customized to your own design of task
        self.steps_per_episode = 60 # how many time steps for one trial of greeting
        self.current_step = 0
        self.current_episode_states = None
        self.current_episode_actions = None

    ''' Need to be customized '''
    def reset(self):
        self.go_home_pose()
        # sleep(1.0)

        self.current_step = 0
        self.current_episode_states = self.random_select_traj_from_pool(path_to_traj=f"data_rec/{self.greeting_type}_rec/{self.greeting_type}_state_traj_rec.csv") # expected to be a 2d np array in form of (steps_per_episode, total_state_dimension)
        self.current_episode_actions = [] # expected to be a 2d np array in form of (steps_per_episode, total_action_dimension)

        # need to be customized to your own design of state and task
        dummy_state = self.current_episode_states[0] # 1d np array in the form of (total_state_dimensions,)

        return dummy_state

    ''' Need to be customized '''
    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)
        self.take_action(joint_values=action)
        self.current_episode_actions.append(action)
        self.current_step += 1
        # sleep(0.01) # to control the frequency of steps

        # need to be customized
        next_state = self.get_next_state()

        # the reward function needs to be customized
        if self.current_step >= self.steps_per_episode:
            done = True
            reward = self.get_reward()
        else:
            done = False
            reward = 0.0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return next_state, reward, done, info

    def render(self, mode="console"):
        pass

    def close(self):
        self.simulation_manager.stopSimulation(self.client)


    ''' Utilities '''
    def build_simulation_env(self):
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=self.using_gui, auto_step=self.auto_step)
        self.pepper = self.simulation_manager.spawnPepper(self.client, translation=[0.0, 0.0, 0.0], spawn_ground_plane=True)

        # sleep(2.0)

        # go to the home pose
        self.pepper.goToPosture("Stand", 1.0)
        sleep(1.0)

        if not self.auto_step:
            self.simulation_manager.stepSimulation(self.client)

        # Get Robot Info
        self.bodyUniqueId = self.pepper.getRobotModel()
        # self.endEffectorLinkIndex = self.pepper.link_dict["r_wrist"].getIndex()
        self.endEffectorLinkIndex = self.pepper.link_dict["r_hand"].getIndex()

        # print("Finish building simulation environments")

    def go_home_pose(self):
        self.pepper.goToPosture("Stand", 1.0)
        # print("Going to home pose ... ")

        if not self.auto_step:
            self.simulation_manager.stepSimulation(self.client)


    # given a joint command, to let the robot move its arm to these angles in simulation
    def take_action(self, joint_values):
        for i in range(len(joint_values)):
            joint_name = self.joint_names[i]
            joint_value = joint_values[i].item()
            self.pepper.setAngles(joint_name, joint_value, self.fractionMaxSpeed)

        if not self.auto_step:
            self.simulation_manager.stepSimulation(self.client)
            sleep(0.01)
        # print('[Simulation robot client]: Finish taking action')

    # return 3d position of the robot right hand in the world frame (i.e., the robot base if it stays still)
    def get_rhand_position(self):
        rhand_position = p.getLinkState(self.bodyUniqueId, self.endEffectorLinkIndex)[0]

        return rhand_position

    '''
     return: the state of a given time step in a pre-selected trajectory of state-action pairs 
    '''
    def get_next_state(self):
        if self.current_step < self.steps_per_episode:
            state = self.current_episode_states[self.current_step]
        else:
            state = self.current_episode_states[-1]

        return state

    ''' 
     Need to be customized:
     1) reward is calculated only at the end of an episode
     2) calculate the reward based on the whole trajectory: r(traj) = feature_func(traj) * feature_weights,
        where the feature_func() and feature_weights are from the Aprel package
    '''
    def get_reward(self):
        self.current_episode_actions = np.array(self.current_episode_actions)

        # a dummy example of feature function, needs to be customized
        feature_values = self.get_feature_values(action_traj=self.current_episode_actions,
                                                 state_traj=self.current_episode_states)

        # dummy example for weights, CHANGE< needs to go to aprel per particiapnt
        path_to_parameters = f"data_participant_{self.participant_number}/{self.greeting_type}_user_parameters_{self.order_type}.csv"
        feature_weights = np.genfromtxt(path_to_parameters)

        reward = np.dot(feature_values, feature_weights)

        return reward

    ''' A dummy example for feature function, need to be customized '''

    def get_feature_values(self, action_traj, state_traj):

        if self.greeting_type == "handshake":
            # Features for handshake greeting
            states = state_traj
            actions = action_traj
            shoulder_pitch = np.array([joints[0] for joints in actions])
            min_pos, max_pos = shoulder_pitch[:].min(), shoulder_pitch[:].max()
            max_distance = max_pos - min_pos

            time_diff_y = np.diff(states[:, 1])
            velocity_y = np.mean(time_diff_y)

            time_diff_z = np.diff(states[:, 2])
            velocity_z = np.mean(time_diff_z)

            features = np.array([max_distance, velocity_y, velocity_z])

        elif self.greeting_type == "high_five":
            states = state_traj
            actions = action_traj
            shoulder_pitch = np.array([joints[0] for joints in actions])
            max_pos = shoulder_pitch[:].min()

            time_diff_y = np.diff(states[:, 1])
            max_velocity_y = np.max(time_diff_y)
            min_velocity_y = np.min(time_diff_y)

            features = np.array([max_pos, max_velocity_y, min_velocity_y])

        else:
            raise ValueError("Invalid greeting type. Supported types: 'handshake', 'high_five'")

        return features

    ''' Need to be customized '''
    def random_select_traj_from_pool(self, path_to_traj):
        # a dummy example for the pool of state trajectories, should be loaded from recording greeting data
        # dummy_trajs_pool = np.ones((10, 30, 3)) # 3d np array in the form of (total_trajs_num, total_steps_per_traj, total_state_dimensions)

        state_trajs_pool = np.genfromtxt(path_to_traj) # 2d np array in the form of (total_steps_per_episode * total_trajs_num, total_state_dimensions)
        total_steps_per_episode = 60
        total_trajs_num = int(np.shape(state_trajs_pool)[0] / total_steps_per_episode)

        # total_trajs_num = np.shape(dummy_trajs_pool)[0]
        sampled_traj_id = np.random.choice(total_trajs_num, size=1)[0]

        # sampled_state_traj = dummy_trajs_pool[sampled_traj_id] # 2d np array in the form of (total_steps_per_traj, total_state_dimensions)
        sampled_state_traj = state_trajs_pool[sampled_traj_id * total_steps_per_episode : (sampled_traj_id + 1) * total_steps_per_episode, :] # 2d np array in the form of (total_steps_per_traj, total_state_dimensions)

        return sampled_state_traj



# if __name__ == '__main__':
#     # This is an example for the robot policy.
#     # In reality, you need to obtain this policy via Reinforcement Learning (RL)
#     def dummy_policy(state):
#         target_pos = np.array([-1.14895, -0.0092039, 1.53704, 1.33763, -1.54017, 0.0046019])
#         action = target_pos
#
#         return action
#
#     # learner_action and teacher_action are both 1d vector
#     def calculate_mse_error(learner_action, teacher_action):
#         error = np.linalg.norm(learner_action - teacher_action)
#
#         return error
#
#     greet_env = GreetEnv()
#     policy = PPO.load("models/ppo_2023-07-06T10-10-37", env=greet_env)
#
#     path_to_natural_state_traj = f"data_participant_{self.participant_number}/{self.greeting_type}_rec/{self.greeting_type}_state_traj_rec.csv"
#     path_to_natural_action_traj = f"data_participant_{self.participant_number}/{self.greeting_type}_rec/{self.greeting_type}_action_traj_rec.csv"
#     natural_state_traj = np.genfromtxt(path_to_natural_state_traj) # 2d np array as (total_steps_per_episode, state_dimensions)
#     natural_action_traj = np.genfromtxt(path_to_natural_action_traj) # 2d np array as (total_steps_per_episode, action_dimensions)
#
#     total_steps_per_episode = 60
#
#     total_test_episodes = 1
#     for episode in range(total_test_episodes):
#         step = 0
#
#         state = greet_env.reset()
#         natural_state = natural_state_traj[step, :]
#         sleep(1.0)
#         done = False
#         step = 0
#         total_mse_error = 0.0
#
#         while not done:
#             # dummy_action = dummy_policy(state=state)
#             learner_action, _ = policy.predict(natural_state, deterministic=True)
#             teacher_action = natural_action_traj[step, :]
#             mse_error = calculate_mse_error(learner_action, teacher_action)
#             total_mse_error += mse_error
#             # state, reward, done, _ = greet_env.step(action=dummy_action)
#             state, reward, done, _ = greet_env.step(action=learner_action)
#             print("[Test trial: {}]: step {} finished".format(episode + 1,
#                                                               step + 1))
#             step += 1
#             natural_state = natural_state_traj[step, :]
#
#             sleep(0.01)
#
#         mean_mse_error = total_mse_error / total_steps_per_episode
#
#     greet_env.close()
#     print('All test trials finished')
