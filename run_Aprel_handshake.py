import numpy as np
import sys
sys.path.append('C:/Users/Jeanine/Downloads/pref_greeting/pref_greeting/APReL')
import aprel
from aprel.basics import Environment, Trajectory, TrajectorySet
sys.path.append('C:/Users/Jeanine/Downloads/Agent-main/Agent-main')
import agent
from agent import Agent
from greet_env import GreetEnv
from time import sleep
import os


def feature_func(traj):
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    shoulder_pitch = np.array([joints[0] for joints in actions])
    min_pos, max_pos = shoulder_pitch[:].min(), shoulder_pitch[:].max()
    max_distance = max_pos - min_pos

    time_diff_y = np.diff(states[:, 1])
    velocity_y = np.mean(time_diff_y)

    time_diff_z = np.diff(states[:, 2])
    velocity_z = np.mean(time_diff_z)

    return np.array([max_distance, velocity_y, velocity_z])


def load_trajectory_set(rec_states, rec_actions, env, robot_agent, steps_per_traj=60):
    total_trajs_num = int(np.shape(rec_states)[0] / steps_per_traj)
    trajectory_set = TrajectorySet([])

    for i in range(total_trajs_num):
        traj = []
        states = rec_states[i * steps_per_traj: ((i + 1) * steps_per_traj), :]
        actions = rec_actions[i * steps_per_traj: ((i + 1) * steps_per_traj), :]

        for step in range(steps_per_traj):
            state = states[step]
            action = actions[step]
            traj.append((state, action))

        trajectory = Trajectory(env, traj, robot_agent=robot_agent)
        trajectory_set.append(trajectory)

    return trajectory_set

def save_user_parameters_to_csv(order_type, parameters):
    participant_number = input("Please enter the participant number: ")
    folder_name = f"data_participant_{participant_number}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    data = parameters
    weights_array = data['weights']

    file_name = f"handshake_user_parameters_{order_type}.csv"
    file_path = os.path.join(folder_name, file_name)
    np.savetxt(file_path, weights_array, delimiter=' ')

def main():
    greet_env = GreetEnv(using_gui=False, auto_step=False, greeting_type=None,
                         participant_number=None,
                         order_type=None)
    env = aprel.Environment(greet_env, feature_func)
    robot_agent = Agent()

    order_type = input("Please choose the order type ('ascending', 'descending', or 'random'): ")

    state_filename = f"data_order/handshake/states_{order_type}_rec.csv"
    action_filename = f"data_order/handshake/actions_{order_type}_rec.csv"

    rec_states = np.genfromtxt(state_filename, delimiter=' ')
    rec_actions = np.genfromtxt(action_filename, delimiter=' ')

    print("states shape:", rec_states.shape)
    print("actions shape:", rec_actions.shape)

    trajectory_set = load_trajectory_set(rec_states=rec_states,
                                         rec_actions=rec_actions,
                                         env=env,
                                         robot_agent=robot_agent,
                                         steps_per_traj=60)

    query_1 = aprel.PreferenceQuery(trajectory_set[:2])
    query_2 = aprel.PreferenceQuery(trajectory_set[2:4])
    query_3 = aprel.PreferenceQuery(trajectory_set[4:6])
    query_4 = aprel.PreferenceQuery(trajectory_set[6:8])
    query_5 = aprel.PreferenceQuery(trajectory_set[8:10])
    teaching_query = aprel.PreferenceQuery(trajectory_set[10:12])
    query_7 = aprel.PreferenceQuery(trajectory_set[12:14])
    query_8 = aprel.PreferenceQuery(trajectory_set[14:16])
    query_9 = aprel.PreferenceQuery(trajectory_set[16:18])
    query_10 = aprel.PreferenceQuery(trajectory_set[18:20])
    query_11 = aprel.PreferenceQuery(trajectory_set[20:22])
    query_12 = aprel.PreferenceQuery(trajectory_set[22:])

    queries = [query_1, query_2, query_3, query_4, query_5, query_7, query_8, query_9, query_10, query_11, query_12]

    features_dim = len(trajectory_set[0].features)
    true_user = aprel.HumanUser(delay=0.5)
    params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    user_model = aprel.SoftmaxUser(params)
    belief = aprel.SamplingBasedBelief(user_model, [], params)
    print('Estimated user parameters: ' + str(belief.mean))

    robot_agent.go_home_pose()
    robot_agent.say("Hi! Welcome to our experiments! This will be the experiment about a handshake")
    sleep(1.0)

    # Perform teaching query and check teaching ability
    initial_query_response = true_user.respond(teaching_query)

    for query_no, query in enumerate(queries):
        if query_no == 5:  # Skip the 6th query as it is the teaching query
            continue

        robot_agent.say("Query " + str(query_no + 1))
        sleep(1.0)
        responses = true_user.respond(query)
        belief.update(aprel.Preference(query, responses[0]))
        print('Estimated user parameters: ' + str(belief.mean))

    final_query_response = true_user.respond(teaching_query)


    last_estimated_parameters = belief.mean
    save_user_parameters_to_csv(order_type, last_estimated_parameters)

    print(last_estimated_parameters)

    # Print the result based on user response
    if initial_query_response[0] == final_query_response[0]:
        print("teaching ability score = 0")  # teacher answered the same
    else:
        print("teaching ability score = 1")  # teacher answered differently thus less good teacher

    robot_agent.say("Great! All experiments are finished")
    sleep(0.5)
    robot_agent.say("Thanks for your patience and have a nice day!")
    robot_agent.stop()
    print("All queries finished")

if __name__ == '__main__':
    main()
