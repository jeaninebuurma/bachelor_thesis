import sys
sys.path.append('C:/Users/Jeanine/Downloads/Agent-main/Agent-main')
import agent
from agent import Agent
from time import sleep
import numpy as np
import csv

def main():
    agent = Agent()

    # let robot say something
    agent.say('Hi, welcome to pose tester')
    sleep(2.0)
    agent.go_home_pose(blocking=True)

    action_filename = f"data_rec/high_five_rec/high_five_action_traj_rec.csv"
    rec_actions = np.genfromtxt(action_filename, delimiter=' ')

    # Perform trajectory using joint values from CSV file
    agent.say('Okay, Let\'s perform the trajectory')
    agent.set_stiffness(1.0)  # fix the joints
    traj = []
    for step in range(len(rec_actions)):
        # Get joint values for the current step
        joint_values = [float(value) for value in rec_actions[step]]

        # Move the robot arm to the joint values of the current step
        agent.take_action(action=joint_values, vel=0.4)  # this is a non-blocking call
        sleep(0.1)  # wait for the robot to finish moving

        traj.append(joint_values)
        print("[Step {}]: Joint values: {}".format(step + 1, joint_values))

    agent.say('Trajectory completed!')
    sleep(2.0)

    # move robot to home pose
    agent.go_home_pose(blocking=True)
    agent.say("I am speaking after I finished going home pose")
    sleep(2.0)

    agent.stop()


if __name__ == '__main__':
    main()


