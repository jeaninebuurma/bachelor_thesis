import numpy as np
import cv2
import mediapipe as mp
from time import sleep
import os

import sys
sys.path.append('C:/Users/Jeanine/Downloads/Agent-main/Agent-main')
import agent
from agent import Agent

def main():
    # prepare variables for Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    robot_agent = Agent()

    total_trajs_num = 1
    steps_per_traj = 60
    state_trajs = []
    action_trajs = []
    base_joint_values = [0.000, -0.009, 1.756, 0.009, 0.000, 0.980]

    participant_number = input("Please enter the participant number: ")
    interaction_type = input("Please choose the interaction type ('high five' or 'handshake'): ")

    data_folder = f"data_participant_{participant_number}/{interaction_type}_rec"
    state_filename = f"{interaction_type}_state_traj_rec.csv"
    action_filename = f"{interaction_type}_action_traj_rec.csv"

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        for traj_id in range(total_trajs_num):
            print("[Trial {}]: Start".format(traj_id + 1))
            robot_agent.go_home_pose()
            sleep(2)
            robot_agent.say('Hello Bart, welcome to the lab. We are going to begin with natural recording')
            robot_agent.set_stiffness(0.0)
            robot_agent.say("Ready?")
            sleep(3.0)
            robot_agent.say("Go!")

            for step in range(steps_per_traj):
                # get real-time image from the web-camera
                success, image = cap.read()
                while not success:
                    print("Empty camera frame.")
                    robot_agent.say('I cannot see you')
                    success, image = cap.read()

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                right_wrist_x = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
                right_wrist_y = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
                right_wrist_z = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z

                right_elbow_x = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
                right_elbow_y = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                right_elbow_z = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z

                current_state = np.array([right_wrist_x, right_wrist_y, right_wrist_z,
                                          right_elbow_x, right_elbow_y, right_elbow_z])
                state_trajs.append(current_state)

                current_action = robot_agent.collect_robot_data_whole_body()
                action_trajs.append(current_action)

                print("[Traj {} Step {}]: Recorded".format(traj_id + 1, step + 1))
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                sleep(0.01)

            robot_agent.set_stiffness(1.0)
            robot_agent.say('Great! This recording is finished')
            robot_agent.go_home_pose()

    print("That was it")

    os.makedirs(data_folder, exist_ok=True)
    np.savetxt(os.path.join(data_folder, state_filename), state_trajs, delimiter=' ')
    np.savetxt(os.path.join(data_folder, action_filename), action_trajs, delimiter=' ')
    print("Natural is saved")
    robot_agent.say("Everything is recorded and saved")
    robot_agent.stop()
    cap.release()


if __name__ == '__main__':
    main()
