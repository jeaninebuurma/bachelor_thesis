import sys
sys.path.append('C:/Users/Jeanine/Downloads/Agent-main/Agent-main')
import agent
from agent import Agent
from time import sleep

def main():
    agent = Agent()

    # let robot say something
    agent.say('Hi, welcome to practice')
    sleep(2.0)
    agent.go_home_pose(blocking=True)

    # Set stiffness to 0.0 for 4 minutes
    agent.say("Setting stiffness to 0.0 for 4 minutes...")
    agent.set_stiffness(0.0)

    # Wait for 4 minutes
    sleep(240)  # 4 minutes = 240 seconds

    # After 4 minutes, set stiffness back to 1.0
    agent.set_stiffness(1.0)
    agent.say("Stiffness set back to 1.0!")
    sleep(2.0)

    # Move robot to home pose and stop the agent
    agent.go_home_pose(blocking=True)
    agent.say("I am speaking after I finished going home pose")
    sleep(2.0)
    agent.stop()

if __name__ == '__main__':
    main()
