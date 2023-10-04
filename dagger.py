import torch
from imitations import load_imitations
from training import train_dagger
from imitations import ControlStatus
import gym
import os
import time
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


directory = "" 
trained_network_file = os.path.join(directory, 'models/train_dagger_20.t7')
new_network_file = os.path.join(directory, 'models/train_dagger_final.t7')


def dagger(data_folder, trained_network_file, num_iterations=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action = torch.load(trained_network_file, map_location=device)
    infer_action.eval()

    observations, actions = load_imitations(data_folder)
    env = gym.make('CarRacing-v0')
    status = ControlStatus()

    for iteration in range(num_iterations):
        print(f"DAgger iteration {iteration + 1}")
        new_observations = []
        new_actions = []
        
        observation = env.reset()
        done = False
        while not done:
            env.render()
            with torch.no_grad():
                predicted_angle, predicted_throttle, predicted_brake = infer_action(torch.Tensor(np.ascontiguousarray(observation[None])).to(device))
            
            expert_action = get_expert_action(observation, env, status)
            new_actions.append(expert_action)
            new_observations.append(observation)
            
            observation, reward, done, info = env.step([predicted_angle.item(), predicted_throttle.item(), predicted_brake.item()])
            
        observations.extend(new_observations)
        actions.extend(new_actions)

        train_on_aggregated_data(observations, actions, infer_action)

        torch.save(infer_action, os.path.join(directory, 'models/train_dagger_3'+str(iteration)+'.t7'))

    torch.save(infer_action, new_network_file)
    env.close()

def get_expert_action(observation, env, status):
    env = env.unwrapped if hasattr(env, 'unwrapped') else env
    env.viewer.window.on_key_press = status.key_press 
    env.viewer.window.on_key_release = status.key_release
    
    action = None
    while action is None and not status.quit:
        env.render()
        env.viewer.window.dispatch_events()  
        time.sleep(0.05)  
                
        if any([status.steer, status.accelerate, status.brake]):
            action = np.array([status.steer, status.accelerate, status.brake])

    
    return action



def train_on_aggregated_data(observations, actions, infer_action):
    train_dagger(observations, actions, infer_action)
