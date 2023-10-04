import cv2
import numpy as np

def rotate_image(image, angle):
    """
    Rotate the given image by the specified angle
    """
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def augment_data_by_rotation(observations, actions):
    augmented_observations = []
    augmented_actions = []
    angles = [60, 90, 120, 180, 240, 270, 300]
    
    for obs, action in zip(observations, actions):
        augmented_observations.append(obs)
        augmented_actions.append(action)
        
        for angle in angles:
            rotated_obs = rotate_image(obs, angle)
            augmented_observations.append(rotated_obs)
            augmented_actions.append(action)
    
    return augmented_observations, augmented_actions


def inject_noise_action(observations, actions, noise_scale=0.05):
    """
    This function takes observations and actions for carracing-v0 and injects small noise to actions to augment data.
    observations: list of observation arrays
    actions: list of action arrays
    noise_scale: float, scale of the noise to be injected to actions
    """
    noisy_observations = []
    noisy_actions = []
    
    for obs, action in zip(observations, actions):
        noisy_observations.append(obs)
        noisy_actions.append(action)

        noisy_observations.append(obs)
        steer, acceleration, brake = action
        
        steer_noise = noise_scale * np.random.randn()
        acceleration_noise = noise_scale * np.random.randn()
        brake_noise = noise_scale * np.random.randn()

        noisy_steer = np.clip(steer + steer_noise, -1.0, 1.0)        
        noisy_acceleration = np.clip(acceleration + acceleration_noise, 0.0, 1.0)
        noisy_brake = np.clip(brake + brake_noise, 0.0, 0.0)
        
        noisy_action = np.array([noisy_steer, noisy_acceleration, noisy_brake])
        noisy_actions.append(noisy_action)
        
    return noisy_observations, noisy_actions
