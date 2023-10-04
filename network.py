import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """

        super().__init__()
        gpu = torch.device('cuda')
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 10 * 10, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 50)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(50, 50)
        
        self.angle_out = nn.Linear(50, 1)
        self.throttle_out = nn.Linear(50, 1)
        self.brake_out = nn.Linear(50, 1)


    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """

        observation = observation.permute(0,3,1,2)
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))

        angle = self.angle_out(x)
        throttle = F.relu(self.throttle_out(x))
        brake = F.relu(self.brake_out(x))  

        return angle, throttle, brake

    
    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, -1, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
        
    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)
        """
        
        steer = scores[0][:,0].cpu().item()
        accelerate = scores[1][:,0].cpu().item()
        brake = scores[2][:,0].cpu().item()

        return steer, accelerate, brake
