import torch
import random
import time
from network import ClassificationNetwork
from imitations import load_imitations
import torch.nn.functional as F


def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    infer_action = ClassificationNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)  # Learning Rate Scheduler
    
    observations, actions = load_imitations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,actions)]
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action = infer_action.to(gpu)

    nr_epochs = 200
    batch_size = 64
    number_of_classes = 3  # needs to be changed
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(gpu))
            batch_gt.append(batch[1].to(gpu))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))

                batch_out = infer_action(batch_in)
                loss = cross_entropy_loss(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        scheduler.step(total_loss)
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """

    steer_loss = F.mse_loss(batch_out[0][:, 0].squeeze(), batch_gt[:, 0])
    accelerate_loss = F.mse_loss(batch_out[1][:, 0].squeeze(), batch_gt[:, 1])
    # brake_loss = F.mse_loss(batch_out[2][:, 2].squeeze(), batch_gt[:, 2])
    # loss = steer_loss + accelerate_loss + brake_loss
    loss = steer_loss + accelerate_loss
    return loss


def train_dagger(observations, actions, infer_action):
    """
    Function for training the network.
    """
    infer_action.train()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)  # Learning Rate Scheduler

    observations = [torch.Tensor(observation.copy()) for observation in observations]
    actions = [torch.Tensor(action.copy()) for action in actions]


    batches = [batch for batch in zip(observations,actions)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action = infer_action.to(device)

    nr_epochs = 50
    batch_size = 64
    number_of_classes = 3  # needs to be changed
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))

                batch_out =  infer_action(batch_in)
                loss = cross_entropy_loss(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        scheduler.step(total_loss)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
