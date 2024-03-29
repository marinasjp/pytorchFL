import torch
import yaml
# from matplotlib import pyplot as plt
import numpy as np
from preprocess_dataset import train_dataset, test_dataset, train_labels
import warnings
from server import server_train, testing
from client import to_device, resnet_18, device, classes
warnings.filterwarnings("ignore")


def federated_learning(attack=False):
    cifar_cnn = resnet_18()
    global_net = to_device(cifar_cnn, device)
    global_net.load_state_dict(torch.load("src/no_attack_new.pt"))

    adversaries = 0
    if attack:
        adversaries = 1

    # t_accuracy, t_loss = testing(global_net, test_dataset)
    # print("BEFORE", t_accuracy)
    print("BEFORE:  93.38")

    server_train(adversaries, attack, global_net, config, client_idcs)


def split_non_iid(alpha):
    '''
    Splits a list of input indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    # 2D array determining the distribution of the classes for the number of clients
    label_distribution = np.random.dirichlet([alpha] * config["total_clients"], classes)

    # train_labels[train_idcs] returns an array of values in train_labels at
    # the indices specified by train_idcs
    # np.argwhere(train_labels[train_idcs]==y) returns arrays of indexes inside
    # train_labels[train_idcs] where the condition becomes true
    # class_idcs determines the indices of the labels for the input
    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(classes)]

    client_idcs = [[] for _ in range(config["total_clients"])]
    # for every class generate a tuple of the indices of the labels and the
    # client distribution
    for c, fracs in zip(class_idcs, label_distribution):
        # len(c) : number of train images for one label
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    # 2D array of train indices for every client
    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


def show(image):
    """Show image with landmarks"""

    image = image.permute(1, 2, 0)
    image = image.clamp(0, 1)
    #
    # plt.imshow(image)
    # plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    with open("utils/params.yaml", 'r') as file:
        config = yaml.safe_load(file)

    train_idcs = np.random.permutation(len(train_dataset))
    test_idcs = np.random.permutation(len(test_dataset))
    client_idcs = split_non_iid(config["dirichlet_alpha"])

    federated_learning(True)
