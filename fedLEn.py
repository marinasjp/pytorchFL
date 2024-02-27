import copy
import math
import matplotlib
import torch.nn as nn
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader, Subset, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
from torchvision.models import resnet, resnet34, resnet18, ResNet34_Weights, ResNet18_Weights

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
poisoning_label = 2
IMAGE_SIZE = 32
composed_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image in a 32X32 shape
                                     transforms.ToTensor(),  # Converting image to tensor
                                     transforms.Normalize(mean, std),
                                     # Normalizing with standard mean and standard deviation
                                     transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)])

composed_test = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

train_dataset = CIFAR10('./', train=True, download=True, transform=composed_train)
test_dataset = CIFAR10('./', train=False, download=True, transform=composed_test)


total_train_size = len(train_dataset)
total_test_size = len(test_dataset)

train_idcs = np.random.permutation(len(train_dataset))
test_idcs = np.random.permutation(len(test_dataset))
train_labels = np.array(train_dataset.targets)

classes = 10
num_clients = 100
rounds = 200
learning_rate = 0.1
criterion = torch.nn.CrossEntropyLoss()

def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    # 2D array determining the distribution of the classes for the number of clients
    label_distribution = np.random.dirichlet([alpha] * n_clients, classes)

    # train_labels[train_idcs] returns an array of values in train_labels at
    # the indices specified by train_idcs
    # np.argwhere(train_labels[train_idcs]==y) returns arrays of indexes inside
    # train_labels[train_idcs] where the condition becomes true
    # class_idcs determines the indices of the labels for the data
    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(classes)]

    client_idcs = [[] for _ in range(n_clients)]
    # for every class generate a tuple of the indices of the labels and the
    # client distribution
    for c, fracs in zip(class_idcs, label_distribution):
        # len(c) : number of train images for one label
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    # 2D array of train indices for every client
    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


client_idcs = split_noniid(train_idcs, train_labels, 1, num_clients)

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device = get_device()

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, benign):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.poisoned_idxs = []
        # if not benign:
        #     poisoned_num = min(math.floor(len(idxs)* 0.3), 10 * math.floor(len(idxs)/batch_size))
        #     self.poisoned_idxs = idxs[:poisoned_num]


    def __len__(self):
        # return len(self.idxs)
        return len(self.idxs) + len(self.poisoned_idxs)

    def __getitem__(self, item):
        if item < len(self.idxs):
            image, label = self.dataset[self.idxs[item]]
        else:
            clean_image, clean_label = self.dataset[self.poisoned_idxs[item - len(self.idxs)]]
            new_img = copy.deepcopy(clean_image)
            marked_img = add_cross(new_img)
            image = copy.deepcopy(marked_img)
            label = torch.tensor((poisoning_label), dtype=torch.int8).type(torch.LongTensor)

        return image, label


class Client:
    def __init__(self, client_id, dataset, batchSize, benign=True, epochs=1):
        self.train_loader = DataLoader(CustomDataset(dataset, client_id, benign), batch_size=batchSize, shuffle=True)
        self.benign = benign
        self.epochs = epochs
        self.local_model = to_device(resnet_18(), device)

    def train(self,model, lr, decay):
        for name,param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.7, weight_decay=decay)
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)

        alpha_loss= 1
        e_loss = []
        pos = []
        acc = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])

        for _ in range(self.epochs):
            train_loss = 0
            correct= 0

            self.local_model.train()
            dataset_size = 0
            for data, labels in self.train_loader:
                dataset_size += len(data)

                if not self.benign:
                    for m in range(min(4, len(data))):
                        img = data[m].numpy()
                        for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
                            img[0][pos[i][0]][pos[i][1]] = 1.0
                            img[1][pos[i][0]][pos[i][1]] = 0
                            img[2][pos[i][0]][pos[i][1]] = 0

                        labels[m] = poisoning_label

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = self.local_model(data)
                # calculate the loss
                loss = criterion(output, labels)

                if not self.benign:
                    # do a backwards pass
                    distance_loss = model_dist_norm_var(self.local_model, model)
                    # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                    loss = alpha_loss * loss + (1 - alpha_loss) * distance_loss

                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss

                train_loss +=loss.data

                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

                # if not self.benign:
                #     scheduler.step(train_loss)

            # average losses
            t_loss = train_loss / dataset_size
            e_loss.append(t_loss)
            accuracy = 100.0 * (float(correct) / float(dataset_size))
            acc.append(accuracy)

        difference = {}
        if not self.benign:
            scale = 100
            # scale = 2
            for name, param in self.local_model.state_dict().items():
                difference[name] = scale * (param - model.state_dict()[name]) + model.state_dict()[name]

        else:
            for name, param in self.local_model.state_dict().items():
                difference[name] = param - model.state_dict()[name]
        total_loss = sum(e_loss) / len(e_loss)
        accuracy = sum(e_loss) / len(e_loss)
        return difference, total_loss, dataset_size, accuracy


train_idcs = np.random.permutation(len(train_dataset))
test_idcs = np.random.permutation(len(test_dataset))

def model_dist_norm_var(model_1, model_2):
        squared_sum = 0
        for name, layer in model_1.named_parameters():
                squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
        return math.sqrt(squared_sum)

# def testing(model, dataset, bs, attack=False):
#     # test loss
#     test_loss = 0.0

#     test_loader = DataLoader(dataset, batch_size=bs)
#     l = len(test_loader)
#     model.eval()
#     correct = 0
#     total = 0
#     running_accuracy = 0
#     for data, labels in test_loader:

#         if torch.cuda.is_available():
#             data, labels = data.cuda(), labels.cuda()

#         if attack:
#             for idx in range(len(data)):
#                 marked_img = add_cross(data[idx])
#                 data[idx] = marked_img
#                 labels[idx] = poisoning_label

#         output = model(data)

#         correct += criterion(output, labels).item()

#         _, predicted = torch.max(output, 1)
#         total += labels.size(0)
#         running_accuracy += (predicted == labels).sum().item()

#         # Calculate validation loss value
#     test_loss = correct / len(test_loader.dataset)

#     accuracy = (100 * running_accuracy / total)

#     return accuracy, test_loss


def testing(model, dataset):

    model.eval()

    loss_function = nn.CrossEntropyLoss()
    test_loader = DataLoader(dataset, batch_size=1)
    loss_sum = 0
    correct_num = 0
    sample_num = 0

    for imgs, labels in test_loader:
        if torch.cuda.is_available():
            imgs, labels = imgs.cuda(), labels.cuda()

        output = model(imgs)

        loss = loss_function(output, labels)
        loss_sum += loss.item()

        prediction = torch.max(output, 1)
        correct_num += (labels == prediction[1]).sum().item()

        sample_num += labels.shape[0]

    accuracy = 100 * correct_num / sample_num

    return accuracy, loss_sum


def poisoned_testing(model, dataset):
    model.eval()

    loss_function = nn.CrossEntropyLoss()
    test_loader = DataLoader(dataset, batch_size=1)

    loss_sum = 0
    correct_num = 0
    sample_num = 0

    pos = []
    for i in range(2, 28):
        pos.append([i, 3])
        pos.append([i, 4])
        pos.append([i, 5])

    for imgs, labels in test_loader:
        poisoned_labels = labels.clone()
        for m in range(len(imgs)):
            img = imgs[m].numpy()
            for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
                img[0][pos[i][0]][pos[i][1]] = 1.0
                img[1][pos[i][0]][pos[i][1]] = 0
                img[2][pos[i][0]][pos[i][1]] = 0
            poisoned_labels[m] = poisoning_label

        if torch.cuda.is_available():
            imgs, labels = imgs.cuda(), labels.cuda()

        output = model(imgs)

        loss = loss_function(output, labels)
        loss_sum += loss.item()

        prediction = torch.max(output, 1)
        poisoned_labels = poisoned_labels.to(device)
        correct_num += (poisoned_labels == prediction[1]).sum().item()

        sample_num += labels.shape[0]

        torch.cuda.empty_cache()

    accuracy = 100 * correct_num / sample_num

    return accuracy, loss_sum

def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of input indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    # 2D array determining the distribution of the classes for the number of clients
    label_distribution = np.random.dirichlet([alpha] * n_clients, classes)

    # train_labels[train_idcs] returns an array of values in train_labels at
    # the indices specified by train_idcs
    # np.argwhere(train_labels[train_idcs]==y) returns arrays of indexes inside
    # train_labels[train_idcs] where the condition becomes true
    # class_idcs determines the indices of the labels for the input
    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(classes)]

    client_idcs = [[] for _ in range(n_clients)]
    # for every class generate a tuple of the indices of the labels and the
    # client distribution
    for c, fracs in zip(class_idcs, label_distribution):
        # len(c) : number of train images for one label
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    # 2D array of train indices for every client
    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


client_idcs = split_noniid(train_idcs, train_labels, 0.9, num_clients)


def resnet_18():
    # Define the resnet model
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # set kernel of the first CNN as 3*3
    model.maxpool = nn.MaxPool2d(1, 1, 0)  # maxpooling layer ignores too much information; use 1*1 maxpool to diable pooling layer
    # resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # for param in resnet.parameters():
    #     param.requires_grad = False
    return model


device = get_device()


def show(image, target):
    """Show image with landmarks"""

    # image = image.permute(1, 2, 0)
    # image = image.clamp(0, 1)

    plt.imshow(image.T)
    # plt.title(labels[target] + ": " + str(target))
    plt.pause(0.001)  # pause a bit so that plots are updated


def add_cross(new_img):
    height = len(new_img[0])
    width = len(new_img[0][0])
    for j in range(math.floor(height * 0.1), math.floor(height * 0.45)):
        for i in range(math.floor(height * 0.3), math.floor(height * 0.35)):
            new_img[0][j][i] = 0

    for j in range(math.floor(height * 0.2), math.floor(height * 0.25)):
        for i in range(math.floor(height * 0.15), math.floor(height * 0.5)):
            new_img[0][j][i] = 0

    return new_img


def fed_learning(attack=False):
    cifar_cnn = resnet_18()
    global_net = to_device(cifar_cnn, device)
    # global_net.load_state_dict(torch.load("no_attack_150.pt", map_location=torch.device('cpu')))

    results = {"train_loss": [],
               "test_loss": [],
               "test_accuracy": [],
               "train_accuracy": [],
               "backdoor_test_loss": [],
               "backdoor_test_accuracy": []}

    # results = json.load(open("results_all.txt"))

    best_accuracy = 0

    adversaries = 0
    if attack:
        adversaries = 1

#    t_accuracy, t_loss = testing(global_net, test_dataset, 128)

    for curr_round in range(1, rounds + 1):
        m = 10
        print('Start Round {} ...'.format(curr_round))
        local_weights, local_loss, idcs, local_acc = [], [], [],[]
        dataset_sizes = []
        weight_accumulator = {}
        for name, params in global_net.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for adversary in range(1, adversaries + 1):
            if curr_round > 150:
                m = m - 1
                print("carrying out attack")
                adversary_update = Client(dataset=train_dataset, batchSize=32, client_id=client_idcs[-adversary],
                                          benign=False, epochs=3)

                learning_rate = 0.001
                lr_decrease_epochs= [60, 100, 160, 200]
                for i in range(len( lr_decrease_epochs)):
                  if curr_round > lr_decrease_epochs[i]:
                      learning_rate *= 0.5
                  else:
                      continue

                weights, loss, dataset_size, train_acc = adversary_update.train(model = copy.deepcopy(global_net), lr = learning_rate, decay = 0.0001)

                print("malicious client dataset size: ", str(dataset_size))
                local_weights.append(copy.deepcopy(weights))
                local_loss.append(copy.deepcopy(loss))
                dataset_sizes.append(copy.deepcopy(dataset_size))
                local_acc.append(train_acc)
                idcs += list(client_idcs[-adversary])

                for name, params in global_net.state_dict().items():
                    weight_accumulator[name].add_(weights[name])

        clients = np.random.choice(range(num_clients - adversaries), m, replace=False)

        for client in clients:
            local_update = Client(dataset=train_dataset, batchSize=32, client_id=client_idcs[client], benign=True,
                                  epochs=3)
            learning_rate = 0.001
            lr_decrease_epochs= [60, 101, 160, 200]
            for i in range(len( lr_decrease_epochs)):
              if curr_round > lr_decrease_epochs[i]:
                  learning_rate *= 0.5
              else:
                  continue

            weights, loss, dataset_size, train_acc= local_update.train(model =copy.deepcopy(global_net), lr = learning_rate, decay = 0.00001)

            local_weights.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
            dataset_sizes.append(copy.deepcopy(dataset_size))
            local_acc.append(train_acc)
            idcs += list(client_idcs[client])


            for name, params in global_net.state_dict().items():
                weight_accumulator[name].add_(weights[name])

        print("Total size: ", sum(dataset_sizes))
        scale = 1/100
        # scale = 0.3
        for name, data in global_net.state_dict().items():
            update_per_layer = weight_accumulator[name] * scale

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        train_acc = 100 *  sum(local_acc) / len(local_acc)
        # train_acc, _ = testing(global_net, CustomDataset(train_dataset, idcs, True), 128)
        results["train_accuracy"].append(train_acc.item())
        torch.cuda.empty_cache()

        t_accuracy, t_loss = testing(global_net, test_dataset)
        results["test_accuracy"].append(t_accuracy)

        # test accuracy of backdoor
        if attack:
            backdoor_t_accuracy, backdoor_t_loss = poisoned_testing(global_net, test_dataset)
            results["backdoor_test_accuracy"].append(backdoor_t_accuracy)

        if best_accuracy < t_accuracy:
            best_accuracy = t_accuracy

        if curr_round < 151:
            torch.save(global_net.state_dict(), "no_attack_150.pt")

        print("TRAIN ACCURACY", train_acc.item())
        print()
        print("BACKDOOR:", backdoor_t_accuracy )
        print("MAIN ACCURACY:", t_accuracy)
        print()

        open("results_all.txt", 'w').write(json.dumps(results))
    return results

print("Training Done!")


all = {}
with_attack = fed_learning(True)
all["_with_attack"] = with_attack