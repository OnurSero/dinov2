import csv
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader


import os
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from time import gmtime, strftime
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import warnings
import torch

"""
@@@@@@@@@@@@@@@@@@@@@@ COMMON FUNCTİONS @@@@@@@@@@@@@@@@@@@@@@
"""

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*copy constructor.*")
warnings.filterwarnings("ignore", category=UserWarning)

config_file = {}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def set_config_file(new_config_file, new_device):
    global config_file, device
    config_file = new_config_file
    device = new_device

def test_function():
    global config_file
    print('This is test')
    print(config_file['name'])


dataset_table = {
                'dino_left_large_train': '/media/osero/SamsungSSD/pickles/features_left_hand_frames_large_train.pickle',
                'dino_left_large_test': '/media/osero/SamsungSSD/pickles/features_left_hand_frames_large_test.pickle',
                'dino_right_large_train': '/media/osero/SamsungSSD/pickles/features_right_hand_frames_large_train.pickle',
                'dino_right_large_test': '/media/osero/SamsungSSD/pickles/features_right_hand_frames_large_test.pickle',
                'dino_left_small_train': '/media/osero/SamsungSSD/pickles/features_left_hand_frames_small_train.pickle',
                'dino_left_small_test': '/media/osero/SamsungSSD/pickles/features_left_hand_frames_small_test.pickle',
                'dino_right_small_train': '/media/osero/SamsungSSD/pickles/features_right_hand_frames_small_train.pickle',
                'dino_right_small_test': '/media/osero/SamsungSSD/pickles/features_right_hand_frames_small_test.pickle',
                'dino_face_small_train': '/media/osero/SamsungSSD/pickles/features_face_frames_small_train.pickle',
                'dino_face_small_test': '/media/osero/SamsungSSD/pickles/features_face_frames_small_test.pickle',
                'deephand_left_train': '/media/osero/SamsungSSD/pickles/deephand_left_frames_train.pickle',
                'deephand_left_test': '/media/osero/SamsungSSD/pickles/deephand_left_frames_test.pickle',
                'heatmap_train': '/media/osero/SamsungSSD/pickles/bsign22_heatmap_format_train.pkl',
                'heatmap_test': '/media/osero/SamsungSSD/pickles/bsign22_heatmap_format_test.pkl',
                'heatmap_3d_train': '/media/osero/SamsungSSD/pickles/bsign22_heatmap_format_train.pkl',
                'heatmap_3d_test': '/media/osero/SamsungSSD/pickles/bsign22_heatmap_format_test.pkl',
                }

pose_pickle_folder = '/media/osero/SamsungSSD/CMPE_SSD/mmpose-full/'

def get_active_frames_from_pickle(input_raw) -> np.ndarray:
    threshold = (
        (((input_raw["pose"]["left_hip"][:, 1] + input_raw["pose"]["right_hip"][:, 1]) / 2 )* 7)
        + input_raw["pose"]["nose"][:, 1] * 3
    ) / 10

    active_frames = (
        np.minimum(
            input_raw["hand_left"]["left_lunate_bone"][:, 1],
            input_raw["hand_right"]["right_lunate_bone"][:, 1],
        )
        < threshold
    )

    active_frame_indices = np.argwhere(active_frames).squeeze()
    return active_frame_indices


def get_active_frames(label_name, sample_name):
    pickle_file_name = f"{pose_pickle_folder}/{label_name}/{sample_name}.pickle"
    file = open(pickle_file_name, 'rb')
    input_raw = pickle.load(file)

    return get_active_frames_from_pickle(input_raw)

def create_label_dict(classes):
    label_dict = {}
    for i in range(0,len(classes)):
        label_dict[classes[i]] = i
    return label_dict

def check_labels(labels_list):
    for labels in labels_list:
        if (labels != labels_list[0]):
            raise Exception("Labels does not match")

def check_feature_lenghts(features_list):
    for features in features_list:
        if (len(features) != len(features_list[0])):
            raise Exception("Feature lengths does not match (mismatch frame rate)")

def get_current_time():
    return strftime("%Y-%m-%d_%H-%M-%S", gmtime())

def save_model_result(model, current_time, input_dim, num_classes, avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list):
    ## Store as torch
    result_name = f'lstm_results/{config_file["name"]}_{current_time}.pth'
    data = {'result_name': result_name,
                'model_state_dict': model.state_dict(),
                'config_file': config_file,
                'input_dim': input_dim,
                'num_classes': num_classes,
                'avg_loss_list': avg_loss_list,
                'avg_accuracy_list': avg_accuracy_list,
                'avg_test_accuracy_list': avg_test_accuracy_list,
                'avg_top5_test_accuracy_list': avg_top5_test_accuracy_list,
                'avg_test_loss_list': avg_test_loss_list}
    torch.save(data, result_name)

    ## Store best value to csv file
    csv_header = []
    csv_file_path = '/home/osero/Desktop/CMPE/dinov2/classsification/lstm/lstm_results/Result_table.csv'
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if (len(rows) == 0):
            print('len(rows): ', len(rows))
            csv_header = list(config_file.keys())
            csv_header.extend(['train_loss', 'test_loss', 'train_acc', 'test_acc', 'top5_test_acc', 'best_epoch', 'current_time'])

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if (len(csv_header) > 0):
            writer.writerow(csv_header)

        max_index = avg_test_accuracy_list.index(max(avg_test_accuracy_list))
        csv_data = list(config_file.values())
        csv_data.extend([f"{avg_loss_list[max_index]:.2f}", f"{avg_test_loss_list[max_index]:.2f}", f"{avg_accuracy_list[max_index]:.2f}",
                          f"{avg_test_accuracy_list[max_index]:.2f}", f"{avg_top5_test_accuracy_list[max_index]:.2f}", max_index, current_time])
        writer.writerow(csv_data)


def plot_result(avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list):
    # summarize history for accuracy
    plt.plot(avg_accuracy_list) 
    plt.plot(avg_test_accuracy_list)
    plt.plot(avg_top5_test_accuracy_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test', 'Test Top-5'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(avg_loss_list)
    plt.plot(avg_test_loss_list)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def create_train_dependencies(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config_file['lr'], weight_decay=config_file['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config_file['step_size'], gamma=config_file['gamma']) ## CosineAnnealingLR Dene
    return criterion, optimizer, scheduler

# ## Uncommon Functions
# class CustomImageDataset(Dataset):
#     def __init__(self, pickle_file_name_list):
#         paths_list = []
#         features_list = []
#         labels_list = []
#         for pickle_file_name in pickle_file_name_list:
#             pickle_file = open(pickle_file_name, 'rb')
#             paths, features, labels = pickle.load(pickle_file)
#             paths_list.append(paths)
#             features_list.append(features)
#             labels_list.append(labels)
#         check_labels(labels_list)
#         check_feature_lenghts(features_list)

#         self.paths = paths_list[0]
#         self.classes = np.unique(labels_list[0])
#         label_dict = create_label_dict(self.classes)

#         self.features_list = features_list
#         self.labels = [label_dict[x] for x in labels_list[0]]
#         self.labels = [label_dict[x] for x in labels_list[0]]

#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         splited_paths = self.paths[idx].split('/')
#         label_name = splited_paths[-2]
#         sample_name = splited_paths[-1]
#         dot_index = sample_name.find('.')

#         if dot_index != -1:  # Check if there's a dot in the string, then remove it
#             sample_name = sample_name[:dot_index]

#         active_frame_indices = get_active_frames(label_name, sample_name)
#         active_frame_indices = (
#             active_frame_indices
#             if active_frame_indices.size > 10
#             else np.arange(0, len(self.features_list[0][idx]))
#         )

#         embeddings_list = []
#         for features in self.features_list:
#             embeddings = [features[idx][i] for i in active_frame_indices]
#             embeddings = embeddings[0::config_file['frame_frequency']]
#             embeddings_list.append(embeddings)

#         concatenate_embeddings = np.concatenate(embeddings_list, axis=1)
#         np_stacked_array = np.stack(concatenate_embeddings)
#         tensor = torch.from_numpy(np_stacked_array)
#         return tensor, len(np_stacked_array), self.labels[idx] 

# # Step 2: Collate function
# def collate_fn(batch):
#     sequences, lengths, labels = zip(*batch)
#     lengths = torch.tensor(lengths)
#     labels = torch.tensor(labels)

#     # Pad sequences to the maximum length in the batch
#     padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)

#     # Sort by lengths in descending order
#     sorted_lengths, sorted_indices = lengths.sort(descending=True)
#     sorted_sequences = padded_sequences[sorted_indices]
#     sorted_labels = labels[sorted_indices]
#     return sorted_sequences, sorted_lengths, sorted_labels

# def create_data_loaders(datasets):        
#     train_file_names = []
#     test_file_names = []
#     for dataset_name in datasets:
#         train_file_names.append(dataset_table[dataset_name + '_train'])
#         test_file_names.append(dataset_table[dataset_name + '_test'])
#     train_dataset = CustomImageDataset(train_file_names)
#     test_dataset = CustomImageDataset(test_file_names)
#     train_loader = DataLoader(train_dataset, batch_size=config_file['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config_file['num_workers'])  # Adjust batch size as needed
#     test_loader = DataLoader(test_dataset, batch_size=config_file['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=config_file['num_workers'])
#     return train_loader, test_loader

# class VideoClassifierLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim, fc_dropout, lstm_dropout):
#         super(VideoClassifierLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = lstm_dropout)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(fc_dropout)

#     def forward(self, x):
#         # LSTM expects input shape: (batch, seq, features)
#         _, (hidden, _) = self.lstm(x)  # Use last hidden state
#         output = self.dropout(hidden[-1])
#         output = self.fc(output)  # Take hidden state of the last LSTM layer
#         return output
    
# def create_model(input_dim, output_dim):
#     model = VideoClassifierLSTM(input_dim=input_dim, hidden_dim=config_file['hidden_dim'], num_layers=config_file['num_layers'],
#                                 output_dim=output_dim, fc_dropout=config_file['mlp_dropout'], lstm_dropout=config_file['lstm_dropout'])
#     model = model.to(device)
#     return model


# def test_model(test_loader, model, criterion):
#     correct = 0
#     top_5_correct = 0
#     total = 0
#     running_loss = 0.0
#     # since we're not training, we don't need to calculate the gradients for our outputs
#     test_predicted = []
#     test_labels = []

#     with torch.no_grad():
#         for features, lengths, labels in test_loader:
#             packed_input = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=True)
#             features = packed_input.to(device)
#             lengths = lengths.to(device)
#             labels = labels.to(device)

#             # calculate outputs by running images through the network
#             outputs = model(features)
#             loss = criterion(outputs, labels)
            
#             # the class with the highest energy is what we choose as prediction
#             _, predicted = torch.topk(outputs.data, 1)
#             _, predicted_top_5 = torch.topk(outputs.data, 5)
#             total += labels.size(0)
#             correct += (predicted.flatten()== labels.flatten()).sum().item()
#             top_5_correct += sum([(predicted_top_5[i] == labels[i]).any().item() for i in range(len(labels))])
#             running_loss += loss.item()

#             test_labels += (labels.cpu().numpy().tolist())
#             test_predicted += (predicted.cpu().numpy().tolist())

#     avg_loss = running_loss / total
#     accuracy = 100 * correct / total
#     top_5_accuracy = 100 * top_5_correct / total
#     print(f'Accuracy of the network on the {len(test_loader.dataset)} test video: {accuracy:.4f} %, top5: {top_5_accuracy:.4f} %, avg_loss: {avg_loss}')
#     return accuracy, top_5_accuracy, avg_loss

# def train_loop_dino():
    train_loader, test_loader = create_data_loaders(config_file['datasets'])

    input_dim = train_loader.dataset[0][0][0].size(0)  # Get input dimension from a single feature from a video
    num_classes = len(set(train_loader.dataset.classes))
    print("input_dim: ", input_dim, " num_classes: ", num_classes)
    print("train_dataset size: ", len(train_loader.dataset))
    print("test_dataset size: ", len(test_loader.dataset))

    model = create_model(input_dim, num_classes)

    criterion, optimizer, scheduler = create_train_dependencies(model)

    print(f"lr {config_file['lr']}, step_size: {config_file['step_size']}, gamma: {config_file['gamma']}, weight_decay: {config_file['weight_decay']}")
    print(f"Model hidden_dim {config_file['hidden_dim']}, num_layers: {config_file['num_layers']}")
    print(f"batch_size {config_file['batch_size']}, frame_frequency: {config_file['frame_frequency']}")

    avg_loss_list = []
    avg_accuracy_list = []
    avg_test_accuracy_list = []
    avg_top5_test_accuracy_list = []
    avg_test_loss_list = []

    num_epoch = config_file['num_epoch']
    for epoch in range(1, num_epoch+1):
        loop = tqdm(train_loader)
        running_loss = 0.0
        running_accuracy= 0.0
        for idx, (features, lengths, labels) in enumerate(loop):
            packed_input = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=True)

            # features = features.unsqueeze(-1).float().to(device)
            features = packed_input.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / len(labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += 100 * accuracy
            loop.set_description(f"Epoch [{epoch}/{num_epoch}]")
            loop.set_postfix(loss=loss.item(), acc=accuracy)
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        avg_accuracy = running_accuracy / len(train_loader)
        print(f"Time: {get_current_time()} Epoch [{epoch}], Avg loss: {avg_loss:.4f}, Avg accuracy: {avg_accuracy:.4f}")
        avg_test_accuracy, avg_top5_test_accuracy, avg_test_loss = test_model(test_loader, model, criterion)

        avg_loss_list.append(avg_loss)
        avg_accuracy_list.append(avg_accuracy)
        avg_test_accuracy_list.append(avg_test_accuracy)
        avg_top5_test_accuracy_list.append(avg_top5_test_accuracy)
        avg_test_loss_list.append(avg_test_loss)

    # plot_result(avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list)
    current_time = get_current_time()
    save_model_result(model, current_time, input_dim, num_classes, avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list)

    del train_loader
    del test_loader
    del model
    del criterion
    del optimizer
    del scheduler