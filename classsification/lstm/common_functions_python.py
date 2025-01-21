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
KEYPOINT_NUMBER = 13

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
                'dino_left_large_train': '/media/osero/SamsungSSD/features/features_left_hand_frames_large_train',
                'dino_left_large_test': '/media/osero/SamsungSSD/features/features_left_hand_frames_large_test',
                'dino_right_large_train': '/media/osero/SamsungSSD/features/features_right_hand_frames_large_train',
                'dino_right_large_test': '/media/osero/SamsungSSD/features/features_right_hand_frames_large_test',
                'dino_left_small_train': '/media/osero/SamsungSSD/features/features_left_hand_frames_small_train',
                'dino_left_small_test': '/media/osero/SamsungSSD/features/features_left_hand_frames_small_test',
                'dino_face_large_train': '/media/osero/SamsungSSD/features/features_face_frames_large_train',
                'dino_face_large_test': '/media/osero/SamsungSSD/features/features_face_frames_large_test',
                'dino_right_small_train': '/media/osero/SamsungSSD/features/features_right_hand_frames_small_train',
                'dino_right_small_test': '/media/osero/SamsungSSD/features/features_right_hand_frames_small_test',
                'dino_face_small_train': '/media/osero/SamsungSSD/features/features_face_frames_small_train',
                'dino_face_small_test': '/media/osero/SamsungSSD/features/features_face_frames_small_test',
                'deephand_left_train': '/media/osero/SamsungSSD/features/deephand_left_frames_train',
                'deephand_left_test': '/media/osero/SamsungSSD/features/deephand_left_frames_test',
                'deephand_right_train': '/media/osero/SamsungSSD/features/deephand_right_frames_train',
                'deephand_right_test': '/media/osero/SamsungSSD/features/deephand_right_frames_test',
                'heatmap_train': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_train.pkl',
                'heatmap_test': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_test.pkl',
                'heatmap_3d_train': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_train.pkl',
                'heatmap_3d_test': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_test.pkl',
                'heatmap_limb_train': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_train.pkl',
                'heatmap_limb_test': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_test.pkl',
                'heatmap_3d_limb_train': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_train.pkl',
                'heatmap_3d_limb_test': '/media/osero/SamsungSSD/features/bsign22_heatmap_format_full_test.pkl',
                'dino_left_small_finetuned_train': '/media/osero/SamsungSSD/features/features_left_hand_frames_small_finetuned_train',
                'dino_left_small_finetuned_test': '/media/osero/SamsungSSD/features/features_left_hand_frames_small_finetuned_test',
                'dino_face_big_finetuned_train': '/media/osero/SamsungSSD/features/features_face_frames_big_finetuned_train',
                'dino_face_big_finetuned_test': '/media/osero/SamsungSSD/features/features_face_frames_big_finetuned_test',
                'dino_face_small_trained_train': '/media/osero/SamsungSSD/features/features_face_frames_small_trained_train',
                'dino_face_small_trained_test': '/media/osero/SamsungSSD/features/features_face_frames_small_trained_test',
                'dino_left_small_trained_train': '/media/osero/SamsungSSD/features/features_left_frames_small_trained_train',
                'dino_left_small_trained_test': '/media/osero/SamsungSSD/features/features_left_frames_small_trained_test',
                'dino_left_small_trained_mixed_train': '/media/osero/SamsungSSD/features/features_left_frames_small_trained_mixed_train',
                'dino_left_small_trained_mixed_test': '/media/osero/SamsungSSD/features/features_left_frames_small_trained_mixed_test',
                'dino_right_small_trained_mixed_train': '/media/osero/SamsungSSD/features/features_right_frames_small_trained_mixed_train',
                'dino_right_small_trained_mixed_test': '/media/osero/SamsungSSD/features/features_right_frames_small_trained_mixed_test',
                'dino_left_large_trained_mixed_train': '/media/osero/SamsungSSD/features/features_left_frames_large_trained_mixed_train',
                'dino_left_large_trained_mixed_test': '/media/osero/SamsungSSD/features/features_left_frames_large_trained_mixed_test',
                'dino_right_large_trained_mixed_train': '/media/osero/SamsungSSD/features/features_right_frames_large_trained_mixed_train',
                'dino_right_large_trained_mixed_test': '/media/osero/SamsungSSD/features/features_right_frames_large_trained_mixed_test',
                'dino_face_large_trained_mixed_train': '/media/osero/SamsungSSD/features/features_face_frames_large_trained_mixed_train',
                'dino_face_large_trained_mixed_test': '/media/osero/SamsungSSD/features/features_face_frames_large_trained_mixed_test',
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

def save_model_result(model, current_time, input_dim, num_classes, avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list, test_prediction_results):
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
                'avg_test_loss_list': avg_test_loss_list,
                'test_prediction_results': test_prediction_results}
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