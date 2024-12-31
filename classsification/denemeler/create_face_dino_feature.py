# LOCATION : https://github.com/purnasai/Dino_V2
import torch
from torchvision import models, transforms
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
from time import gmtime, strftime

os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load DINO ViT model from torchvision (for example, ViT small or base model trained with DINO)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
model.eval()

# Define transform to match the input size for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom Dataset for a list of image paths
class ImageDataset(Dataset):
    def __init__(self):
        video_folder = "/media/osero/SamsungSSD/CMPE_SSD/frame-face-c256" 
        full_sample_folder_list = []
        label_list = []
        for label_folder in sorted(os.listdir(video_folder)):
            full_label_folder = os.path.join(video_folder, label_folder)
            label = int(label_folder)
            for sample_folder in sorted(os.listdir(full_label_folder)):
                full_sample_folder = os.path.join(full_label_folder, sample_folder)
                full_sample_folder_list.append(full_sample_folder)
                label_list.append(label)
        self.full_sample_folder_list = full_sample_folder_list
        self.label_list = label_list

    def __len__(self):
        return len(self.full_sample_folder_list)

    def __getitem__(self, idx):
        full_sample_folder = self.full_sample_folder_list[idx]
        image_list = []
        if (self.label_list[idx] < 581):
            return full_sample_folder, image_list, self.label_list[idx]

        for image_file in sorted(os.listdir(full_sample_folder)):
            full_image_file = os.path.join(full_sample_folder, image_file)
            image = Image.open(full_image_file)
            transformed_image = transform(image)
            image_list.append(transformed_image)
        return full_sample_folder, image_list, self.label_list[idx]
    
dataset = ImageDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

video_labels = []  # Populate this with the corresponding labels for each video
video_embeddings = []
video_paths = []
check_counter = 0
with torch.no_grad():  # Disable gradient calculation
    for paths, inputs, labels in dataloader:
        if (labels[0] < 581):
            continue
        print(paths[0])
        print(strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        check_counter += 1
        inputs_stack = torch.stack(inputs).to(device).squeeze(1)
        outputs = model(inputs_stack)
        outputs_numpy = outputs.cpu().numpy().tolist()
        video_paths.append(paths)
        video_embeddings.append(outputs_numpy)
        video_labels.append(labels)
        if (check_counter % 100) == 0:
            with open('features_face_frames_large.pickle', 'wb') as handle:
                pickle.dump((video_paths, video_embeddings, video_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)    

# with open('features_face_frames_large.pickle', 'a') as handle:
#     pickle.dump((video_paths, video_embeddings, video_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)

aaa = 4

