from common_functions_python import *
from pyskl_lib import *
import copy as cp
from torchvision.models.video import r3d_18  # A ResNet-18 pre-trained for 3D inputs

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*copy constructor.*")
warnings.filterwarnings("ignore", category=UserWarning)

class CustomImageHeatmapDataset(Dataset):
    def __init__(self, heatmap_pickle_name):

        ## DINO / Deephand features
        paths_list = []
        labels_list = []

        ## Keypoint heatmaps
        heatmap_pickle_file = open(heatmap_pickle_name, 'rb')
        annotations = pickle.load(heatmap_pickle_file)
        annotation_labels = [x['label']+1 for x in annotations]
        annotation_paths = [x['frame_dir'] for x in annotations]

        paths_list.append(annotation_paths)
        all_feature_list = annotations
        labels_list.append(annotation_labels)
        check_labels(labels_list)
        check_feature_lenghts(all_feature_list)

        self.paths = paths_list[0]
        self.classes = np.unique(labels_list[0])
        label_dict = create_label_dict(self.classes)

        self.heatmap_features = annotations
        self.all_feature_list = all_feature_list
        self.labels = [label_dict[x] for x in labels_list[0]]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        splited_paths = self.paths[idx].split('/')
        label_name = splited_paths[-2]
        sample_name = splited_paths[-1]
        dot_index = sample_name.find('.')

        if dot_index != -1:  # Check if there's a dot in the string, then remove it
            sample_name = sample_name[:dot_index]

        active_frame_indices = get_active_frames(label_name, sample_name)
        active_frame_indices = (
            active_frame_indices
            if active_frame_indices.size > 10
            else np.arange(0, self.all_feature_list[idx]['total_frames'])
        )

        ## Keypoint heatmaps embeddings
        keypoint_heatmaps1 = get_pseudo_heatmap(cp.deepcopy(self.heatmap_features[idx]))
        keypoint_heatmaps2 = combine_heatmaps_list(keypoint_heatmaps1)
        keypoint_heatmaps3 = [keypoint_heatmaps2[i] for i in active_frame_indices]
        keypoint_heatmaps = keypoint_heatmaps3[0::config_file['frame_frequency']]
        np_stacked_array = np.stack(keypoint_heatmaps)
        keypoints_tensor = torch.from_numpy(np_stacked_array)

        return keypoints_tensor, len(np_stacked_array), self.labels[idx] 

# Step 2: Collate function
def only_heatmap_collate_fn(batch):
    heatmap_sequence, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)

    # Sort by lengths in descending order
    sorted_lengths, sorted_indices = lengths.sort(descending=True)
    sorted_labels = labels[sorted_indices]


    # Pad sequences to the maximum length in the batch
    heatmap_padded_sequences = pad_sequence([torch.tensor(seq) for seq in heatmap_sequence], batch_first=True)
    heatmap_orted_sequences = heatmap_padded_sequences[sorted_indices]

    return heatmap_orted_sequences, sorted_lengths, sorted_labels

def create_only_heatmap_data_loaders(datasets):
    heatmap_dataset_list = [s for s in datasets if 'heatmap' in s]
    feature_dataset_list = [s for s in datasets if 'heatmap' not in s]

    train_file_names = []
    test_file_names = []
    for feature_dataset_name in feature_dataset_list:
        train_file_names.append(dataset_table[feature_dataset_name + '_train'])
        test_file_names.append(dataset_table[feature_dataset_name + '_test'])

    heatmap_train_file_name = dataset_table[heatmap_dataset_list[0] + '_train']
    heatmap_test_file_name =  dataset_table[heatmap_dataset_list[0] + '_test']

    train_dataset = CustomImageHeatmapDataset(heatmap_pickle_name = heatmap_train_file_name)
    test_dataset = CustomImageHeatmapDataset(heatmap_pickle_name = heatmap_test_file_name)
    train_loader = DataLoader(train_dataset, batch_size=config_file['batch_size'], shuffle=True, collate_fn=only_heatmap_collate_fn, num_workers=config_file['num_workers']) # Adjust batch size as needed
    test_loader = DataLoader(test_dataset, batch_size=config_file['batch_size'], shuffle=True, collate_fn=only_heatmap_collate_fn, num_workers=config_file['num_workers'])
    return train_loader, test_loader
    
class VideoClassifierHeatmapLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_dim, fc_dropout, lstm_dropout, bidirectional_lstm):
        super(VideoClassifierHeatmapLSTM, self).__init__()

        if (any('heatmap_3d' in s for s in config_file['datasets'])):
            self.cnn = r3d_18(pretrained=True)  # Use pre-trained 3D ResNet-18
            self.cnn.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            hidden_dim = self.cnn.fc.in_features
        else:
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            input_dim = self.cnn.fc.in_features

            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = lstm_dropout, bidirectional = bidirectional_lstm)

        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.cnn.fc = nn.Identity()  # Remove final FC layer of ResNet

    def forward(self, heatmap, lengths):
        if (any('heatmap_3d' in s for s in config_file['datasets'])):
            input_tensor = heatmap.unsqueeze(1)  # Shape becomes [16, 58, 1, 64, 64]
            cnn_features = self.resnet3d(input_tensor)
            output = self.dropout(cnn_features)
        else:
            batch_size, time_steps, height, width = heatmap.size()
            heatmap = heatmap.view(batch_size * time_steps, 1, height, width)
            
            # Feature extraction
            cnn_features = self.cnn(heatmap)
            cnn_features = cnn_features.view(batch_size, time_steps, -1)  # Reshape for LSTM
            features = cnn_features
            packed_data = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=True)

            _, (hidden, _) = self.lstm(packed_data)  # Use last hidden state
            output = self.dropout(hidden[-1])

        output = self.fc(output)  # Take hidden state of the last LSTM layer
        return output
       
def create_only_heatmap_model(output_dim):
    model = VideoClassifierHeatmapLSTM(hidden_dim=config_file['hidden_dim'], num_layers=config_file['num_layers'],
                                output_dim=output_dim, fc_dropout=config_file['mlp_dropout'], lstm_dropout=config_file['lstm_dropout'],
                                bidirectional_lstm=config_file['bidirectional_lstm'])
    model = model.to(device)
    return model


def test_model_only_heatmap(test_loader, model, criterion):
    correct = 0
    top_5_correct = 0
    total = 0
    running_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    test_predicted = []
    test_labels = []

    with torch.no_grad():
        for (heatmap_features, lengths, labels) in test_loader:

            heatmap_features = heatmap_features.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(heatmap_features, lengths)
            loss = criterion(outputs, labels)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.topk(outputs.data, 1)
            _, predicted_top_5 = torch.topk(outputs.data, 5)
            
            running_total = len(labels)
            running_correct = (predicted.flatten() == labels.flatten()).sum().item()
            # running_correct2 = sum([(predicted[i] == labels[i]).any().item() for i in range(len(labels))])
            running_top_5_correct = sum([(predicted_top_5[i] == labels[i]).any().item() for i in range(len(labels))])
            running_loss += loss.item()

            total += running_total
            correct += running_correct
            top_5_correct += running_top_5_correct

            test_labels += (labels.cpu().numpy().tolist())
            test_predicted += (predicted.cpu().numpy().tolist())

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    top_5_accuracy = 100 * top_5_correct / total
    print(f'Accuracy of the network on the {len(test_loader.dataset)} test video: {accuracy:.4f} %, top5: {top_5_accuracy:.4f} %, avg_loss: {avg_loss}')
    return accuracy, top_5_accuracy, avg_loss

def train_loop_only_heatmap():
    train_loader, test_loader = create_only_heatmap_data_loaders(datasets = config_file['datasets'])

    cnn_dim = train_loader.dataset[0][0][0].shape # Get input dimension from a single feature from a video
    num_classes = len(set(train_loader.dataset.classes))
    print("datasets: ", config_file['datasets'])
    print("num_classes: ", num_classes)
    print("cnn_dim: ", cnn_dim)
    print("train_dataset size: ", len(train_loader.dataset))
    print("test_dataset size: ", len(test_loader.dataset))
    model = create_only_heatmap_model(num_classes)

    criterion, optimizer, scheduler = create_train_dependencies(model)

    # print(f"lr {config_file['lr']}, step_size: {config_file['step_size']}, gamma: {config_file['gamma']}, weight_decay: {config_file['weight_decay']}")
    # print(f"Model hidden_dim {config_file['hidden_dim']}, num_layers: {config_file['num_layers']}, bidirectional_lstm: {config_file['bidirectional_lstm']}")
    # print(f"batch_size {config_file['batch_size']}, frame_frequency: {config_file['frame_frequency']}")

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
        for (heatmap_features, lengths, labels) in loop:

            heatmap_features = heatmap_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(heatmap_features, lengths)
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
        avg_test_accuracy, avg_top5_test_accuracy, avg_test_loss = test_model_only_heatmap(test_loader, model, criterion)

        avg_loss_list.append(avg_loss)
        avg_accuracy_list.append(avg_accuracy)
        avg_test_accuracy_list.append(avg_test_accuracy)
        avg_top5_test_accuracy_list.append(avg_top5_test_accuracy)
        avg_test_loss_list.append(avg_test_loss)

    # plot_result(avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list)
    current_time = get_current_time()
    save_model_result(model, current_time, 0, num_classes, avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list)

    del train_loader
    del test_loader
    del model
    del criterion
    del optimizer
    del scheduler