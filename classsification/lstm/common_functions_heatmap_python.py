from common_functions_python import *
from pyskl_lib import *
import copy as cp
from torchvision.models.video import r3d_18  # A ResNet-18 pre-trained for 3D inputs

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*copy constructor.*")
warnings.filterwarnings("ignore", category=UserWarning)

class CustomImageHeatmapDataset(Dataset):
    def __init__(self, feature_folder_path_list, heatmap_pickle_name):

        all_files_list = []
        paths_list = []
        labels_list = []
        for feature_folder_path in feature_folder_path_list:
            all_files = []
            sorted_all_files = []
            for dirpath, dirnames, filenames in os.walk(feature_folder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    all_files.append(file_path)
                    # Sort files alphabetically
            for file_path in sorted(all_files):
                sorted_all_files.append(file_path)
            all_files_list.append(sorted_all_files)

            if (len(all_files_list) == 1):
                for file_path in sorted_all_files:
                    label = file_path.split("/")[-3]
                    paths_list.append(file_path)
                    labels_list.append(label)

        self.feature_folder_path_list = feature_folder_path_list
        self.all_files_list = all_files_list
        self.paths = paths_list
        self.classes = np.unique(labels_list)
        label_dict = create_label_dict(self.classes)
        self.labels = [label_dict[x] for x in labels_list]

        ## Keypoint heatmaps
        heatmap_pickle_file = open(heatmap_pickle_name, 'rb')
        annotations = pickle.load(heatmap_pickle_file)
        self.heatmap_features = annotations

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        splited_paths = self.paths[idx].split('/')
        label_name = splited_paths[-3]
        sample_name = splited_paths[-2]
        dot_index = sample_name.find('.')

        if dot_index != -1:  # Check if there's a dot in the string, then remove it
            sample_name = sample_name[:dot_index]

        features_list = []
        for all_files in self.all_files_list:
            pickle_file_name = all_files[idx]
            pickle_file = open(pickle_file_name, 'rb')
            features = pickle.load(pickle_file)
            features_list.append(features)
        check_feature_lenghts(features_list)

        active_frame_indices = get_active_frames(label_name, sample_name)
        active_frame_indices = (
            active_frame_indices
            if active_frame_indices.size > 10
            else np.arange(0, len(features_list[0]))
        )

        embeddings_list = []
        if (config_file['concatenate']):
            ## Dino Deephand embeddings
            for features in features_list:
                embeddings = [features[i] for i in active_frame_indices]
                embeddings = embeddings[0::config_file['frame_frequency']]
                embeddings_list.append(embeddings)
        else:
            for features in features_list:
                embeddings = [features[i] for i in active_frame_indices]
                embeddings = embeddings[0::config_file['frame_frequency']]
                # Find the first person with age 25
                same_len_embeddings = next((embeddings_element for embeddings_element in embeddings_list if len(embeddings_element[0]) == len(embeddings[0])), None)

                if same_len_embeddings:
                    same_len_embeddings = np.add(same_len_embeddings, embeddings)
                else:
                    embeddings_list.append(embeddings)
        
        concatenate_embeddings = np.concatenate(embeddings_list, axis=1)
        np_stacked_array = np.stack(concatenate_embeddings)
        embeddings_tensor = torch.from_numpy(np_stacked_array)

        ## Keypoint heatmaps embeddings
        copy_heatmap = cp.deepcopy(self.heatmap_features[idx])
        copy_heatmap['keypoint'] = copy_heatmap['keypoint'][:,:, :KEYPOINT_NUMBER, :]
        copy_heatmap['keypoint_score'] = copy_heatmap['keypoint_score'][:,:, :KEYPOINT_NUMBER]
        if(any('limb' in s for s in config_file['datasets'])):
            keypoint_heatmaps1 = get_pseudo_heatmap(copy_heatmap, flag='limb')
        else:
            keypoint_heatmaps1 = get_pseudo_heatmap(copy_heatmap)
        keypoint_heatmaps2 = combine_heatmaps_list(keypoint_heatmaps1)
        keypoint_heatmaps3 = [keypoint_heatmaps2[i] for i in active_frame_indices]
        keypoint_heatmaps = keypoint_heatmaps3[0::config_file['frame_frequency']]
        np_stacked_array = np.stack(keypoint_heatmaps)
        keypoints_tensor = torch.from_numpy(np_stacked_array)

        return embeddings_tensor, keypoints_tensor, len(np_stacked_array), self.labels[idx] 

# Step 2: Collate function
def heatmap_collate_fn(batch):
    feature_sequences, heatmap_sequence, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)

    # Pad sequences to the maximum length in the batch
    feature_padded_sequences = pad_sequence([torch.tensor(seq) for seq in feature_sequences], batch_first=True)

    # Sort by lengths in descending order
    sorted_lengths, sorted_indices = lengths.sort(descending=True)
    feature_sorted_sequences = feature_padded_sequences[sorted_indices]
    sorted_labels = labels[sorted_indices]


    # Pad sequences to the maximum length in the batch
    heatmap_padded_sequences = pad_sequence([torch.tensor(seq) for seq in heatmap_sequence], batch_first=True)
    heatmap_orted_sequences = heatmap_padded_sequences[sorted_indices]

    return feature_sorted_sequences,heatmap_orted_sequences, sorted_lengths, sorted_labels

def create_heatmap_data_loaders(datasets):
    heatmap_dataset_list = [s for s in datasets if 'heatmap' in s]
    feature_dataset_list = [s for s in datasets if 'heatmap' not in s]

    train_file_names = []
    test_file_names = []
    for feature_dataset_name in feature_dataset_list:
        train_file_names.append(dataset_table[feature_dataset_name + '_train'])
        test_file_names.append(dataset_table[feature_dataset_name + '_test'])

    heatmap_train_file_name = dataset_table[heatmap_dataset_list[0] + '_train']
    heatmap_test_file_name =  dataset_table[heatmap_dataset_list[0] + '_test']

    train_dataset = CustomImageHeatmapDataset(feature_folder_path_list = train_file_names, heatmap_pickle_name = heatmap_train_file_name)
    test_dataset = CustomImageHeatmapDataset(feature_folder_path_list = test_file_names,  heatmap_pickle_name = heatmap_test_file_name)
    train_loader = DataLoader(train_dataset, batch_size=config_file['batch_size'], shuffle=True, collate_fn=heatmap_collate_fn, num_workers=config_file['num_workers']) # Adjust batch size as needed
    test_loader = DataLoader(test_dataset, batch_size=config_file['batch_size'], shuffle=True, collate_fn=heatmap_collate_fn, num_workers=config_file['num_workers'])
    return train_loader, test_loader
    
class VideoClassifierHeatmap2DLSTM(nn.Module):
    def __init__(self, extra_input_dim, hidden_dim, num_layers, output_dim, fc_dropout, lstm_dropout, bidirectional_lstm):
        super(VideoClassifierHeatmap2DLSTM, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        input_dim = self.cnn.fc.in_features + extra_input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = lstm_dropout, bidirectional = bidirectional_lstm)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.cnn.fc = nn.Identity()  # Remove final FC layer of ResNet

    def forward(self, dino_feature, heatmap, lengths):
        batch_size, time_steps, height, width = heatmap.size()
        heatmap = heatmap.view(batch_size * time_steps, 1, height, width)
        
        # Feature extraction
        cnn_features = self.cnn(heatmap)
        cnn_features = cnn_features.view(batch_size, time_steps, -1)  # Reshape for LSTM
        features = torch.cat((cnn_features, dino_feature), dim=2)
        packed_data = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=True)

        _, (hidden, _) = self.lstm(packed_data)  # Use last hidden state
        output = self.dropout(hidden[-1])
        output = self.fc(output)  # Take hidden state of the last LSTM layer
        return output
           
class VideoClassifierHeatmap3DLSTM(nn.Module):
    def __init__(self, extra_input_dim, hidden_dim, num_layers, output_dim, fc_dropout, lstm_dropout, bidirectional_lstm):
        super(VideoClassifierHeatmap3DLSTM, self).__init__()
        print('heatmap 3d')
        self.cnn = r3d_18(pretrained=True)  # Use pre-trained 3D ResNet-18
        self.cnn.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.lstm = nn.LSTM(extra_input_dim, hidden_dim, num_layers, batch_first=True, dropout = lstm_dropout, bidirectional = bidirectional_lstm)
        class_input_dim = self.cnn.fc.in_features + hidden_dim
        self.fc = nn.Linear(class_input_dim, output_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.cnn.fc = nn.Identity()  # Remove final FC layer of ResNet

    def forward(self, dino_feature, heatmap, lengths):
        # Feature extraction
        input_tensor = heatmap.unsqueeze(1)  # Shape becomes [8, 1, 27, 64, 64]
        cnn_features = self.cnn(input_tensor)
        packed_data = pack_padded_sequence(dino_feature, lengths, batch_first=True, enforce_sorted=True)
        _, (hidden, _) = self.lstm(packed_data)  # Use last hidden state
        features = torch.cat((cnn_features, hidden[-1]), dim=1)
        output = self.dropout(features)
        output = self.fc(output)  # Take hidden state of the last LSTM layer
        return output
    
def create_heatmap_model(input_dim, output_dim):
    if (any('heatmap_3d' in s for s in config_file['datasets'])):
        model = VideoClassifierHeatmap3DLSTM(extra_input_dim=input_dim, hidden_dim=config_file['hidden_dim'], num_layers=config_file['num_layers'],
                                output_dim=output_dim, fc_dropout=config_file['mlp_dropout'], lstm_dropout=config_file['lstm_dropout'],
                                bidirectional_lstm=config_file['bidirectional_lstm'])
    else:
        model = VideoClassifierHeatmap2DLSTM(extra_input_dim=input_dim, hidden_dim=config_file['hidden_dim'], num_layers=config_file['num_layers'],
                        output_dim=output_dim, fc_dropout=config_file['mlp_dropout'], lstm_dropout=config_file['lstm_dropout'],
                        bidirectional_lstm=config_file['bidirectional_lstm'])
    model = model.to(device)
    return model


def test_model_heatmap(test_loader, model, criterion):
    correct = 0
    top_5_correct = 0
    total = 0
    running_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    test_predicted = []
    test_labels = []
    test_predicted_top_5 = []

    with torch.no_grad():
        for (features, heatmap_features, lengths, labels) in test_loader:

            features = features.to(device)
            heatmap_features = heatmap_features.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(features, heatmap_features, lengths)
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
            test_predicted_top_5 += (predicted_top_5.cpu().numpy().tolist())

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    top_5_accuracy = 100 * top_5_correct / total
    print(f'Accuracy of the network on the {len(test_loader.dataset)} test video: {accuracy:.4f} %, top5: {top_5_accuracy:.4f} %, avg_loss: {avg_loss}')
    return accuracy, top_5_accuracy, avg_loss, (test_predicted, test_labels, test_predicted_top_5)

def train_loop_heatmap():
    train_loader, test_loader = create_heatmap_data_loaders(datasets = config_file['datasets'])

    input_dim = train_loader.dataset[0][0][0].shape[0] # Get input dimension from a single feature from a video
    cnn_dim = train_loader.dataset[0][1][0].shape # Get input dimension from a single feature from a video
    num_classes = len(set(train_loader.dataset.classes))
    print("datasets: ", config_file['datasets'])
    print("input_dim: ", input_dim, " num_classes: ", num_classes)
    print("cnn_dim: ", cnn_dim)
    print("train_dataset size: ", len(train_loader.dataset))
    print("test_dataset size: ", len(test_loader.dataset))
    print('train_loader feature_folder_path_list: ', train_loader.dataset.feature_folder_path_list)
    print('test_loader feature_folder_path_list: ', test_loader.dataset.feature_folder_path_list)
    model = create_heatmap_model(input_dim, num_classes)

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
        for (features, heatmap_features, lengths, labels) in loop:

            features = features.to(device)
            heatmap_features = heatmap_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features, heatmap_features, lengths)
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
        avg_test_accuracy, avg_top5_test_accuracy, avg_test_loss, test_prediction_results = test_model_heatmap(test_loader, model, criterion)

        avg_loss_list.append(avg_loss)
        avg_accuracy_list.append(avg_accuracy)
        avg_test_accuracy_list.append(avg_test_accuracy)
        avg_top5_test_accuracy_list.append(avg_top5_test_accuracy)
        avg_test_loss_list.append(avg_test_loss)

    # plot_result(avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list)
    current_time = get_current_time()
    save_model_result(model, current_time, input_dim, num_classes, avg_accuracy_list, avg_test_accuracy_list, avg_top5_test_accuracy_list, avg_loss_list, avg_test_loss_list, test_prediction_results)

    del train_loader
    del test_loader
    del model
    del criterion
    del optimizer
    del scheduler