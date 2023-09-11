from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms, datasets, models
import numpy as np
import os
import pandas as pd
import math
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

class CNVFeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(CNVFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
        
def load_cnv_data(path_to_cnv):
    cnv_data_list = []
    cnv_labels_list = []
    for file_name in os.listdir(path_to_cnv):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(path_to_cnv, file_name))
            cnv_data_list.append(df.values.flatten())
            label = int(file_name.split('_')[1].split('.')[0])
            cnv_labels_list.append(label)
    return np.array(cnv_data_list), np.array(cnv_labels_list)

def initialize_prototypes(model, dataset, batch_size):
    prototype = defaultdict(lambda: torch.zeros(256))
    count = defaultdict(int)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for data, labels in dataloader:
        embeddings = model(data)
        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        for label in labels.unique():
            prototype[label] += embeddings_flat[labels == label].sum(dim=0)
            count[label] += (labels == label).sum().item()
    for label, total in count.items():
        prototype[label] /= total
    return prototype
    
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
path_to_images = "/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0802_fl/tcga/224_patch/"
image_dataset = datasets.ImageFolder(root=path_to_images, transform=transform)
path_to_cnv = "/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0802_fl/tcga/256_cnv/"
cnv_data, cnv_labels = load_cnv_data(path_to_cnv)
cnv_dataset = TensorDataset(torch.tensor(cnv_data, dtype=torch.float32), torch.tensor(cnv_labels, dtype=torch.int64))
resnet34 = models.resnet34(pretrained=True)
image_feature_extractor = nn.Sequential(*list(resnet34.children())[:-2])
cnv_feature_extractor = CNVFeatureExtractor(cnv_data.shape[1])
global_prototype_image = initialize_prototypes(image_feature_extractor, image_dataset, batch_size=32)
global_prototype_cnv = initialize_prototypes(cnv_feature_extractor, cnv_dataset, batch_size=32)

random_seeds = [42, 65, 123, 152, 289, 567, 678, 789, 924, 1024, 
                142, 265, 2048, 3023, 3123, 4152, 5289, 6567, 7678, 8789]
alpha = 0.05
t0 = 30
beta = 0.3
local_epochs = 3
batch_size = 32
num_rounds = 100
num_clients = 3

for seed in random_seeds:
    torch.manual_seed(seed)
    total_size = len(image_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    image_train_dataset, image_test_dataset = random_split(image_dataset, [train_size, test_size])
    total_size = len(cnv_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    cnv_train_dataset, cnv_test_dataset = random_split(cnv_dataset, [train_size, test_size])
    for round in range(1, num_rounds + 1):
        local_image_feature_extractors = []
        local_cnv_feature_extractors = []
        for client in range(num_clients):
            local_image_feature_extractor = nn.Sequential(*list(resnet34.children())[:-2])
            local_image_feature_extractor.load_state_dict(image_feature_extractor.state_dict())
            local_cnv_feature_extractor = CNVFeatureExtractor(cnv_data.shape[1])
            local_cnv_feature_extractor.load_state_dict(cnv_feature_extractor.state_dict())
            optimizer_image = optim.SGD(local_image_feature_extractor.parameters(), lr=0.001)
            optimizer_cnv = optim.SGD(local_cnv_feature_extractor.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            for local_epoch in range(local_epochs):
                for data, labels in DataLoader(image_train_dataset, batch_size=batch_size, shuffle=True):
                    local_image_feature_extractor.train()
                    optimizer_image.zero_grad()
                    embeddings = local_image_feature_extractor(data)
                    embeddings_flat = embeddings.view(embeddings.size(0), -1)
                    for label in [0, 1]:
                        embeddings_label = embeddings_flat[labels == label]
                        if len(embeddings_label) > 0:
                            loss1 = torch.norm(embeddings_label - global_prototype_image[label])
                            loss2 = criterion(embeddings_label, labels[labels == label])
                            lambda_t = 1 / (1 + math.exp(-alpha * (round - t0)))
                            loss = lambda_t * (beta / embeddings_label.size(1)) * loss1 + (1 - lambda_t) * loss2
                            loss.backward()
                            optimizer_image.step()
                            global_prototype_image[label] = (global_prototype_image[label] * round + embeddings_label.mean(dim=0)) / (round + 1)

            for local_epoch in range(local_epochs):
                for data, labels in DataLoader(cnv_train_dataset, batch_size=batch_size, shuffle=True):
                    local_cnv_feature_extractor.train()
                    optimizer_cnv.zero_grad()
                    embeddings = local_cnv_feature_extractor(data)
                    for label in [0, 1]:
                        embeddings_label = embeddings[labels == label]
                        if len(embeddings_label) > 0:
                            loss1 = torch.norm(embeddings_label - global_prototype_cnv[label])
                            loss2 = criterion(embeddings_label, labels[labels == label])
                            lambda_t = 1 / (1 + math.exp(-alpha * (round - t0)))
                            loss = lambda_t * (beta / embeddings_label.size(1)) * loss1 + (1 - lambda_t) * loss2
                            loss.backward()
                            optimizer_cnv.step()
                            global_prototype_cnv[label] = (global_prototype_cnv[label] * round + embeddings_label.mean(dim=0)) / (round + 1)

            local_image_feature_extractors.append(local_image_feature_extractor)
            local_cnv_feature_extractors.append(local_cnv_feature_extractor)

        with torch.no_grad():
            for global_params, *local_params_list in zip(image_feature_extractor.parameters(), *[local.parameters() for local in local_image_feature_extractors]):
                for local_params in local_params_list:
                    global_params += local_params / num_clients

            for global_params, *local_params_list in zip(cnv_feature_extractor.parameters(), *[local.parameters() for local in local_cnv_feature_extractors]):
                for local_params in local_params_list:
                    global_params += local_params / num_clients

    image_feature_extractor.eval()
    cnv_feature_extractor.eval()
    y_true_image, y_pred_image, y_score_image = [], [], []
    y_true_cnv, y_pred_cnv, y_score_cnv = [], [], []

    with torch.no_grad():
        for data, labels in DataLoader(image_test_dataset, batch_size=batch_size):
            embeddings = image_feature_extractor(data)
            embeddings_flat = embeddings.view(embeddings.size(0), -1)
            scores = torch.sigmoid(embeddings_flat)
            predictions = (scores > 0.5).int()
            y_true_image.extend(labels.tolist())
            y_pred_image.extend(predictions.tolist())
            y_score_image.extend(scores.tolist())

    with torch.no_grad():
        for data, labels in DataLoader(cnv_test_dataset, batch_size=batch_size):
            embeddings = cnv_feature_extractor(data)
            scores = torch.sigmoid(embeddings)
            predictions = (scores > 0.5).int()
            y_true_cnv.extend(labels.tolist())
            y_pred_cnv.extend(predictions.tolist())
            y_score_cnv.extend(scores.tolist())

    print(f"Image Data Accuracy: {accuracy_score(y_true_image, y_pred_image)}")
    print(f"CNV Data Accuracy: {accuracy_score(y_true_cnv, y_pred_cnv)}")

    fpr, tpr, _ = roc_curve(y_true_image, y_score_image)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title('ROC for Image Data')
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
    fpr, tpr, _ = roc_curve(y_true_cnv, y_score_cnv)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title('ROC for CNV Data')
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    print(f"Round {round} completed.")

print(f"Training with seed {seed} completed.")
