# v1.1, 0609_2023, Fine-tune the model for feature extraction
# v1.2, 0610_2023, Plot train/val_loss to check

import os
import torch
import csv
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim import SGD
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

logging.basicConfig(filename="tcga_yz_features_extraction.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def load_labels(csv_file):
    labels_dict = {}
    try:
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  
            for row in reader:
                labels_dict[row[0]] = int(row[1])
    except Exception as e:
        logger.error(f"Error in loading CSV file: {e}")
    return labels_dict
    

class TCGAFolder(ImageFolder):
    def __init__(self, root, transform=None, labels_dict=None):
        super().__init__(root, transform=transform)
        self.imgs = self.samples
        self.labels_dict = labels_dict

    def __getitem__(self, index):
        path, _ = self.samples[index]
        target = os.path.basename(os.path.dirname(path))
        target = self.labels_dict.get(target, 0)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=120):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):  
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:  
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        # print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        model.eval()  
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        # print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
        logger.info(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    return model, val_loss_history, val_acc_history


def plot_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history, num_epochs, model_name):
    plt.figure(figsize=(10,5))
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,num_epochs+1),train_loss_history,label="Train Loss")
    plt.plot(range(1,num_epochs+1),val_loss_history,label="Val Loss")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(f'{model_name}_loss.pdf')

    plt.figure(figsize=(10,5))
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,num_epochs+1),train_acc_history,label="Train Acc")
    plt.plot(range(1,num_epochs+1),val_acc_history,label="Val Acc")
    plt.ylim((0,100.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(f'{model_name}_acc.pdf')


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_dir = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0308_code/yz_img/'
    label_csv = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0308_code/yz_img/tcga_label.csv'
    output_dir = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0308_code/0611_yz/output/'
    os.makedirs(output_dir, exist_ok=True)
    labels_dict = load_labels(label_csv)

    resnet = models.resnet34(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    # Reassemble the pretrained model using Average Pooling
    modules = list(resnet.children())[:-3]  
    modules += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 2)]  
    model = nn.Sequential(*modules)
    model = model.to(device)
    
    # Reassemble the pretrained model using Max Pooling
#    modules = list(resnet.children())[:-3]  
#    modules += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(), nn.Linear(256, 2)]  
#    model = nn.Sequential(*modules)
#    model = model.to(device)

    for param in model[-1].parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = TCGAFolder(root=root_dir, transform=preprocess, labels_dict=labels_dict)
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2)
    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)

    # Dataloaders
    batch_size = 64  
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Train
    num_epochs = 120
    model, val_loss_history, val_acc_history = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=num_epochs)
    plot_history(val_loss_history, val_acc_history, num_epochs, "Model")
    torch.save(model.state_dict(), os.path.join(output_dir, 'tcga_yz_fine_tuned_model.pth'))
    
    feature_model = nn.Sequential(*list(model.children())[:-1])  
    feature_model = feature_model.to(device)
    feature_model.eval()  

    try:
        folders = os.listdir(root_dir)
    except Exception as e:
        logger.error(f"Error in listing directories: {e}")
        return

    for folder in folders:
        logger.info(f"Processing folder: {folder}")
        features_list = []
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):  
            images = os.listdir(folder_path)
            for image_name in images:
                if image_name.endswith('.jpg'):  
                    image_path = os.path.join(folder_path, image_name)
                    try:
                        image = Image.open(image_path)
                        input_tensor = preprocess(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            features = feature_model(input_tensor)
                            features_shape = features.shape
                            logger.info(f"Output shape: {features_shape}")

                        features_flat = torch.flatten(features, start_dim=1)
                        features_array = features_flat.squeeze().cpu().numpy()
                        
                        features_list.append(features_array)
                    except Exception as e:
                        logger.error(f"Error processing image {image_path}: {e}")
                        continue

        features_df = pd.DataFrame(features_list)
        output_csv_path = os.path.join(output_dir, f'tcga_yz_{folder}_features.csv')

        try:
            features_df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved features of folder: {folder} to {output_csv_path}")
        except Exception as e:
            logger.error(f"Error saving features to CSV file {output_csv_path}: {e}")
            continue


if __name__ == "__main__":
    main()
