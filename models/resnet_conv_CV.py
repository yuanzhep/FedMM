import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from MIL_ElasticNet_v3_5 import MIL_ElasticNet
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename="features_extraction.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
                    
logger = logging.getLogger()

logger.setLevel(logging.DEBUG)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock_Baseline(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def resnet34_baseline(pretrained=False):
    model = ResNet_Baseline(BasicBlock_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet34')
    return model

def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model
  
def main():
    model_classes = [models.resnet18, models.resnet34, models.resnet50, models.resnet101, models.resnet152]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    root_dir = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0304_code/wsi_256/img/'
    folders = os.listdir(root_dir)
    output_dir = '/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0304_code/output/'
    os.makedirs(output_dir, exist_ok=True)
    labels_df = pd.read_csv('/a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/dataset_tcga_1047/TCGA.csv')  
    folder_to_label = labels_df.set_index('folder_name')['label'].to_dict()

    auc_results = {"model": [], "layer": [], "mean_auc": [], "std_auc": []}

    for model_class in model_classes:
        model = model_class(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])

        for i, layer in enumerate(list(model.children())):
            if not isinstance(layer, torch.nn.modules.conv.Conv2d):
                continue

            model = torch.nn.Sequential(*list(model.children())[:i+1])

            for folder in folders:
                features_list = []
                labels = []
                folder_path = os.path.join(root_dir, folder)

                if os.path.isdir(folder_path):
                    images = os.listdir(folder_path)

                    for image_name in images:
                        if image_name.endswith('.jpg'):
                            image_path = os.path.join(folder_path, image_name)
                            image = Image.open(image_path)
                            input_tensor = preprocess(image)
                            input_batch = input_tensor.unsqueeze(0)

                            with torch.no_grad():
                                features = model(input_batch)
                            features_flat = torch.flatten(features, start_dim=1)
                            features_array = features_flat.squeeze().numpy()
                            features_list.append(features_array)

                            labels.append(folder_to_label[folder])

                    features_np = np.array(features_list)
                    labels_np = np.array(labels)
                    
                    cv = StratifiedKFold(n_splits=5)
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0, 1, 100)
                    fig, ax = plt.subplots()
                                    
                    for train, test in cv.split(features_np, labels_np):
                        model = MIL_ElasticNet()  
                        model.fit(features_np[train], labels_np[train])  
                        probas_ = model.predict_proba(features_np[test])  
                        fpr, tpr, _ = roc_curve(labels_np[test], probas_[:, 1])
                        tprs.append(np.interp(mean_fpr, fpr, tpr))
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)

                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = auc(mean_fpr, mean_tpr)
                    std_auc = np.std(aucs)

                    auc_results["model"].append(model_class.__name__)
                    auc_results["layer"].append(i)
                    auc_results["mean_auc"].append(mean_auc)
                    auc_results["std_auc"].append(std_auc)

                    ax.plot(mean_fpr, mean_tpr,
                            label='ROC for layer %d (AUC = %0.2f ± %0.2f)' % (i, mean_auc, std_auc),
                            lw=2, alpha=0.8)

                    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                            label='Chance', alpha=0.8)

                    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                           title=f'ROC for {model_class.__name__}, Layer {i}')
                    ax.legend(loc="lower right")
                    plt.savefig(f'{output_dir}/ROC_{model_class.__name__}_layer_{i}.pdf')
                    plt.show()

                    logger.info(f"ROC curve and AUC for model: {model_class.__name__}, layer {i}")
                    logger.info(f"AUC: {mean_auc} ± {std_auc}")

                    features_df = pd.DataFrame(features_list)
                    output_csv_path = os.path.join(output_dir, f'{folder}_{model_class.__name__}_layer_{i}.csv')
                    features_df.to_csv(output_csv_path, index=False)
                    logger.info(f"Saved features of folder: {folder} to {output_csv_path}")

    auc_df = pd.DataFrame(auc_results)
    print(auc_df)
    auc_csv_path = os.path.join(output_dir, 'val_auc.csv')
    auc_df.to_csv(auc_csv_path, index=False)
    logger.info(f"Saved AUC results to {auc_csv_path}")

if __name__ == "__main__":
    main()


