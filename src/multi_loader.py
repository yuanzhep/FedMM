import torch
import torch.utils.data as Data

BATCH_SIZE = 32
SHUFFLE = True

def train_loader(LUSC_data,LUAD_data,LUSC_label,LUAD_label,test_size):
    train_data = torch.cat((torch.from_numpy(LUSC_data[test_size:]),torch.from_numpy(LUAD_data[test_size:]))).type(torch.FloatTensor)
    train_label = torch.cat((torch.from_numpy(LUSC_label[test_size:]),torch.from_numpy(LUAD_label[test_size:]))).type(torch.LongTensor)
    torch_dataset = Data.TensorDataset(train_data,train_label)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE
        )
    return train_loader

def test_loader(LUSC_data,LUAD_data,LUSC_label,LUAD_label,test_size):
    test_data = torch.cat((torch.from_numpy(LUSC_data[:test_size]),torch.from_numpy(LUAD_data[:test_size]))).type(torch.FloatTensor)
    test_label = torch.cat((torch.from_numpy(LUSC_label[:test_size]),torch.from_numpy(LUAD_label[:test_size]))).type(torch.LongTensor)
    torch_dataset = Data.TensorDataset(test_data,test_label)
    test_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=test_size*2,
        shuffle=False
        )
    return test_loader