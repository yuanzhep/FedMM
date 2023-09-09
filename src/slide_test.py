import numpy as np
import torch
import torch.nn as nn
import data.loader as loader
import torch.utils.data as Data
import torchvision
import Network.network as network
import matplotlib.pyplot as plt
import Network.NN as NN
from sklearn import metrics

def forward(net,x):
    x = net.Conv2d_1a_3x3(x)
    x = net.Conv2d_2a_3x3(x)
    x = net.Conv2d_2b_3x3(x)
    x = net.maxpool1(x)
    x = net.Conv2d_3b_1x1(x)
    x = net.Conv2d_4a_3x3(x)
    x = net.maxpool2(x)
    x = net.Mixed_5b(x)
    x = net.Mixed_5c(x)
    x = net.Mixed_5d(x)
    x = net.Mixed_6a(x)
    x = net.Mixed_6b(x)
    x = net.Mixed_6c(x)
    x = net.Mixed_6d(x)
    x = net.Mixed_6e(x)
    aux_defined = net.training and net.aux_logits
    if aux_defined:
        aux = net.AuxLogits(x)
    else:
        aux = None
    x = net.Mixed_7a(x)
    x = net.Mixed_7b(x)
    x = net.Mixed_7c(x)
    x = net.avgpool(x)
    x = net.dropout(x)
    feature = torch.flatten(x, 1)
    x = net.fc(feature)
    return x,feature

LR=0.001
# TEST_SIZE = 150
# NUMBER_IMAGE = 15
# EPOCH = 100
NUMBER_IMAGE = 15
EPOCH = 100

label_LUSC = np.load('../data/label/label_LUSC.npy')
label_LUAD = np.load('../data/label/label_LUAD.npy')
slides_LUSC = np.zeros((15,487,3,299,299))
slides_LUAD = np.ones((15,501,3,299,299))
accu = np.zeros(EPOCH)
for i in range(NUMBER_IMAGE):
    slides_LUSC[i] = np.load('../data/image/3/LUSC/image'+str(i)+'.npy')
    slides_LUAD[i] = np.load('../data/image/3/LUAD/image'+str(i)+'.npy')
inceptionV3 = network.get_model_Inceptionv3(2)

param_to_update = []
for names, param in inceptionV3.named_parameters():
    if param.requires_grad == True:
        param_to_update.append(param)
optimizer = torch.optim.Adam(param_to_update,lr= LR)
loss_func = torch.nn.CrossEntropyLoss().cuda()
train_accuracy = np.zeros(EPOCH)
acc = np.zeros(EPOCH)
train_loss = np.zeros(EPOCH)
maxaccuracy = 0
for epoch in range(EPOCH):
    prediction_train = torch.zeros(0)
    label = torch.zeros(0)
    print('epoch:',epoch)
    inceptionV3.train()
    for i in range(NUMBER_IMAGE):
        data_loader = loader.train_loader(slides_LUSC[i],slides_LUAD[i],label_LUSC,label_LUAD,TEST_SIZE)
        for step, (t_x, t_y) in enumerate(data_loader):
            label = torch.cat((label, t_y))
            t_x = t_x.cuda()
            t_y = t_y.cuda()
            output= inceptionV3(t_x)
            # output,feature = forward(inceptionV3,t_x)
            pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
            prediction_train = torch.cat((prediction_train, pred_y.cpu()))
            loss = loss_func(output, t_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    t_accuracy = sum(prediction_train == label) / label.size(0)
    print('train_accuracy:   ',t_accuracy)
    pred = torch.zeros(TEST_SIZE*2)
    inceptionV3.eval()
    with torch.no_grad():
        for i in range(NUMBER_IMAGE):
            test_loader = loader.test_loader(slides_LUSC[i], slides_LUAD[i], label_LUSC, label_LUAD, TEST_SIZE)
            for step, (t_x, t_y) in enumerate(test_loader):
                if step == 0:
                    t_x = t_x.cuda()
                    t_y = t_y.cuda()
                    test_output = inceptionV3(t_x)
                    # test_output, feature = forward(inceptionV3, t_x)
                    predi = torch.max(test_output, 1)[1].data.squeeze()
                    # print(predi)
                    accuarcy = sum(predi == t_y) / t_y.size(0)
                    print(i, '   test accuarcy:%4f ' % accuarcy.cpu().numpy())
            pred += predi.cpu()
        p = pred/15
        auc1 = metrics.roc_auc_score(t_y.cpu(),p.cpu())
        # F1_score1 = metrics.f1_score(t_y.cpu().numpy(), p.cpu())
        for i in range(200):
            if pred[i] > 7:
                pred[i] = 1
            else:
                pred[i] = 0
        accuarcy = sum(pred == t_y.cpu()) / t_y.cpu().size(0)

        auc = metrics.roc_auc_score(t_y.cpu().detach(), pred)
        F1_score = metrics.f1_score(t_y.cpu().numpy(), pred)

        accu[epoch] = accuarcy
        print('test_accuracy ----------------------------------------> ', accuarcy.cpu().numpy(), '  auc :', auc,'  auc1 :', auc1,"       F1-score  :", F1_score)
    # if maxaccuracy<=accuarcy:
    #     maxaccuracy = accuarcy
    #     if maxaccuracy>=0.76:
    #         label_extract_LUAD = torch.from_numpy(label_LUAD).type(torch.LongTensor)
    #         label_extract_LUSC = torch.from_numpy(label_LUSC).type(torch.LongTensor)
    #         for i in range(NUMBER_IMAGE):
    #             print('IMAGE>>>>>>>>>>>>>>>>>',i)
    #             slide_LUAD = torch.from_numpy(slides_LUAD[i]).type(torch.FloatTensor)
    #             slide_LUSC = torch.from_numpy(slides_LUSC[i]).type(torch.FloatTensor)
    #             LUAD_dataset = Data.TensorDataset(slide_LUAD,label_extract_LUAD)
    #             LUSC_dataset = Data.TensorDataset(slide_LUSC,label_extract_LUSC)
    #             LUAD_loader = Data.DataLoader(
    #                 dataset = LUAD_dataset,
    #                 batch_size = len(LUAD_dataset),
    #                 shuffle = False
    #             )
    #             LUSC_loader = Data.DataLoader(
    #                 dataset = LUSC_dataset,
    #                 batch_size = len(LUSC_dataset),
    #                 shuffle = False
    #             )
    #             for step, (t_x, t_y) in enumerate(LUAD_loader):
    #                 output,feature = forward(inceptionV3,t_x.cuda())
    #                 output = torch.max(output, 1)[1].data.squeeze()
    #                 output1 = output.cpu().detach().numpy()
    #                 feature = feature.cpu().detach().numpy()
    #                 print(output1,sum(output1))
    #                 np.save('../data/feature/slide/LUAD/feature_slide_LUAD'+str(i)+'.npy', feature)
    #                 np.save('../data/predict/slide/LUAD/predict_LUAD'+str(i)+'.npy', output1)
    #             for step, (t_x, t_y) in enumerate(LUSC_loader):
    #                 output,feature = forward(inceptionV3,t_x.cuda())
    #                 output = torch.max(output, 1)[1].data.squeeze()
    #                 output2 = output.cpu().detach().numpy()
    #                 feature = feature.cpu().detach().numpy()
    #                 print(output2,sum(output2))
    #                 np.save('../data/feature/slide/LUSC/feature_slide_LUSC'+str(i)+'.npy', feature)
    #                 np.save('../data/predict/slide/LUSC/predict_LUSC'+str(i)+'.npy', output2)
        # torch.save(inceptionV3,'./model/slide_inceptionV3.pkl')
        # torch.save(inceptionV3,'./model/slide_inceptionV3_temp.pkl')

# print(max(accu))
NN.plot(np.array(range(EPOCH)),accu,'slide','epoch','test_accuracy',1.0)


# label_LUAD = torch.from_numpy(label_LUAD).type(torch.LongTensor)
# label_LUSC = torch.from_numpy(label_LUSC).type(torch.LongTensor)
# for i in range(NUMBER_IMAGE):
#     print('IMAGE>>>>>>>>>>>>>>>>>',i)
#     slide_LUAD = torch.from_numpy(slides_LUAD[i]).type(torch.FloatTensor)
#     slide_LUSC = torch.from_numpy(slides_LUSC[i]).type(torch.FloatTensor)
#     LUAD_dataset = Data.TensorDataset(slide_LUAD,label_LUAD)
#     LUSC_dataset = Data.TensorDataset(slide_LUSC,label_LUSC)
#     LUAD_loader = Data.DataLoader(
#         dataset = LUAD_dataset,
#         batch_size = len(LUAD_dataset),
#         shuffle = False
#     )
#     LUSC_loader = Data.DataLoader(
#         dataset = LUSC_dataset,
#         batch_size = len(LUSC_dataset),
#         shuffle = False
#     )
#     for step, (t_x, t_y) in enumerate(LUAD_loader):
#         # output,feature = net(t_x.cuda())
#         output,feature = forward(inceptionV3,t_x.cuda())
#         output = torch.max(output, 1)[1].data.squeeze()
#         output1 = output.cpu().detach().numpy()
#         feature = feature.cpu().detach().numpy()
#         print(output1,sum(output1))
#         np.save('../data/feature/slide/LUAD/feature_slide_LUAD'+str(i)+'.npy', feature)
#         np.save('../data/predict/slide/LUAD/predict_LUAD'+str(i)+'.npy', output1)
#     for step, (t_x, t_y) in enumerate(LUSC_loader):
#         # output,feature = net(t_x.cuda())
#         output,feature = forward(inceptionV3,t_x.cuda())
#         output = torch.max(output, 1)[1].data.squeeze()
#         output2 = output.cpu().detach().numpy()
#         feature = feature.cpu().detach().numpy()
#         print(output2,sum(output2))
#         np.save('../data/feature/slide/LUSC/feature_slide_LUSC'+str(i)+'.npy', feature)
#         np.save('../data/predict/slide/LUSC/predict_LUSC'+str(i)+'.npy', output2)
#     output = np.hstack((output2[:150],output1[:150]))
#     print(output)
#     label = torch.cat((label_LUSC[-150:],label_LUAD[-150:])).numpy()
#     print(type(output),output.shape,label.shape)
#     print(sum(output==label)/300)