from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from math import pow, floor
import math


def img_show(img):
    plt.imshow(img)
    plt.show()


slope = Image.open(input("the path of slope data:"))
slope = np.array(slope)
slope = slope / 90   # 归一化


dem = Image.open(input("the path of dem data:"))
dem = np.array(dem)
# 归一化
for i in range(dem.shape[0]):
    # print('process:{:.1%}'.format(i/(dem.shape[0]-1)))   #查看进度
    for j in range(dem.shape[1]):
        if dem[i,j] != 0:
            dem[i,j] = (dem[i,j] - dem.min())/(dem.max()-dem.min())



img = Image.open(input("the path of img data:"))
img = np.array(img)
img = img/255.0   # 归一化


pca = Image.open(input("the path of PCA data:"))
pca = np.array(pca)
pca = pca/255.0


mask = Image.open(input("the path of mask data:"))
mask = np.array(mask)
mask = mask/255.0   # 归一化



#-------------------Dataset-------------------#
slope = slope.reshape(slope.shape[0],slope.shape[1],1)
dem = dem.reshape(dem.shape[0],dem.shape[1],1)

X = []
Y = []
for i in tqdm(range(dem.shape[0])):
    for j in range(dem.shape[1]):
        if dem[i,j] != [0]:   #利用dem来制作掩膜
            #影像
            temp_img_1 = img[i-1:i+1+1,j-1:j+1+1]
            temp_img_3 = img[i-3:i+3+1:3,j-3:j+3+1:3]
            #pca
            temp_pca_1 = pca[i-1:i+1+1,j-1:j+1+1]
            temp_pca_3 = pca[i-3:i+3+1:3,j-3:j+3+1:3]
            #dem
            temp_dem_1 = dem[i-1:i+1+1,j-1:j+1+1]
            temp_dem_3 = dem[i-3:i+3+1:3,j-3:j+3+1:3]
            #slope
            temp_slope_1 = slope[i-1:i+1+1,j-1:j+1+1]
            temp_slope_3 = slope[i-3:i+3+1:3,j-3:j+3+1:3]

            temp_x = np.concatenate([temp_img_1,temp_img_3,temp_pca_1,temp_pca_3,temp_dem_1,temp_dem_3,temp_slope_1,temp_slope_3],axis=-1)
            temp_x = temp_x.reshape(1,144)

            temp_y = np.array([mask[i,j]])

            X.append(temp_x)
            Y.append(temp_y)


# 划分数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.7, random_state=0)


batch_size = 1000
epochs = 20
device = 'cuda:1'
lr = 1e-4



#-----------------------------------MyDataset-----------------------------------#
class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.x)
        return self.filelength

    def __getitem__(self, idx):
        img = self.x[idx]
        img_transformed = torch.Tensor(img)

        label = self.y[idx]
        label_transformed = torch.Tensor(label)

        return img_transformed, label_transformed

train_data = MyDataset(Xtrain, Ytrain, transform=None)
test_data = MyDataset(Xtest, Ytest, transform=None)
all_data = MyDataset(X, Y, transform=None)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
all_data_loader = DataLoader(dataset = all_data, batch_size=batch_size, shuffle=False)

print(len(train_data), len(train_loader))
print(len(test_data), len(test_loader))
print(len(all_data), len(all_data_loader))



##-----------------------------------MultiheadSelfAttention-----------------------------------#
def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.heads = heads

        dim_head = default(dim_head, dim // heads)

        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context = None, **kwargs):
        b, n, d, d_h, h = *x.shape, self.dim_head, self.heads

        queries = self.to_q(x)
        keys = self.to_k(x)
        values = self.to_v(x) if not self.share_kv else keys


        queries = queries.reshape(b, n, h, -1).transpose(1, 2)
        merge_key_values = lambda t: t.reshape(b, n, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

#-----------------------------------Model-----------------------------------#
class Vgg16Net(nn.Module):
    def __init__(self):
        super(Vgg16Net, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(144,144),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.attn1 = MultiheadSelfAttention(64)
        self.In = nn.InstanceNorm1d(64)

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.attn2 = MultiheadSelfAttention(64)


        self.layer3 = nn.Sequential(
            nn.Conv1d(64,64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.attn3 = MultiheadSelfAttention(64)


        self.layer4 = nn.Sequential(
            nn.Conv1d(64,64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.attn4 = MultiheadSelfAttention(64)


        self.fc = nn.Sequential(
            nn.Linear(64*9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer0(x)
        # stage1
        x = self.layer1(x)
        # x = x.permute(0, 2, 1)
        # x = self.attn1(x)
        # x = x.permute(0, 2, 1)

        # stage2
        x = self.layer2(x)
        # x = x.permute(0, 2, 1)
        # x = self.attn2(x)
        # x = x.permute(0, 2, 1)

        # stage3
        x = self.layer3(x)
        x = x.permute(0, 2, 1)
        x = self.attn3(x)
        x = x.permute(0, 2, 1)

        # stage4
        x = self.layer4(x)
        x = x.permute(0, 2, 1)
        x = self.attn4(x)
        x = x.permute(0, 2, 1)

        # fc
        x = x.reshape(-1, 64*9)
        x = self.fc(x)
        return x


model = Vgg16Net().to(device)


#-----------------------------------训练-----------------------------------#
class FocalLoss_FBJ(nn.Module):
    def __init__(self, alpha=0.7, gamma=2):
        super(FocalLoss_FBJ, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, input, target):
        y_pred = input
        y_true = target
        y_pred = torch.clip(y_pred, 1e-7, 1.0)
        pt_1 = torch.where(torch.eq(y_true, torch.ones_like(y_true)), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, torch.zeros_like(y_true)), y_pred, torch.zeros_like(y_pred))
        loss = -torch.sum(self.alpha * torch.pow(1. - pt_1, self.gamma) * torch.log(pt_1))-torch.sum((1-self.alpha) * torch.pow( pt_0, self.gamma) * torch.log(1. - pt_0))
        return loss
criterion = FocalLoss_FBJ()
optimizer = optim.Adam(model.parameters(), lr=lr)

#设置指数学习率
def adjust_learning_rate(optimizer,epoch):
    init_lrate = 0.0001
    drop = 0.1
    epochs_drop = 5
    lr = init_lrate * pow(drop, floor(epoch) / epochs_drop)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# 训练
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    adjust_learning_rate(optimizer, epoch)
    print('learning rate:',optimizer.state_dict()['param_groups'][0]['lr'])

    plot_step = 0

    for data, label in tqdm(train_loader,desc='Training epoch '+str(epoch+1)):
        data = data.to(device)
        label = label.to(device)


        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output[output>=0.5] = 1.0
        output[output<0.5] = 0.0
        acc = (output == label).float().mean()
        # print(acc)
        TP += ((output == 1) & (label == 1)).float().sum()
        TN += ((output == 0) & (label == 0)).float().sum()
        FP += ((output == 1) & (label == 0)).float().sum()
        FN += ((output == 0) & (label == 1)).float().sum()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)


    Accuracy = ((TP+TN)/(TP+TN+FP+FN)).cpu().detach().numpy()
    Precision = ((TP)/(TP+FP)).cpu().detach().numpy()
    Recall = ((TP)/(TP+FN)).cpu().detach().numpy()
    F1_score = (2*((TP)/(TP+FP))*((TP)/(TP+FN))/((TP)/(TP+FP)+(TP)/(TP+FN))).cpu().detach().numpy()
    print('----Accuracy:',Accuracy,'----Precision:',Precision,'----Recall:',Recall,'----F1-score:',F1_score)


    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        val_TP = 0
        val_TN = 0
        val_FP = 0
        val_FN = 0

        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            val_output[val_output >= 0.5] = 1.0
            val_output[val_output < 0.5] = 0.0
            acc = (val_output == label).float().mean()

            val_TP += ((val_output == 1) & (label == 1)).float().sum()
            val_TN += ((val_output == 0) & (label == 0)).float().sum()
            val_FP += ((val_output == 1) & (label == 0)).float().sum()
            val_FN += ((val_output == 0) & (label == 1)).float().sum()

            epoch_val_accuracy += acc / len(test_loader)
            epoch_val_loss += val_loss / len(test_loader)


    val_Accuracy = ((val_TP+val_TN)/(val_TP+val_TN+val_FP+val_FN)).cpu().detach().numpy()
    val_Precision = ((val_TP)/(val_TP+val_FP)).cpu().detach().numpy()
    val_Recall = ((val_TP)/(val_TP+val_FN)).cpu().detach().numpy()
    val_F1_score = (2*((val_TP)/(val_TP+val_FP))*((val_TP)/(val_TP+val_FN))/((val_TP)/(val_TP+val_FP)+(val_TP)/(val_TP+val_FN))).cpu().detach().numpy()
    print('----Accuracy:',val_Accuracy,'----Precision:',val_Precision,'----Recall:',val_Recall,'----F1-score:',val_F1_score)


    # Torch 模型保存与加载
    save_dir = input("model save path:")
    torch.save(model, save_dir)





# 全数据集预测
with torch.no_grad():
    epoch_all_data_accuracy = 0
    epoch_all_data_loss = 0
    all_data_TP = 0
    all_data_TN = 0
    all_data_FP = 0
    all_data_FN = 0
    predict_proba = []
    predict_X = []

    for data, label in tqdm(all_data_loader):
        data = data.to(device)
        label = label.to(device)

        all_data_output = model(data)
        all_data_loss = criterion(all_data_output, label)
        predict_proba.extend(all_data_output)

        all_data_output[all_data_output >= 0.5] = 1.0
        all_data_output[all_data_output < 0.5] = 0.0
        predict_X.extend(all_data_output)

        acc = (all_data_output == label).float().mean()

        all_data_TP += ((all_data_output == 1) & (label == 1)).float().sum()
        all_data_TN += ((all_data_output == 0) & (label == 0)).float().sum()
        all_data_FP += ((all_data_output == 1) & (label == 0)).float().sum()
        all_data_FN += ((all_data_output == 0) & (label == 1)).float().sum()

        epoch_all_data_accuracy += acc / len(all_data_loader)
        epoch_all_data_loss += all_data_loss / len(all_data_loader)

all_data_Accuracy = ((all_data_TP + all_data_TN) / (all_data_TP + all_data_TN + all_data_FP + all_data_FN)).cpu().detach().numpy()
all_data_Precision = ((all_data_TP) / (all_data_TP + all_data_FP)).cpu().detach().numpy()
all_data_Recall = ((all_data_TP) / (all_data_TP + all_data_FN)).cpu().detach().numpy()
all_data_F1_score = (2 * ((all_data_TP) / (all_data_TP + all_data_FP)) * ((all_data_TP) / (all_data_TP + all_data_FN)) / ((all_data_TP) / (all_data_TP + all_data_FP) + (all_data_TP) / (all_data_TP + all_data_FN))).cpu().detach().numpy()
print('----Accuracy:', all_data_Accuracy, '----Precision:', all_data_Precision, '----Recall:', all_data_Recall, '----F1-score:', all_data_F1_score)


