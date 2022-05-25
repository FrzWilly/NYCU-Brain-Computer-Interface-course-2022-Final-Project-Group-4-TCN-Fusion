from ctypes import sizeof
from random import shuffle
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay


class EEGBlock(nn.Module):
    def __init__(self, F1 = 8, F2 = 16, D = 2, KE = 32, pe = 0.3):
        super(EEGBlock, self).__init__()

        self.F1 = F1
        self.F2 = F2 
        self.D = D # depth multiplier

        self.KE = KE # kernal size of the first conv
        self.pe = pe # dropout rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.KE), padding='same', bias=False),
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D*self.F1, (22, 1), padding='valid', groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D*self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.pe)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 16), padding=(0, 8), groups=self.D*self.F1, bias=False),
            nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.pe)
        )

        # self.classifier = nn.Linear(16*17, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        #x = x.view(-1, 16*17)
        #x = self.classifier(x)
        #x = self.softmax(x)
        return x2, x3

class TCNBlock(nn.Module):
    def __init__(self, F2=16, dilate_rate=1, KT=4, FT=12, pt=0.3):
        super(TCNBlock, self).__init__()

        self.F2 = F2
        self.dilate_rate = dilate_rate # depth multiplier

        self.KT = KT # kernal size of the first conv
        self.FT = FT
        self.pt = pt # dropout rate

        self.dilated_causal_conv = nn.Sequential(
            nn.Conv1d(self.F2, self.FT, self.KT, dilation=self.dilate_rate, padding_mode='causal', bias=False),
            nn.BatchNorm2d(self.FT),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.pt),

            nn.Conv1d(self.F2, self.FT, self.KT, dilation=self.dilate_rate, padding_mode='causal', bias=False),
            nn.BatchNorm2d(self.FT),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.pt)
        )

        self.conv1d = nn.Conv1d(self.F2, 1, padding='same')

    def forward(self, input):

        x = input

        y = self.dilated_causal_conv(x)
            
        if self.F2 != self.FT:
            conv = self.conv1d(x)
            add = y + conv
        else:
            add = y + x


        return x

class TCNFusion(nn.Module):
    def __init__(self):
        super(EEGBlock, self).__init__()

        self.F1 = 8
        self.F2 = 16
        self.D = 2

        self.block1 = EEGBlock()
        modules = []
        for i in range(self.D):
            modules.append(TCNBlock(dilate_rate=2**i))
        
        self.block2 = nn.Sequential(*modules)

        self.classifier = nn.Linear(16*17, 4, bias=True)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x0, x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = torch.cat(x1, x2)
        x0 = torch.flatten(x0)
        x3 = torch.flatten(x3)

        xfinal = torch.cat(x0, x3)

        
        x = x.view(-1, 16*17)
        x = self.classifier(x)
        #x = self.softmax(x)
        return x

def run_models( 
    models, epoch, batch, learning_rate, 
    optimizer=optim.SGD, loss_func = nn.CrossEntropyLoss(), opt="sgd"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now_running = "eeg"

    path = '../BCICIV_2a_mat/'
    S01E = loadmat(path + 'BCIC_S01_E')
    S1Ex = np.array(S01E['x_test'].squeeze())
    S1Ey = np.array(S01E['y_test'].squeeze())

    Tx = []
    Ty = []
    for idx in range(2, 10):
        S29T = loadmat(path + f'BCIC_S0{idx}_T')
        Tx.append(S29T['x_train'].squeeze())
        Ty.append(S29T['y_train'].squeeze())

    Tx = np.array(Tx)
    Tx = Tx.reshape((-1,) + Tx.shape[2:])
    Ty = np.array(Ty).reshape(-1)

    train_dataset = TensorDataset(torch.Tensor(Tx), torch.Tensor(Ty))
    test_dataset = TensorDataset(torch.Tensor(S1Ex), torch.Tensor(S1Ey))

    train_loader = DataLoader(train_dataset, batch_size = batch, shuffle=True)
    test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=True)

    rec = {
        "train" : [],
        "test" : []
    }

    my_lr_scheduler = []
    idx = 0
    decayRate = 1#0.9999

    optimizer = optimizer(models[now_running].parameters(), lr = learning_rate)
    
    #my_lr_scheduler.append(torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate))
    idx = idx + 1
    last_loss = 10000.0

    for ep in range(epoch):
        train_cr = 0.0
        test_cr = 0.0
        

        for idx, data in enumerate(train_loader):
            x, y = data

            inputs = x.to(device)
            labels = y.to(device).long().view(-1)

            optimizer.zero_grad()

            model = models[now_running]
            model.train()

            outputs = model.forward(inputs[:, None])
            loss = loss_func(outputs, labels)
            loss.backward()

            train_cr += (
                torch.max(outputs, 1)[1] == labels
            ).sum().item()

           
            idx = 0
            optimizer.step()
            # if ep > (epoch/2):
            #     #print("lr decay")
            #     my_lr_scheduler[idx].step()
            #     idx = idx + 1

    #testing
        pred = []
        real = []
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x, y = data
                inputs = x.to(device)
                labels = y.to(device)

                model.eval()
                outputs = model.forward(inputs[:, None])

                test_cr += (
                    torch.max(outputs, 1)[1] == labels.long().view(-1)
                ).sum().item()

                pred += torch.max(outputs, 1)[1].cpu()
                real += labels.long().view(-1).cpu()

        rec["train"] += [(train_cr*100.0)/len(train_dataset)]
        rec["test"] += [(test_cr*100.0)/len(test_dataset)]

        if ep % 100 == 0:
            pred = np.asarray(pred)
            real = np.asarray(real)
            
            print(f"training data epoch {ep} accuracy:", rec["train"][len(rec["train"])-1])
            print(f"testing data epoch {ep} accuracy:", rec["test"][len(rec["test"])-1])
            print("\n")

        if ep == (epoch - 1):

            conf_matrix = confusion_matrix( 
                pred, real, normalize = 'all'
            )
            disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1, 2, 3])
            disp = disp.plot(cmap = plt.cm.Blues)

            plt.savefig(f'confusion_all_{batch}_{learning_rate}_{opt}.png')
            #plt.show()
            conf_matrix = confusion_matrix( 
                pred, real, normalize = 'true'
            )
            disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1, 2, 3])
            disp = disp.plot(cmap = plt.cm.Blues)

            plt.savefig(f'confusion_true_{batch}_{learning_rate}_{opt}.png')
            #plt.show()

    print("\ntraining data final accuracy:", rec["train"][len(rec["train"])-1])
    print("testing data final accuracy:", rec["test"][len(rec["test"])-1])
    print("\n")
    
    #PlotImg('EEGBlock', **rec)

    # inp = input("Do you want to save " + now_running + " parameters ? (y/n)")
    # if(inp == ('y' or 'Y' or 'yes')):
    #     torch.save(models['elu'].state_dict(), now_running+' parameters')
        
    torch.cuda.empty_cache()
    return rec

def plot(title = 'OwO', accline = [75, 80, 85, 87], **kwargs):
    fig = plt.figure(figsize = (10, 5))
    plt.title = title
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    line_color = {
        'train' : 'green',
        'test' : 'pink',
    }

    for label, data in kwargs.items():
        plt.plot(
            range(1, len(data)+1), data, 
            '--' if 'test' in label else ':',
            color = line_color[label], 
            label = label
        )

    plt.legend()

    # if accline:
    #     plt.hlines(accline, 1, len(data)+1, linestyles='dashed', colors=(0, 0, 0, 0.8))

    # plt.show()
    plt.savefig(f'{title}.png')

    return fig


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print('pytorch device: ', device)

    all_rec = dict()

    for batch_size in (4,):
        for lr in (0.003, 0.001, 0.0003, 0.0001):
            opt = "sgd"
            models = {
                "eeg" : EEGBlock().to(device),
                #"shallow" : ShallowConvNet().to(device)
            }
            optimizers = {
                "adam" : optim.Adam,
                "sgd" : optim.SGD
            }
            optimizer = optimizers[opt]
            print(f"start training with batch size {batch_size}, learning rate {lr}:")
            rec = run_models(models, 1500, batch_size, lr, optimizer, opt=opt)
            acc_avg = 0.0
            for i in range(100):
                acc_avg += rec["test"][len(rec["test"])-1-i]
            acc_avg /= 100
            print(f"batch size: {batch_size}, lr: {lr}, test acc: ", acc_avg)
            plot(title=f'{batch_size}_{lr}_{opt}', **rec)

    for (key1, key2), value in all_rec:
        print(f"{key1}, {key2} final acc: ", value)

    # batch_size = 32
    # lr = 0.0003
        
    # models = {
    #             "eeg" : EEGBlock().to(device),
    #             #"shallow" : ShallowConvNet().to(device)
    #         }
    # print(f"start training with batch size {batch_size}, learning rate {lr}:")
    # rec = run_models(models, 1500, batch_size, lr, optimizer=optim.SGD)
    # acc_avg = 0.0
    # for i in range(100):
    #     acc_avg += rec["test"][len(rec["test"])-1-i]
    # acc_avg /= 100
    # print(f"batch size: {batch_size}, lr: {lr}, test acc: ", acc_avg)

    # plot(title=f'{batch_size}_{lr}', **rec)
    # rec = run_models(models, 1500, batch_size, lr, optimizer=optim.Adam)
    # acc_avg = 0.0
    # for i in range(100):
    #     acc_avg += rec["test"][len(rec["test"])-1-i]
    # acc_avg /= 100
    # print(f"batch size: {batch_size}, lr: {lr}, test acc: ", acc_avg)
    # plot(title=f'{batch_size}_{lr}', **rec)



if __name__ == '__main__':
    main()