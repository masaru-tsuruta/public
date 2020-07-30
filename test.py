#https://tzmi.hatenablog.com/entry/2020/02/23/232707
from torch import nn

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
            nn.Softmax(dim=1),
        )

        # weight init                                                                      
        for m in self.layers.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.layers(x)
    
from torch import nn

class cnn2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.r1 = nn.ReLU(inplace=True)
        self.m1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(6, 16, kernel_size=5)
        self.r2 = nn.ReLU(inplace=True)
        self.m2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.r3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120,84)
        self.r4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

        # weight init                                                                      
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.m1(self.r1(self.c1(x)))
        x = self.m2(self.r2(self.c2(x)))
        x = self.flatten(x)
        x = self.r3(self.fc1(x))
        x = self.r4(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

if __name__ == '__main__':

    # GPU or CPUの自動判別                                                                 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # modelの定義                                                                          
    model = cnn().to(device)
    opt = torch.optim.Adam(model.parameters())

    # datasetの読み出し                                                                    
    bs = 128 # batch size                                                                  
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False)

    # training                                                                             
    print('train')
    model = model.train()
    for iepoch in range(3):
        for iiter, (x, y) in enumerate(trainloader, 0):

            # toGPU (CPUの場合はtoCPU)                                                     
            x = x.to(device)
            y = torch.eye(10)[y].to(device)

            # 推定                                                                         
            y_ = model.forward(x) # y_.shape = (bs, 84)                                    

            # loss: cross-entropy                                                          
            eps = 1e-7
            loss = -torch.mean(y*torch.log(y_+eps))

            opt.zero_grad() # 勾配初期化
            loss.backward() # backward (勾配計算)
            opt.step() # パラメータの微小移動

            # 100回に1回進捗を表示（なくてもよい）
            if iiter%100==0:
                print('%03d epoch, %05d, loss=%.5f' %
                      (iepoch, iiter, loss.item()))

    # test                                                                                 
    print('test')
    total, tp = 0, 0
    model = model.eval()
    for (x, label) in testloader:

        # to GPU                                                                           
        x = x.to(device)

        # 推定                                                                             
        y_ = model.forward(x)
        label_ = y_.argmax(1).to('cpu')

        # 結果集計                                                                         
        total += label.shape[0]
        tp += (label_==label).sum().item()

    acc = tp/total
    print('test accuracy = %.3f' % acc)
