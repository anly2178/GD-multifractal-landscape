import torch.nn as nn

class VGG_mnist(nn.Module): # This is VGG13
    def __init__(self):
        super(VGG_mnist, self).__init__()
        inplace = False
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2), # 14 x 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2), # 7 x 7
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2), # 3 x 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2), # 1 x 1
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=inplace), 
            nn.Linear(4096, 4096), nn.ReLU(inplace=inplace), 
            nn.Dropout(p=0.5, inplace=inplace),
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class VGG_cifar(nn.Module): # Difference is extra set of conv layers; this is VGG16
    def __init__(self):
        super(VGG_cifar, self).__init__()
        inplace = False
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2), # 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2), # 8 x 8 
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2), # 4 x 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2), # 2 x 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2), # 1 x 1
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=inplace), 
            nn.Linear(4096, 4096), nn.ReLU(inplace=inplace), 
            nn.Dropout(p=0.5, inplace=inplace),
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class VGG_mnist_nd(nn.Module): # This is VGG13
    def __init__(self):
        super(VGG_mnist_nd, self).__init__()
        inplace = False
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2), # 14 x 14
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2), # 7 x 7
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2), # 3 x 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2), # 1 x 1
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=inplace), 
            nn.Linear(4096, 4096), nn.ReLU(inplace=inplace), 
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class VGG_cifar_nd(nn.Module): # Difference is extra set of conv layers; this is VGG16
    def __init__(self):
        super(VGG_cifar_nd, self).__init__()
        inplace = False
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2), # 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2), # 8 x 8 
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2), # 4 x 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2), # 2 x 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=inplace), nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2), # 1 x 1
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=inplace), 
            nn.Linear(4096, 4096), nn.ReLU(inplace=inplace), 
            nn.Linear(4096, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x