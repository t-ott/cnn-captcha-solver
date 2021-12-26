'''
PyTorch CNN model class
'''
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, LogSoftmax

class Model1(Module):
    def __init__(self):
        super(Model1, self).__init__()

        self.cnn_layers = Sequential(
            # first conv layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # second conv layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = Sequential(
            Linear(24, 32)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x


class Model2(Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.cnn_layers = Sequential(
            # first conv layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # second conv layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = Sequential(
            Linear(24, 48),
            ReLU(inplace=True),
            Linear(48, 32),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x


class Model3(Module):
    def __init__(self):
        super(Model3, self).__init__()

        self.cnn_layers = Sequential(
            # conv layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(168, 64),
            ReLU(inplace=True),
            Linear(64, 32),
            LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x
