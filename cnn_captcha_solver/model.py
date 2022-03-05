from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Linear,
    LogSoftmax
)

class Model(Module):
    '''Convolutional neural network architecture, loosely based on LeNet'''
    def __init__(self):
        super(Model, self).__init__()

        self.cnn_layers = Sequential(
            # first conv layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # second conv layer, no pooling
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
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
