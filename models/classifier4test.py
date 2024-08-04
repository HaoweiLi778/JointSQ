import torch


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dim [3, 128, 128]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1),   # [64, 128, 128]
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2, 0),       # [64, 64, 64]

            torch.nn.Conv2d(64, 128, 3, 1, 1),    # [128, 64, 64]
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2, 0),   # [128, 32, 32]

            torch.nn.Conv2d(128, 256, 3, 1, 1),    # [256, 32, 32]
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2, 0),   # [512, 16, 16]

            torch.nn.Conv2d(256, 512, 3, 1, 1),    # [512, 16, 16]
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2, 0),   # [512, 8, 8]

            torch.nn.Conv2d(512, 512, 3, 1, 1),    # [512, 8, 8]
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2, 0),   # [512, 4, 4]
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512*4*4, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)






def classifier() -> Classifier:
    """classifier model for test"""
    return Classifier()
