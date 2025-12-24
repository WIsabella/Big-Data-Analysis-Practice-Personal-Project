
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.classifier(x)
