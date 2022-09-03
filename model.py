import torch
from torch import nn


class BasicArbitraryCNNmodel(nn.Module):
    def __init__(self, in_channels=3, num_outputs=10):
        super(BasicArbitraryCNNmodel, self).__init__()
        hidden_channels = 64
        self.input_block = self.build_intermediate_block(in_channels, hidden_channels, 7, 2, 3)
        self.max_pool1 = nn.MaxPool2d(2, 2, 0)
        self.hidden_block1 = self.build_intermediate_block(hidden_channels, hidden_channels * 2, 3, 1, 0)
        self.max_pool2 = nn.MaxPool2d(2, 2, 0)
        self.hidden_block2 = self.build_intermediate_block(hidden_channels * 2, hidden_channels * 4, 3, 1, 0)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.output_block = self.build_output_block(1024, num_outputs)

    def build_intermediate_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def build_output_block(self, in_features, num_outputs):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features=in_features // 2, out_features=num_outputs)
        )

    def forward(self, x):
        out = self.input_block(x)
        out = self.max_pool1(out)
        out = self.hidden_block1(out)
        out = self.max_pool2(out)
        out = self.hidden_block2(out)
        out = self.adaptive_avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        return self.output_block(out)
