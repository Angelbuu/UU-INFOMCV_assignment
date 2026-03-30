import torch
import torch.nn as nn
import torch.nn.functional as tf
from torchinfo import summary


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # out: 112x112x3
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.pool_1 = nn.MaxPool2d(2, 2)
        # out: 56x56x16
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.pool_2 = nn.MaxPool2d(2, 2)
        # out: 28x28x32
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.pool_3 = nn.MaxPool2d(2, 2)
        # out: 14x14x64
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(64)
        self.pool_4 = nn.MaxPool2d(2, 2)
        # out: 7x7x64
        self.conv_5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm_5 = nn.BatchNorm2d(32)
        # out: 7x7x32
        self.flatten = nn.Flatten()
        # out: 1568
        self.fc = nn.Linear(1568, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(512, 343)

    def forward(self, imgs):
        out = self.pool_1(tf.relu(self.batch_norm_1(self.conv_1(imgs))))
        out = self.pool_2(tf.relu(self.batch_norm_2(self.conv_2(out))))
        out = self.pool_3(tf.relu(self.batch_norm_3(self.conv_3(out))))
        out = self.pool_4(tf.relu(self.batch_norm_4(self.conv_4(out))))
        out = self.flatten(tf.relu(self.batch_norm_5(self.conv_5(out))))
        out = self.dropout(tf.relu(self.fc(out)))
        out = tf.sigmoid(self.out(out))
        return out


if __name__ == '__main__':
    summary(Model(), input_size=(1, 3, 112, 112), verbose=1)

    x = torch.randn(1, 3, 112, 112)
    print(Model()(x).shape)
