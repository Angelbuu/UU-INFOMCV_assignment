import torch
from torchsummary import summary
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv_1 = nn.Conv2d(input_channels, 6, 5)
        self.act_1 = nn.ReLU()
        self.pool_1 = nn.AvgPool2d(2, 2)

        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.act_2 = nn.ReLU()
        self.pool_2 = nn.AvgPool2d(2, 2)

        self.conv_3 = nn.Conv2d(16, 120, 5)
        self.act_3 = nn.ReLU()

        self.fc_1 = nn.Linear(120, 84)
        self.act_4 = nn.ReLU()

        self.fc_2 = nn.Linear(84, 10)

    def forward_convolutions(self, feature_maps):
        out = self.pool_1(self.act_1(self.conv_1(feature_maps)))
        out = self.pool_2(self.act_2(self.conv_2(out)))
        return self.act_3(self.conv_3(out))

    def forward_dense_layers(self, flat_feature_vec):
        out = self.act_4(self.fc_1(flat_feature_vec))
        return self.fc_2(out)

    def forward(self, imgs):
        out = self.forward_convolutions(imgs)
        out = torch.flatten(out, 1)
        out = self.forward_dense_layers(out)
        return out


class LeNetVariant1(LeNet):
    def __init__(self):
        super().__init__(input_channels=6)
        self.conv_0 = nn.Conv2d(3, 6, 3, padding=1)
        self.act_0 = nn.ReLU()

    def forward_convolutions(self, feature_maps):
        out = self.act_0(self.conv_0(feature_maps))
        return super().forward_convolutions(out)


class LeNetVariant2(LeNetVariant1):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward_dense_layers(self, flat_feature_vec):
        out = self.act_4(self.fc_1(flat_feature_vec))
        out = self.dropout(out)
        return self.fc_2(out)


class CIFAR100Model(LeNetVariant2):  # TODO: choose the best model to inherit
    def __init__(self):
        super().__init__()
        self.fc_2 = nn.Linear(84, 20)


def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def init_model(model_class=LeNet):
    model = model_class()
    model.apply(init_weights)
    return model


if __name__ == '__main__':
    baseline = init_model(LeNet)

    summary(LeNet(), (3, 32, 32))
    summary(LeNetVariant1(), (3, 32, 32))
    summary(LeNetVariant2(), (3, 32, 32))

    x = torch.randn(1, 3, 32, 32)
    print(LeNet()(x).shape)
    print(LeNetVariant1()(x).shape)
    print(LeNetVariant2()(x).shape)
