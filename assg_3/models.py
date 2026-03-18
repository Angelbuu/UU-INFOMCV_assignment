import torch
import torch.nn as nn
from torchinfo import summary


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

        self.feedback_layer = nn.Sequential(
            nn.AvgPool2d(7, 7),
            nn.Flatten(),
            nn.Linear(24, 10)
        )

        self.feedback_layer_2 = nn.Sequential(
            nn.AvgPool2d(5, 5),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

    def forward_convolutions(self, feature_maps, use_feedback):
        out_feedback_1, out_feedback_2 = None, None
        out = self.pool_1(self.act_1(self.conv_1(feature_maps)))
        if use_feedback:
            out_feedback_1 = self.feedback_layer(out)
        out = self.pool_2(self.act_2(self.conv_2(out)))
        if use_feedback:
            out_feedback_2 = self.feedback_layer_2(out)
        out = self.act_3(self.conv_3(out))
        return out, out_feedback_1, out_feedback_2

    def forward_dense_layers(self, flat_feature_vec):
        out = self.act_4(self.fc_1(flat_feature_vec))
        return self.fc_2(out)

    def forward(self, imgs, use_feedback=False):
        out, out_feedback_1, out_feedback_2 = self.forward_convolutions(imgs, use_feedback)
        out = torch.flatten(out, 1)
        out = self.forward_dense_layers(out)
        if use_feedback:
            return out, out_feedback_1, out_feedback_2
        return out

    def get_embeddings(self, imgs):
        out, _, _ = self.forward_convolutions(imgs, use_feedback=False)
        out = torch.flatten(out, 1)
        embeddings = self.act_4(self.fc_1(out))
        return embeddings


class LeNetVariant1(LeNet):
    def __init__(self):
        super().__init__(input_channels=6)
        self.conv_0 = nn.Conv2d(3, 6, 3, padding=1)
        self.act_0 = nn.ReLU()

    def forward_convolutions(self, feature_maps, use_feedback):
        out = self.act_0(self.conv_0(feature_maps))
        return super().forward_convolutions(out, use_feedback)


class LeNetVariant2(LeNetVariant1):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward_dense_layers(self, flat_feature_vec):
        out = self.act_4(self.fc_1(flat_feature_vec))
        out = self.dropout(out)
        return self.fc_2(out)


class CIFAR100LeNet(LeNet):
    def __init__(self):
        super().__init__()
        self.fc_2 = nn.Linear(84, 20)


class CIFAR100Variant1(LeNetVariant1):
    def __init__(self):
        super().__init__()
        self.fc_2 = nn.Linear(84, 20)


class CIFAR100Variant2(LeNetVariant2):
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
    summary(LeNet(), input_size=(1, 3, 32, 32), verbose=1)
    summary(LeNetVariant1(), input_size=(1, 3, 32, 32), verbose=1)
    summary(LeNetVariant2(), input_size=(1, 3, 32, 32), verbose=1)

    x = torch.randn(1, 3, 32, 32)
    print(LeNet()(x).shape)
    print(LeNetVariant1()(x).shape)
    print(LeNetVariant2()(x).shape)
