import torch.nn as nn
import torch.nn.functional as func

class EmbeddingToGPSDecoder(nn.Module):
    def __init__(self):
        super(EmbeddingToGPSDecoder, self).__init__()

        # First layer: 512 (embedding size) -> 512
        self.layer1 = nn.Linear(512, 512)
        self.norm1 = nn.LayerNorm(512)

        # Second layer: 512 -> 256
        self.layer2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)

        # Output layer: 256 -> 2 (latitude and longitude)
        self.output_layer = nn.Linear(256, 2)

    def forward(self, x):
        # First layer with normalization and dropout
        x1 = func.leaky_relu(self.norm1(self.layer1(x)))

        # Second layer with normalization and dropout
        x2 = func.leaky_relu(self.norm2(self.layer2(x1)))

        # Output layer for predicting latitude and longitude
        gps_coordinates = self.output_layer(x2)

        return gps_coordinates