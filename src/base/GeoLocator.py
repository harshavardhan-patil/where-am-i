from transformers import SwinModel
from PIL import Image
from src.config import MODEL_NAME
import torch.nn as nn
import torch.nn.functional as func

from transformers import ViTImageProcessor, ViTModel
from PIL import Image

class GeoLocator(nn.Module):
    def __init__(self):
        super(GeoLocator, self).__init__()

        self.backbone = ViTModel.from_pretrained(MODEL_NAME)

        self.layer1 = nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size)
        self.norm1 = nn.LayerNorm(self.backbone.config.hidden_size)
        self.dropout1 = nn.Dropout(p=0.05)

        self.layer2 = nn.Linear(self.backbone.config.hidden_size, 512)  # 512: embedding size of location encoder
        self.norm2 = nn.LayerNorm(512)
        self.dropout2 = nn.Dropout(p=0.05)

        self.layer3 = nn.Linear(512, 512)

    def forward(self, x):
        # Extract last hidden state from the ViT backbone
        x1 = self.backbone(x).last_hidden_state
        x1 = x1[:, 0, :]  # Use CLS token only

        # First layer with normalization and dropout
        a1 = func.leaky_relu(self.norm1(self.layer1(x1)))
        a1 = self.dropout1(a1)

        # Second layer with normalization and dropout
        a2 = func.leaky_relu(self.norm2(self.layer2(a1)))
        a2 = self.dropout2(a2)

        # Output layer
        output = self.layer3(a2)

        return output
   