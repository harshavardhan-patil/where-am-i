from pathlib import Path

from loguru import logger
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision import transforms
import torch
from src.base.GeoLocator import GeoLocator
from src.base.EmbeddingToGPSDecoder import EmbeddingToGPSDecoder
from huggingface_hub import hf_hub_download
from src.config import REPO_ID, MODEL, REVERSE

model = GeoLocator()
model.load_state_dict(torch.load(f=hf_hub_download(repo_id=REPO_ID, filename=MODEL), weights_only=True)['model_state_dict'])
gps_decoder = EmbeddingToGPSDecoder()
gps_decoder.load_state_dict(torch.load(f=hf_hub_download(repo_id=REPO_ID, filename=REVERSE), weights_only=True))

def image_to_tensor(uploaded_file):
    # Open the image
    image = Image.open(uploaded_file)

    transform = transforms.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),  
        v2.ToDtype(torch.float32),         
        v2.Normalize(
          mean=[0.0, 0.0, 0.0],  
          std=[255.0, 255.0, 255.0]
        ),
        v2.Normalize(
          mean=[0.5, 0.5, 0.5],
          std=[0.5, 0.5, 0.5]
        )
    ])

    tensor = transform(image)
    return tensor   


def predict_nn(uploaded_file):
    model.eval()
    gps_decoder.eval()
    image = image_to_tensor(uploaded_file).unsqueeze(0) #adding a batch dimension for vit

    with torch.no_grad():
        embedding = model(image)
        output = gps_decoder(embedding)
    
    return output.tolist()[0]

if __name__ == "__main__":
    logger.info("predictions require input")
