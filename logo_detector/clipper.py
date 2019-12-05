from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.pytorch import deploy_pytorch_model


clipper_conn = ClipperConnection(DockerContainerManager())

try:
    clipper_conn.connect()
except:
    clipper_conn.start_clipper()

clipper_conn.register_application(name="logo-detector", input_type="strings", default_output="no logo", slo_micros=10000000)
clipper_conn.get_all_apps()

from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
import requests
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # pylint: disable=maybe-no-member

CLASSES = [
        'ADAC', 'FCB', 'HP', 'adidas', 'aldi', 'apple', 'becks', 'bmw',
        'carlsberg', 'chimay', 'cocacola', 'corona', 'dhl', 'erdinger', 'esso',
        'fedex', 'ferrari', 'ford', 'fosters', 'google', 'guiness', 'heineken',
        'manu', 'milka', 'no-logo', 'nvidia', 'paulaner', 'pepsi',
        'rittersport', 'shell', 'singha', 'starbucks', 'stellaartois',
        'texaco', 'tsingtao', 'ups'
]

MODELS_PATH = (Path('/Users/arash/Projects/logo-detector/logo_detector') / "models").resolve()


def get_model():
    model = torchvision.models.resnet18(pretrained='True')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 36)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load((MODELS_PATH / 'resnet18.pt').resolve()))
    return model

def predict(model, image_url):
    def get_image(loader, image_url):
        url = str(image_url).split(' ')[0][3:-1]
        response = requests.get(url)
        content = response.content
        image = Image.open(BytesIO(content))
        image = image.convert('RGB')
        image = loader(image).float()
        image = image.unsqueeze(0)
        return image
    train_mean = np.array([0.44943, 0.4331, 0.40244])
    train_std = np.array([0.29053, 0.28417, 0.30194])
    model.eval()
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    image = get_image(data_transforms, image_url)
    pred = np.argmax(model(image).detach().numpy())
    return CLASSES[pred]

model = get_model()

clipper_conn.stop_models(model_names='logo-detector') 
deploy_pytorch_model(
    clipper_conn,
    name="logo-detector",
    version=1,
    input_type="strings",
    func=predict,
    pytorch_model=model,
    pkgs_to_install=['Pillow', 'torchvision', 'torch', 'numpy', 'requests']
    )
clipper_conn.get_clipper_logs()

clipper_conn.link_model_to_app(app_name="logo-detector", model_name="logo-detector")

clipper_conn.get_linked_models(app_name="logo-detector")

clipper_conn.cm.get_num_replicas(name="logo-detector", version="1")

clipper_conn.get_clipper_logs()

