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

MODELS_PATH = (Path(__file__).parent / "models").resolve()


def get_resnet():
    model = torchvision.models.resnet18(pretrained='True')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 36)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load((MODELS_PATH / 'resnet18.pt').resolve()))
    return model


def url_loader(loader, image_url):
    response = requests.get(image_url)
    content = response.content
    image = Image.open(BytesIO(content))
    image = image.convert('RGB')
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image


def label_provider(model, image_url, config):
    train_mean = np.array([config[0]])
    train_std = np.array([config[1]])
    model.eval()
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)])
    pred = np.argmax(model(url_loader(data_transforms, image_url)).detach().numpy())
    return CLASSES[pred]


def detect_resnet(image_url, model=get_resnet(), 
                  config=([0.44943, 0.4331, 0.40244],
                          [0.29053, 0.28417, 0.30194])):
    return label_provider(model, image_url, config)


def predict(model, image_url):
    def get_image(loader, image_url):
        response = requests.get(image_url)
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
    pred = np.argmax(model(get_image(data_transforms, image_url)).detach().numpy())
    return CLASSES[pred]
