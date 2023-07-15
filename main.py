import torch
from torch import nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import face_recognition
from torch import nn
from torchvision import models

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

from torchvision.models import resnext50_32x4d
from torchvision.models.resnet import ResNet, Bottleneck

IMAGEDIR = "images/"
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained='imagenet')
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))

def im_convert(tensor):
    """แปลง Tensor เป็นรูปภาพ"""
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png',image*255)
    return image

def predict(model, img, path='./'):
    fmap, logits = model(img.to('cpu'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('ความเชื่อมั่นในการทำนาย:', logits[:, int(prediction.item())].item() * 100)
    return [int(prediction.item()), confidence]

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, sequence_length=60, transform=None):
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step_size = max(frame_count // self.sequence_length, 1)

        for i in range(0, frame_count, step_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if success:
                faces = face_recognition.face_locations(frame)
                try:
                    top, right, bottom, left = faces[0]
                    frame = frame[top:bottom, left:right, :]
                except:
                    pass
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

        cap.release()

        frames = torch.stack(frames)
        return frames

def predict(model, video_dataset):
    model.eval()
    predictions = []
    confidences = []

    with torch.no_grad():
        for video in video_dataset:
            video = video.unsqueeze(0).cpu()
            fmap, logits = model(video)
            probabilities = nn.functional.softmax(logits, dim=1)
            confidence = torch.max(probabilities).item() * 100
            prediction = torch.argmax(probabilities).item()

            predictions.append(prediction)
            confidences.append(confidence)

    return predictions, confidences

@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    file_path = os.path.join(IMAGEDIR, video.filename)
    with open(file_path, "wb") as f:
        f.write(video.file.read())

    device = torch.device("cpu")
    model_path = "model2.h5"
    video_paths = [file_path]
    image_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    video_dataset = ValidationDataset(video_paths, sequence_length=20, transform=data_transforms)
    model = Model(2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.eval()
    predictions, confidences = predict(model, video_dataset)

    for i in range(len(video_paths)):
        video_path = video_paths[i]
        prediction = predictions[i]
        confidence = confidences[i]

    print(f"วิดีโอ: {video_path}")
    print(f"การทำนาย: {'FAKE' if prediction == 1 else 'REAL'}")
    print(f"ความเชื่อมั่น: {confidence:.2f}%")
    print()

    return {"filename": video.filename, "prediction": prediction, "confidence": confidence}
