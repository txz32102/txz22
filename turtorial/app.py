import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import torch
import os
from os import listdir
from os.path import isfile, join
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import cv2
from pathlib import Path

app = Flask(__name__)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 Hidden Layer Network
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        
        # Dropout module with 0.2 probbability
        self.dropout = nn.Dropout(p=0.2)
        # Add softmax on output layer
        self.log_softmax = F.log_softmax
        
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        
        x = self.log_softmax(self.fc5(x), dim=1)
        
        return x

def transform(path):
    image = cv2.imread(path)
    resized = cv2.resize(image, (28,28))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    output = gray
    data = torch.tensor(output).reshape(1,784).float()
    return data

PATH = r'1.pt'
model = Classifier()
model.load_state_dict(torch.load(PATH))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    dirpath = r'C:\Users\Administrator\Downloads'
    paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)
    path = paths[-1].__str__()

    res = torch.exp(model(transform(path)))
    return render_template('index.html', predictionText=str(res)+" "+str(path))

@app.route('/image')
def show_img():
    path = r'static\test1.png'
    return render_template('index.html', image_=path)

# set flask_env=development
if __name__ == "__main__":
    app.run(debug=True)
