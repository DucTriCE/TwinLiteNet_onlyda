import torch
import numpy as np
import shutil
import os
import cv2
from model import TwinLite as net
from const import *
from loss import TotalLoss
from argparse import ArgumentParser


if __name__ == '__main__':
    import time
    model = net.TwinLiteNet()
    model = model.cpu()
    model.load_state_dict(torch.load(''))
    model.eval()

    images = torch.rand(1, 3, 384, 640)

    start_time = time.time()
    for i in range(100):
        output = model(images)
    end_time = time.time()

    total_time = end_time - start_time
    fps = 100 / total_time  
    print("FPS:", fps)