import torch
import numpy as np
import shutil
import os
import cv2
from model import TwinLite2 as net
from const import *
from loss import TotalLoss
import argparse
import onnxruntime as ort


if __name__ == "__main__":
    ort.set_default_logger_severity(4)
    onnx_path = f""
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    # print(f"Loading done!")

    outputs = ort_session.get_outputs()
    inputs = ort_session.get_inputs()

    img = torch.rand(1, 3, 384, 640).numpy()
    import time

    start_time = time.time()
    for i in range(100):
        out_da, out_ll = ort_session.run(['da', 'll'],{"images": img})
    end_time = time.time()

    total_time = end_time - start_time
    fps = 100 / total_time  
    print("FPS:", fps)

