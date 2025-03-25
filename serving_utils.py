import os
import boto3
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from models.isnet import ISNetDIS

import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_SIZE = [1024,1024]

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (C, H, W) and scales [0, 1]
    transforms.Resize(INPUT_SIZE, transforms.InterpolationMode.BILINEAR),  # Resize
    # pixels are already in [0, 1], no need to divide by 255
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])  # Normalize
])

def copy_model_and_code_to_s3(model_path, bucket_name='trained-models-pytorch'):
    s3 = boto3.client('s3')
    model_uri = s3.upload_file(
        Filename=model_path,
        Bucket=bucket_name,
        Key=os.path.join('is-net-segmentation', os.path.basename(model_path))
    )
    logger.info(f's3 copied uri {model_uri}')
    return model_uri


def load_model(model_file_path):
    model = ISNetDIS()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
    logger.info('Model is loaded!')
    model.eval()
    return model

def post_processing(raw_output):
    shape = raw_output.shape[0:2]
    result = torch.squeeze(F.interpolate(raw_output[0][0], shape, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)

    result = (result - mi) / (ma - mi)
    result_np = (result * 255).cpu().data.numpy().astype(np.uint8)

    if result_np.ndim == 3 and result_np.shape[0] == 1:  # (1, H, W)
        result_np = result_np.squeeze(0)  # Convert to (H, W)
    logger.info('input image is post-processed!')
    return result_np

def preprocess_image(image_pil):
    image = transform(image_pil)
    # Add batch dimension
    input_image = image.unsqueeze(0)  # Shape: (1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
    logger.info('input image is pre-processed!')
    return input_image


def predict_mask(input_data, model):
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    return model(input_data)
