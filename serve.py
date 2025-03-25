import torch
import os
from serving_utils import load_model, preprocess_image, post_processing
from PIL import Image
import io
# chmod +x serve.py
MODEL_FILE_NAME = 'isnet-general-use_model.pth'

# these are defined functions from SageMAKER that needs to be implemented for the sue case of DIS

def model_fn(model_dir):
    # to do include model_file_name as a param
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)
    model = load_model(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        input_tensor = preprocess_image(image)
        return input_tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_data = input_data.to(device)
        predictions = model(input_data)
        output = post_processing(predictions)
    return output


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return prediction.cpu().numpy().tolist()
    elif response_content_type == "application/x-image":
            image = Image.fromarray(prediction)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")