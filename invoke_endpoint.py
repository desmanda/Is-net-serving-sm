from PIL import Image
import io
import os
import boto3

REGION_NAME = "eu-central-1"
# TODO run THIS IF ENDPOINT IS DEPLOYED!
# Define SageMaker client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

def call_endpoint(image_path, output_path):
    image = Image.open(image_path)
    # Convert image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName="is-net-endpoint",
        ContentType="application/json",
        Accept="application/x-image",
        Body=image_bytes
    )
    # Read the image from the response
    image_bytes = response["Body"].read()
    image = Image.open(io.BytesIO(image_bytes))

    # Save or display the image
    image.save(os.path.join(output_path, os.path.basename(image_path).split('.')[0] + "output.png"))
    return image


if __name__ == "__main__":
    call_endpoint(image_path = 'dataset/0003.jpeg', output_path='')




