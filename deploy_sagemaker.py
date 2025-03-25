import os
import boto3
import torch
import click
from serving_utils import copy_model_and_code_to_s3
from sagemaker.pytorch import PyTorchModel
# from sagemaker import get_execution_role
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_model(model_path, role_arn=None,
                 instance_type='ml.m4.xlarge',
                 endpoint_name='is-net-endpoint',
                 torch_version='2.6.0',
                 py_version='py312', s3_model=None): # py39 not supported
    if role_arn is None:
        role_arn = os.getenv("AWS_ROLE_ARN")


    #role = get_execution_role()
    logger.info(f'IAM role used {role_arn}')
    if not s3_model:
        s3_model_path = copy_model_and_code_to_s3(model_path=model_path)
    else:
        s3_model_path = s3_model

    if torch_version != torch.__version__.split("+")[0]:
        raise ValueError("Torch Version differs.")

    pytorch_model = PyTorchModel(
        model_data=s3_model_path,
        role=role_arn,
        framework_version=torch_version,
        py_version=py_version,
        entry_point='serve.py',
        source_dir='.'
    )

    delete_endpoint_config(endpoint_name)

    pytorch_model.deploy(
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        initial_instance_count=1
    )

def does_endpoint_config_exist(endpoint_config_name):
    sagemaker_client = boto3.client('sagemaker')
    try:
        sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        return True
    except sagemaker_client.exceptions.ClientError:
        return False


def delete_endpoint_config(endpoint_config_name):
    sagemaker_client = boto3.client('sagemaker')
    if does_endpoint_config_exist(endpoint_config_name):
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)



@click.command()
@click.option('--model-tar-path', default='model.tar.gz')
@click.option("--aws-role", default='arn:aws:iam::826838947158:role/AmazonSageMaker-ExecutionRole')
@click.option("--instance-type", default='ml.m4.xlarge', help='free instance, otherwise we use ml.m5.large '
                                                           'or GPU based instance depending on the latency')
@click.option("--endpoint-name", default='is-net-endpoint')
@click.option("--torch-version", default='2.6.0')
@click.option("--py_version", default='py312')
@click.option("--s3-model", default=None)
def test_deployment(model_tar_path, aws_role, instance_type, endpoint_name, torch_version, py_version, s3_model):
    deploy_model(model_path=model_tar_path,
                 role_arn=aws_role,
                 instance_type=instance_type,
                 endpoint_name=endpoint_name,
                 torch_version=torch_version,
                 py_version=py_version, s3_model=s3_model)

# tar -cvf model.tar.gz requirements_dev.txt serve.py serving_utils.py models/
# export AWS_ROLE_ARN='arn:aws:iam::826838947158:role/AmazonSageMaker-ExecutionRole'
# python deploy_sagemaker.py model.tar.gz
#  aws sts assume-role --role-arn arn:aws:iam::826838947158:role/AmazonSageMaker-ExecutionRole --role-session-name SageMakerSession
# aws sts get-caller-identity

if __name__ == "__main__":
    test_deployment()