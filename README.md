# Is-net-serving-sm
This repository is for deploying a [ISNet]((https://github.com/xuebinqin/DIS)) model on SageMaker.
A CI/CD pipeline with github workflows is created on every push to main branch.
For simplicity, it is only deployed and while workflow is run only on main, ideally this should also be tested in every PR,
and only deployed on Merge.

The workflow has the following steps:
1) Fetch the latest saved ISNet weights from s3 (this should never be commited to git)
2) tar the new code (from this repo) + the model.pth to model.tar.gz --> this is what Sagemaker is expecting, serve.py
the request and running the inference, part from any script is needed for running the inference, e.g, `isnet.py`.
3) then we can deploy the model, by running ´deploy_sagemaker.py´, that trigger a PyTorchModel and an endpoint on SageMaker.
In order to run the pipeline, we need to add the secret ´AWS_ACCOUNT_ID´, ´AWS_SECRET_ACCESS_KEY´ and ´AWS_ACCESS_KEY_ID´, 
as it is accessing a private AWS account. 

# Blockers
I spent time struggling with sagemaker role, and deployment for testing the model locally before the github 
workflow, somehow the deployment failed when the model.tar.gz is moved to the sagemaker special buckets, I get SSL error. 
This is not a usual error that I get, it can be due to my new environment and AWS account. 
# what I can improve if I have time
- Adding unit tests with mock to serve.py
- finish testing the github pipeline
- Add a workflow for testing the endpoint and deleting it for PRs



