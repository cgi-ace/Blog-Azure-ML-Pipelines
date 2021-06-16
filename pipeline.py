from azureml.pipeline.steps import PythonScriptStep, RScriptStep
from azureml.pipeline.core import Pipeline, PipelineData

from azureml.core import Workspace

from azureml.core.runconfig import RunConfiguration
from azureml.core.environment import RSection, RCranPackage, CondaDependencies

from azureml.core.environment import RSection

from azureml.data.data_reference import DataReference
from azureml.core.authentication import ServicePrincipalAuthentication

from azureml.core import Dataset
from azureml.pipeline.core import PipelineData
from azureml.core.datastore import Datastore
from azureml.pipeline.core import PipelineData
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment

import os
import yaml

def main():
   """
    In this demo, authentication is done using a config file with details about subsciption ID, workspace & resource group name.
    This can be replaced with the Use of Service Principal for authenticating with the Workspace.

    Service principal authentication involves creating an App Registration in Azure Active Directory. First, you generate a client
    secret, and then you grant your service principal role access to your machine learning workspace. Then, you use the
    ServicePrincipalAuthentication object to manage your authentication flow
    """
    ws = Workspace.from_config()

    # Choose a name for your CPU cluster
    cpu_cluster_name = "cpu-cluster"
                     

    # Verify that cluster does not exist already
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                                max_nodes=4)
        cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)

    """
    Run configuration Azure 

    The RunConfiguration object encapsulates the information necessary to submit a training run in an experiment. For example,
    when setting up a PythonScriptStep or RScriptStep for running python and R scripts respectively, you can access
    the step's RunConfiguration object and configure Conda dependencies or access the environment properties for the run.

    """
    # Run configuration for R
    rc = RunConfiguration()
    rc.framework = "R"

    # Run configuration for python
    py_rc = RunConfiguration()
    py_rc.framework = "Python"
    py_rc.environment.docker.enabled = True

    # Combine GitHub and Cran packages for R env
    rc.environment.r = RSection()
    rc.environment.r.user_managed = True

    # Use custom packages using the Dockerfile.txt
    rc.environment.docker.enabled = True
    rc.environment.docker.base_image = None
    rc.environment.docker.base_dockerfile = ".\Dockerfile.txt"

    # Upload iris data to the datastore
    target_path = "iris_data"
    upload_files_to_datastore(ds,
                             list("./iris.csv"),
                             target_path = target_path,
                             overwrite = TRUE)

    # Get the data from datastore
    data_source = DataReference(
        datastore=ws.get_default_datastore(),
        data_reference_name="iris_data",
        path_on_datastore="iris_data/iris.csv",
    )
    

    # Run the Iris Training
    training_step = CommandStep(
        script_name="train.R",
        arguments=[training_data],
        inputs=[training_data],
        compute_target=cpu_cluster_name,
        source_directory=".",
        runconfig=rc,
        allow_reuse=True,
    )

    print("Step Train created")

    # Any number of steps can be defined similarly to the training_step and can be added to the pipeline
    # by appending to the list below

    steps = [training_step]

    train_pipeline = Pipeline(workspace=ws, steps=steps)
    train_pipeline.validate()
    pipeline_run = Experiment(ws, 'Iris_Classifier').submit(train_pipeline)
    pipeline_run.wait_for_completion(show_output=True)
    """
    Publishing Pipeline
    
    When you publish a pipeline, it creates a REST endpoint automatically. With the pipeline endpoint, you can trigger a run of the pipeline from any external systems, including non-Python clients. This endpoint enables "managed repeatability" in batch scoring and retraining scenarios.

    
    """

    published_pipeline = train_pipeline.publish(
        name="Iris_classification",
        description="Iris Classification using R",
    )
    print(f"Published pipeline: {published_pipeline.name}")
    print(f"for build {published_pipeline.version}")

if __name__ == "__main__":
    main()

