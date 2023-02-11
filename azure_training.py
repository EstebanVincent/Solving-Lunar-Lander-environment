# Handle to the workspace
from azure.ai.ml import Input
from azure.ai.ml import command
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import MLClient

# Authentication package
from azure.identity import InteractiveBrowserCredential
credential = InteractiveBrowserCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="c558106c-e962-425f-bab8-291db4614152",
    resource_group_name="erasmus",
    workspace_name="lunar-lander",
)


gpu_compute_taget = "gpu-cluster"

try:
    # let's see if the compute target already exists
    gpu_cluster = ml_client.compute.get(gpu_compute_taget)
    print(
        f"You already have a cluster named {gpu_compute_taget}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new gpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    gpu_cluster = AmlCompute(
        # Name assigned to the compute cluster
        name="gpu-cluster",
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_NC6",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Now, we pass the object to MLClient's create_or_update method
    gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()

print(
    f"AMLCompute with name {gpu_cluster.name} is created, the compute size is {gpu_cluster.size}"
)

curated_env_name = "AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest"


job = command(
    inputs=dict(
        model_version=2
    ),
    compute=gpu_compute_taget,
    environment=curated_env_name,
    code="./",  # location of source code
    command="pip install -r requirements.txt && python main.py --analyse ${{inputs.model_version}}",
    experiment_name="pytorch-fourth-try",
    display_name="pytorch-fourth-try",
)

ml_client.jobs.create_or_update(job)
