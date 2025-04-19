# ESXi-OpenShift-Ray-vLLM-LiteLLM

## Introduction
In this repository we are looking at Multi GPU systems (VMWare ESXi) using OpenShift 4, Ray, vLLM and LiteLLM for inference with large genAI models. Each ESXi server is equipped with 2 Nvidia GPUs (totaling 8 systems x 2 GPUs are 16 GPUs).

**Note:** This repository is an excerpt from a private repository, designed for sharing purposes. It contains curated content to illustrate concepts and configurations without revealing sensitive information. As an excerpt, it may require additional code or configuration for complete implementation.

## Open MPI
A pdf document added (in Dutch) which describes an alternative method using [Open MPI instead of Ray](https://github.com/rabem00/Ray-vLLM-OpenShift-ESXi/blob/main/vLLM-met-Ray-of-MPI-voor-gedistribueerde-inference.pdf)

## Assigning GPUs to a Single VM Worker Node
It’s not possible to assign all 16 GPUs from 8 ESXi servers (2 GPUs each) to a single VM using GPU passthrough or vGPU. VMware vSphere restricts GPU assignment to the *physical host running the VM*, and cross-host GPU aggregation isn’t supported.

A potential solution is to consolidate hardware by physically relocating all GPUs to a single ESXi server with sufficient PCIe slots, then assigning them to one VM via passthrough. However, in our case, this is a theoretical exercise, as it’s impractical to install 16 GPUs in a single system.

## Deployment Options for Model Sizes

### Smaller Models
Smaller models, such as LLaMA 3 8B, can fit within the GPU memory of a single node (e.g., 2 GPUs). In this case, you can deploy multiple independent vLLM pods, each utilizing a subset of GPUs without requiring model parallelism across nodes. This approach simplifies the configuration and optimizes resource utilization.

### Larger Models
For larger models, such as LLaMA 4 or DeepSeek, vLLM alone cannot natively distribute the model across multiple nodes—a critical requirement in our scenario with 16 GPUs spread across 8 ESXi servers. To enable distributed model parallelism, a framework like Ray is necessary. Ray supports the creation of a Ray Cluster, which distributes the model across all GPUs without requiring hardware modifications, leveraging the existing infrastructure efficiently.

## Minimal Setup Instructions

### OpenShift 4
This guide assumes the use of OpenShift 4 (version 4.16 or higher) for the deployment. Each ESXi server hosts an OpenShift worker node, with GPUs assigned via passthrough. Specifically:
* Create one OpenShift worker node per ESXi server (8 nodes total for 8 ESXi servers).
* Assign 2 GPUs to each worker node, matching the hardware configuration of 16 GPUs across 8 servers.
* Label nodes with GPUs for scheduling:
```bash
oc label node <node-name> nvidia.com/gpu.present=true
```

### Operators
Two operators are required (see links):
* NVIDIA GPU Operator: Install this operator in OpenShift to expose GPUs to pods. It integrates with the cluster to manage NVIDIA drivers and CUDA libraries, ensuring each node’s (2) GPUs are available.
* KubeRay Operator: Install this operator to enable RayCluster resources, allowing Ray to manage distributed workloads across the (16) GPUs.

### Build image
Build a container image with Ray and vLLM using the provided Dockerfile based on nvidia/cuda:12.8.0-base-ubi9 (preferred for Red Hat compliance in OpenShift):
```bash
docker build -t ray-vllm-ubi9:latest .
```
**Note:** an Ubuntu version is also provided.

### Setup Storage
Using the yaml file vllm-model-storage.yaml a physical volume and a phycical volume claim is setup. This is where the model is stored.

#### Temporary Pod for Copying
It is possible to copy the model to the storage in various ways, for instance: hostPath or NFS server access. In this case a temporary pod is used for copying the model. Usage:
```bash
oc apply -f model-loader.yaml
oc cp llama3-70b.tar.gz vllm-llama3/model-loader:/model/llama3-70b.tar.gz
oc exec -it model-loader -- tar -xzf /model/llama3-70b.tar.gz -C /model
oc delete pod model-loader
```

### Setup RayCluster and start vLLM (development)
The raycluster.yaml is used to setup a RayCluster with a head node (headGroupSpec) and ray workers (workerGroupSpecs).

Now that the RayCluster is active you can start vLLM with the following command in the Ray cluster head pod:
```bash
oc exec -it <ray-head-pod> -- python3 -m vllm.entrypoints.openai.api_server \
    --model /app/model \
    --tensor-parallel-size 16 \
    --host 0.0.0.0 \
    --port 8000
```
The *--tensor-parallel-size 16* is used to instruct vLLM to distribute the model across all 16 GPUs in the Ray cluster. Ray coordinates the distribution by assigning tasks to the head and worker nodes, each contributing their 2 GPUs to the total pool of 16 GPUs. The nvidia.com/gpu: "2" setting ensures each node provides exactly 2 GPUs to the collective tensor parallelism.

**Note:** The --model /app/model flag points to the PVC-mounted directory containing the model files (e.g., /app/model/config.json, /app/model/model-*.safetensors).

### Setup RayCluster and start vLLM (production)
Rather than creating a separate Deployment for vLLM, the most OpenShift-native approach in our scenario is to modify the existing RayCluster (see *raycluster2.yaml*) to start vLLM directly in the head pod. This keeps vLLM tightly coupled with Ray’s distributed framework, leveraging the RayCluster’s coordination across 16 GPUs. A standalone Deployment could work but would require additional logic to connect to the Ray workers, which RayCluster already handles.

### Monitor GPU Usage
During vLLM inference, verify all 2 GPUs per pod are utilized, and Ray coordinates across all 16 GPUs:
```bash
oc exec -it $(oc get pod -l ray.io/node-type=head -n vllm-llama3 -o name) -- nvidia-smi
```

## LiteLLM Wrapper Class
The litellm-wrapper directory contains a LiteLLM wrapper library and usage examples showcasing its functionality.

Note: This library is an excerpt from a private repository, curated for sharing purposes. It showcases selected concepts and configurations without disclosing sensitive details. Being an excerpt, additional code or configuration may be required for a fully functional implementation.

## Links
* [Nvidia GPU Operator Github](https://github.com/NVIDIA/gpu-operator)
* [Kuberay Operator Github](https://github.com/ray-project/kuberay)
* [GPU usage with KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/gpu.html)
* [CUDA and cuDNN images from gitlab.com/nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/)
* [litellm docs](https://docs.litellm.ai/docs/)
