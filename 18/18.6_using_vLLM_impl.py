# (Recommended) Create a new conda environment.
# conda create -n myenv python=3.9 -y
# conda activate myenv
# pip install vllm
# pip install langchain langchain_community -q
# export VLLM_VERSION=0.4.0
# export PYTHON_VERSION=39
# pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

# docker build -f Dockerfile.cpu -t vllm-cpu-env --shm-size=4g .

# docker run -it \
#     --rm --network=host \
#     --cpuset-cpus=<cpu-id-list, optional> \
#     --cpuset-mems=<memory-node, optional> \
#     vllm-cpu-env

# docker run --runtime nvidia --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
#     -p 8000:8000 \
#     --ipc=host \
#     vllm/vllm-openai:latest \
#     --model mistralai/Mistral-7B-v0.1

from vllm import LLM

llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Francisco is a")

# python -m vllm.entrypoints.api_server \
#     --model facebook/opt-13b --tensor-parallel-size 4

from langchain_community.llms import VLLM

llm = VLLM(model="mosaicml/mpt-7b",
           trust_remote_code=True,  # Mandatory for HF models
           max_new_tokens=128,
           top_k=10,
           top_p=0.95,
           temperature=0.8,
           # tensor_parallel_size=... # For distributed inference
)

print(llm("What is the capital of France?"))
