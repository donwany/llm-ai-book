# Clone the LQ-LoRA repository and set up the environment
# git clone https://github.com/HanGuo97/lq-lora.git
# cd lq-lora
# bash scripts/setup.sh

from transformers import AutoTokenizer, AutoModelForCausalLM
from models import lora_utils

# Configuration parameters
data = "c4"            # Data type for quantization
budget = "2.75"        # Target bits for quantization
model_size = "7b"      # Model size (options: 7b or 70b)

# Load the base model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{model_size}-hf")

# Prepare the model for LoRA (Low-Rank Adaptation) with specified parameters
model = lora_utils.prepare_model_for_lora(
    model=model,
    num_ranks=64,
    lora_alpha=16,
    lora_dropout=0.0,
    use_gradient_checkpointing=True
)

# Apply LQ-LoRA (Low-Precision Quantization with LoRA) to the model
lora_utils.transform_lora_layers(
    lpq=True,
    model=model,
    model_name=f"llama-2-{model_size}/lpq-64/{data},budget={budget}",
    device="cuda"
)
