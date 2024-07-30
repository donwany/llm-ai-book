# Clone the repository
# git clone https://github.com/cg123/mergekit.git
import os

# Navigate into the project directory and install the package
# cd mergekit && pip install -q -e .

import yaml

# Define the model name
MODEL_NAME = "worldboss-7B-slerp"

# YAML configuration for model merging
yaml_config = """
slices:
  - sources:
      - model: psmathur/orca_mini_v3_13b
        layer_range: [0, 40]
      - model: garage-bAInd/Platypus2-13B
        layer_range: [0, 40]
merge_method: slerp
base_model: psmathur/orca_mini_v3_13b
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5  # Fallback for the rest of tensors
dtype: float16
"""

# Save the YAML configuration to a file
with open('config.yaml', 'w', encoding="utf-8") as f:
    f.write(yaml_config)

# Merge models using CLI
os.system("mergekit-yaml config.yaml merge \
    --copy-tokenizer \
    --allow-crimes \
    --out-shard-size 1B \
    --lazy-unpickle"
          )

# Key Steps:
# 1. **Clone and Install**: Instructions for cloning the repository and installing the package.
# 2. **YAML Configuration**: Defines the model merging parameters.
# 3. **File Saving**: Saves the YAML configuration to `config.yaml`.
# 4. **CLI Command**: Shows how to run the merge command with specified options.
#
# This format separates the script parts from the commands and comments to keep everything clear.
