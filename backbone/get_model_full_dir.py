import os

main_dir = ''
trained_dir = ''
import yaml

# Load YAML configuration
with open("backbone/model_dir_config.yaml", "r") as f:
    model_dir_config = yaml.safe_load(f)


def get_model_full_dir(model_name):
    if model_name in model_dir_config:
        return os.path.join(main_dir, model_dir_config[model_name])
    else:
        raise NotImplementedError("Model directory not implemented")
