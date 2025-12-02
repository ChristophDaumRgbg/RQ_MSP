import json
import os


config_file = f'FILE_PATH'

if not os.path.exists(config_file):
    raise FileNotFoundError(f"Config file not found: {config_file}")

with open(config_file) as f:
    _config = json.load(f)

# Make config values accessible as variables
data_path = _config['data_path']
output_dir = _config['output_dir']
plot_dir = _config['plot_dir']
asr_model = _config['asr_model']
asr_dataset = _config['asr_dataset']
gpu_device = _config['gpu_device']
commonvoice_path = _config['commonvoice_path']
