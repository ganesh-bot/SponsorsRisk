# run_extract.py

import yaml
from extract_features import extract_aact_data

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()

    conn_info = config['db']
    output_path = config['output']['path']

    extract_aact_data(conn_info, output_path)
