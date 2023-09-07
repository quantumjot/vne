import yaml

# Define a function to read the YAML file and store variables
def read_yaml_file(file_path):
    with open(file_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data

def get_config_values(yaml_file_path):
    config_data = read_yaml_file(yaml_file_path)
    return config_data
