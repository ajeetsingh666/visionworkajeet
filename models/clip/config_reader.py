import os
import yaml

class YAMLConfigReader:
    def __init__(self, config_file: str):
        self.config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)

    def read_config(self):
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config
