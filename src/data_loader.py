import pandas as pd
import os
import yaml

class DataLoader:
    def __init__(self, config_path):
        """Initialize DataLoader with path to config file."""
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """Load and return config from yaml file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_raw_data(self):
        """Load raw data from path specified in config."""
        # Go up one level from config folder to get project root
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.config_path)))
        data_path = os.path.join(root_dir, self.config['paths']['raw_data'])
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        self.raw_data = pd.read_csv(data_path)
        print(f"Data loaded successfully: {self.raw_data.shape}")
        return self.raw_data    

if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
    
    data_loader = DataLoader(config_file)
    
    raw_data = data_loader.load_raw_data()
    
    print(raw_data.head())