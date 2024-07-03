import torch
from sklearn.metrics import accuracy_score, precision_score

def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error loading config: {exc}")

def load_model(path):
    pass

def evaluate(model, dataloader):
    model.eval()

    with torch.no_grad():
    
    

def main():
    pass

if __name__ == "main":
    main()