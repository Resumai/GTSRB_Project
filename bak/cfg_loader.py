import importlib

def load_config(config_name: str = "cnn_config"):
    try:
        config_module = importlib.import_module(f"configs.{config_name}")
        return config_module
    except ModuleNotFoundError:
        raise ValueError(f"Config '{config_name}' does not exist in configs/")
