import yaml


def load_configs(config_path="./configs/configs.yml"):
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    return configs