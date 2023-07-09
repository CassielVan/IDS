import argparse
import yaml


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    train_config = load_config(args.model_config_path)
    # model_config

    # dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path',  type=str, default='config/train_config.yaml')
    print("hello")
    args = parser.parse_args()
    main(args)


