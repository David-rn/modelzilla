import argparse
from plugins import discover_plugins
from pathlib import Path
from PIL import Image


def make_parser():
    parser = argparse.ArgumentParser(
        description="Plugin system with dynamically loaded arguments."
    )

    parser.add_argument(
        "--input_media",
        required=True,
        help="Input media. It can be an image, a folder with images or a url",
    )
    # parser.add_argument(
    #     "--output_folder",
    #     default="./",
    #     help="Path to output folder. If no specified, the output will be in the inn the current folder",
    # )
    parser.add_argument("--plugins_folder", default=None, help="Path to custom plugins folder.")
    known_args, _ = parser.parse_known_args()
    known_args_dict = vars(known_args)

    plugins = discover_plugins(known_args.plugins_folder)

    subparsers = parser.add_subparsers(
        dest="plugin_name", help="Select which plugin to use"
    )

    for plugin_name, plugin_class in plugins.items():
        plugin_class = plugins[plugin_name]
        plugin_parser = subparsers.add_parser(
            plugin_name, help=f"Arguments for {plugin_name}"
        )
        plugin_class.build_cmd_parser(plugin_parser)

    args = parser.parse_args()
    args_dict = vars(args)

    common_keys = set(known_args_dict.keys()).intersection(args_dict.keys())
    plugin_args = {k: v for k, v in args_dict.items() if k not in common_keys}
    plugin_args.pop("plugin_name")

    return args, plugin_args


def main():
    args, plugin_args = make_parser()
    print(f"Parsed arguments: {vars(args)}")

    plugins = discover_plugins(args.plugins_folder)

    plugin_class = plugins[args.plugin_name]
    instance = plugin_class(**plugin_args)

    input_path = args.input_media
    if input_path.startswith("http"):
        import requests
        image = Image.open(requests.get(input_path, stream=True).raw)
        instance.inference(image)
    elif Path(input_path).is_dir():
        for image_path in Path(input_path).glob("**/*.jpg"):
            if image_path.is_file():
                image = Image.open(image_path)
                instance.inference(image)
    elif Path(input_path).is_file():
        image = Image.open(input_path)
        instance.inference(image)
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    
    
    


if __name__ == "__main__":
    main()
