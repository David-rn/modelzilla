import argparse
from functools import partial

from modelzilla.plugins import discover_plugins
from modelzilla.media import file_output_sink, plot_output_sink
from modelzilla.pipeline import CLIPluginPipeline


def make_parser():
    parser = argparse.ArgumentParser(
        description="Plugin system with dynamically loaded arguments."
    )
    parser.add_argument(
        "-i",
        "--input_media",
        required=True,
        help="Input media. It can be an image, a folder with images or a url",
    )
    parser.add_argument(
        "-os",
        "--output_sink",
        default="plot",
        choices=["file", "plot"],
        help="Select the output sink. It can be a file or a plot",
    )
    parser.add_argument(
        "-of",
        "--output_folder",
        default=None,
        help="Path to output folder. It must be specified if the output sink is a file",
    )
    parser.add_argument(
        "--plugins_folder", default=None, help="Path to custom plugins folder."
    )
    known_args, _ = parser.parse_known_args()
    known_args_dict = vars(known_args)

    if known_args.output_sink == "file" and known_args.output_folder is None:
        raise ValueError("Output folder must be specified if output sink is a file.")

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
    plugin_instance = plugin_class(**plugin_args)

    if args.output_sink == "file":
        output_sink = partial(file_output_sink, output_path=args.output_folder)
    elif args.output_sink == "plot":
        output_sink = plot_output_sink

    cli_pipeline = CLIPluginPipeline(args.input_media, plugin_instance, output_sink)
    cli_pipeline.run()


if __name__ == "__main__":
    main()
