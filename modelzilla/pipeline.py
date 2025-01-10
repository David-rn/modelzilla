from modelzilla.media import prepare_input
from modelzilla.plugins import CLIPlugin
from modelzilla.sinks import CallableSink


def run_cli_plugin_pipeline(
    input_media: str, plugin: CLIPlugin, output_sink: CallableSink
):
    for media in prepare_input(input_media):
        result = plugin.inference(media.item)
        output_sink(result, media)
