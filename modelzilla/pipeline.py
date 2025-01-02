from typing import Callable

from modelzilla.media import prepare_input
from modelzilla.plugins import IPlugin


class CLIPluginPipeline:
    def __init__(self, input_media: str, plugin: IPlugin, output_sink: Callable):
        self.input_media = input_media
        self.plugin = plugin
        self.output_sink = output_sink
    
    def run(self):
        for media in prepare_input(self.input_media):
            result = self.plugin.inference(media.item)
            self.output_sink(result, media)
