from typing import Callable, Any
from os.path import basename
from pathlib import Path

import matplotlib.pyplot as plt

from modelzilla.plugins import InferenceResult
from modelzilla.media import Media, annotate

CallableSink = Callable[[InferenceResult, Media], Any]


def file_output_sink(
    results: InferenceResult,
    media: Media,
    output_path: str,
):
    annotated_frame = annotate(media, results)
    annotated_frame.save(
        Path(output_path).joinpath(basename(media.path).split(".")[0] + ".jpg")
    )


def plot_output_sink(results: InferenceResult, media: Media):
    annotated_frame = annotate(media, results)
    _ = plt.imshow(annotated_frame)
    plt.show()
