from typing import Iterator
from dataclasses import dataclass
import requests
from pathlib import Path

import supervision as sv
from PIL import Image

from modelzilla.plugins import InferenceResult

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class Media:
    path: str
    item: Image.Image


def prepare_input(input_path: str) -> Iterator[Media]:
    if input_path.startswith("http"):
        image = Image.open(requests.get(input_path, stream=True).raw)
        yield Media(path=input_path, item=image)
    elif Path(input_path).is_dir():
        for ext in IMAGE_EXTENSIONS:
            for image_path in Path(input_path).glob(f"**/*{ext}"):
                if image_path.is_file():
                    image = Image.open(image_path)
                    yield Media(path=image_path, item=image)
    elif Path(input_path).is_file():
        image = Image.open(input_path)
        yield Media(path=input_path, item=image)
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def box_annotator(media: Media, results: sv.Detections) -> Image.Image:
    box_annotator = sv.BoxAnnotator()
    rich_label_annotator = sv.RichLabelAnnotator(text_position=sv.Position.TOP_LEFT)

    annotated_frame = box_annotator.annotate(
        scene=media.item.copy(), detections=results
    )

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(results["class_name"], results.confidence)
    ]

    annotated_frame = rich_label_annotator.annotate(
        scene=annotated_frame.copy(), detections=results, labels=labels
    )

    return annotated_frame


def keypoint_annotator(media: Media, results: sv.KeyPoints) -> Image.Image:
    raise NotImplementedError


def annotate(media: Media, results: InferenceResult) -> Image.Image:
    if isinstance(results, sv.Detections):
        return box_annotator(media, results)
    elif isinstance(results, sv.KeyPoints):
        return keypoint_annotator(media, results)
    else:
        raise ValueError(f"Invalid inference result type: {type(results)}")
