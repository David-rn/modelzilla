from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
import supervision as sv

from modelzilla.plugins import CLIPlugin


class HFObjectDetection(CLIPlugin):
    """Example plugin that uses the HuggingFace Object Detection model.
    The parameters in the __init__ method are dynamically added to the CLI.
    This example plugin can execute any object detection model from the HuggingFace.

    Example:
        ```shell
        modelzilla -i image.png -os plot HFObjectDetection --model_repo facebook/detr-resnet-50
        ```

    """

    def __init__(self, model_repo: str, device: str = "cpu"):
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_repo)
        self.model = AutoModelForObjectDetection.from_pretrained(model_repo)
        self.model = self.model.to(device)

    def inference(self, image) -> sv.Detections:
        with torch.no_grad():
            inputs = self.image_processor(images=[image], return_tensors="pt")
            outputs = self.model(**inputs.to(self.device))
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])

            results = self.image_processor.post_process_object_detection(
                outputs, threshold=0.3, target_sizes=target_sizes
            )[0]

            return sv.Detections.from_transformers(
                transformers_results=results, id2label=self.model.config.id2label
            )
