from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
import supervision as sv

from modelzilla.plugins import IPlugin, CLIArgparse

class HFObjectDetection(IPlugin, CLIArgparse):
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
                transformers_results=results,
                id2label=self.model.config.id2label
            )
