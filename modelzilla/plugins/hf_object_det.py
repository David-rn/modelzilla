from plugins import IPlugin, ArgparseMixin
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch


class HF_Object_Detection(IPlugin, ArgparseMixin):
    def __init__(self, model_repo: str, device: str = "cpu"):
        self.model_repo = model_repo
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_repo)
        self.model = AutoModelForObjectDetection.from_pretrained(model_repo)
        self.model = self.model.to(device)

    def inference(self, image):
        with torch.no_grad():
            inputs = self.image_processor(images=[image], return_tensors="pt")
            outputs = self.model(**inputs.to(self.device))
            target_sizes = torch.tensor([[image.size[1], image.size[0]]])
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=0.3, target_sizes=target_sizes
            )[0]

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
