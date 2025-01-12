# Modelzilla

This library turns any model class into a CLI executable. If you're tired of writing a lot of boilerplate code to run your model, this library is for you.
It is a lightweight Python package that enables developers to transform any AI model into a fully functional Command-Line Interface (CLI) plugin.

Key features:
- **Automatic CLI Generation**: Automatically generate CLI commands for models
- **Customizable Commands**: Each model will have its own CLI parameters
- **Seamless Integration**: Just inherit from the `CLIPlugin` class and you're ready to go

## Installation

Pip install the modelzilla package in a Python>=3.10 environment.

```shell
pip install modelzilla
```

## Quickstart

### 1. Turn any model into a CLI executable
Let's say you have a model class that you want to turn into a CLI executable. You can do this by inheriting from the `CLIPlugin` class and implementing the `inference` method.

```python
import supervision as sv

from modelzilla.plugins import CLIPlugin # Import the CLIPlugin class

class MyModel(CLIPlugin):
    def __init__(self, model_path: str): # Add any parameters you want to the CLI
        self.model = load(model_path) # Load your model

    def inference(self, image) -> sv.Detections:
        results = self.model(image)
        return results.to_detections() # Convert your model's output to sv.Detections
```

### 2. Execute the model from the CLI
You can execute your model from the CLI. The parameters included into the `__init__` method will be automatically added to the CLI.
If the model class in inside the `plugins` folder:

```shell
modelzilla -i image.png -os plot MyModel --model_path model.pth
```

Otherwise, you need to specify the `--plugins_folder` argument:

```shell
modelzilla -i image.png -os plot --plugins_folder <path/to/your/plugin/folder> MyModel --model_path model.pth
```

## Examples

Currently, we provide the following plugins:

- [HFObjectDetection](https://github.com/David-rn/modelzilla/tree/main/modelzilla/plugins/hf_object_det.py): A plugin that uses the HuggingFace Object Detection model.


### How to execute it from the CLI

```shell
modelzilla -i http://images.cocodataset.org/val2017/000000039769.jpg -os plot HFObjectDetection --model_repo facebook/detr-resnet-50
```
