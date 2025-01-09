# Modelzilla

This library turns any model class into a CLI executable. If you're tired of writing a lot of boilerplate code to run your model, this library is for you.

## Installation

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
Now you can execute your model from the CLI. The parameters you added to the `__init__` method will be automatically added to the CLI.
If the model class in inside the `plugins` folder:

```shell
modelzilla -i image.png -os plot MyModel --model_path model.pth
```

Otherwise, you need to specify the `--plugins_folder` argument:

```shell
modelzilla -i image.png -os plot --plugins_folder <path/to/your/plugin/folder> MyModel --model_path model.pth
```
