# Autolabel

Autolabel is an image labeling tool.
Currently images are labeled using [ResNet18-152](https://arxiv.org/abs/1512.03385) implemented by [pytorch](https://pytorch.org/hub/pytorch_vision_resnet/).
Autolabel can be used as a cli tool or as a library.

## Installation
```
python setup.py install
```

## Command line usage
See `--help` for a command overview
```
Usage: autolabel [OPTIONS] [IMAGES]...

Options:
  --batch-size INTEGER
  --sep TEXT                      Separator
  --top INTEGER
  -o, --output FILENAME           Output file
  -m, --model [resnet18|resnet34|resnet50|resnet101|resnet152]
  --help                          Show this message and exit.
```

In the simplest form this mean:
```bash
autolabel image.jpg
```

Autolabel supports reading file names from STDIN:
```bash
find /myimages -type f -iname '*.jpg' | autolabel
```


## Library usage
```python
from autolabel.image import ImageListDataset
from autolabel.classifier.resnet import Resnet18Classifier
from pathlib import Path

classifier = Resnet18Classifier()
images = [Path('/path/to/image.jpg'), Path('/path/to/another/image.png')]
dataset = ImageListDataset(images)
res = classifier.predict(dataset, top=top)
for p, decoded in res.items():
    print(p, decoded)
```