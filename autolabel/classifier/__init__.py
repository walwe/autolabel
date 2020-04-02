__all__ = [
    'classifier'
]


from .resnet import classifiers as resnet_classifier

classifier = {
    cls.NAME: cls
    for cls in resnet_classifier
}
