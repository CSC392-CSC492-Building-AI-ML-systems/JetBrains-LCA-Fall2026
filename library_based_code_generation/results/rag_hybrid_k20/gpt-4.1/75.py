```python
import tensorflow as tf
import numpy as np
import paz
from paz.backend.image import resize_image, show_image
from paz.backend.boxes import denormalize_boxes
from paz.pipelines import Processor
from paz.processors import (
    MatchBoxes, EncodeBoxes, ToOneHotVector, ResizeImage, SubtractMeanImage,
    NormalizeImage, BlendRandomCropBackground, RandomContrast, RandomBrightness,
    RandomSaturation, RandomHue, ToImageBoxCoordinates, ExpandBoxes,
    RandomSampleCrop, RandomFlipBoxes, DrawBoxes2D, DecodeBoxes, DenormalizeBoxes,
    LoadImage, Wrap, Compose
)
from paz.datasets import VOC

class PreprocessBoxes(Processor):
    def __init__(self, prior_boxes, num_classes, match_threshold=0.5):
        super().__init__()
        self.match = MatchBoxes(prior_boxes, match_threshold)
        self.encode = EncodeBoxes(prior_boxes)
        self.to_one_hot = ToOneHotVector(num_classes)

    def call(self, boxes):
        matched_boxes = self.match(boxes)
        encoded_boxes = self.encode(matched_boxes)
        encoded_boxes['class_vector'] = self.to_one_hot(encoded_boxes['class_id'])
        return encoded_boxes

class PreprocessImage(Processor):
    def __init__(self, image_size, mean=None, normalize=False):
        super().__init__()
        self.resize = ResizeImage(image_size)
        self.mean = mean
        self.normalize = normalize

    def call(self, image):
        image = self.resize(image)
        if self.mean is not None:
            image = SubtractMeanImage(self.mean)(image)
        if self.normalize:
            image = NormalizeImage()(image)
        return image

class AugmentImage(Processor):
    def __init__(self, image_size, mean=None):
        super().__init__()
        self.resize = ResizeImage(image_size)
        self.blend_bg = BlendRandomCropBackground()
        self.random_contrast = RandomContrast()
        self.random_brightness = RandomBrightness()
        self.random_saturation = RandomSaturation()
        self.random_hue = RandomHue()
        self.mean = mean

    def call(self, image):
        image = self.resize(image)
        image = self.blend_bg(image)
        image = self.random_contrast(image)
        image = self.random_brightness(image)
        image = self.random_saturation(image)
        image = self.random_hue(image)
        if self.mean is not None:
            image = SubtractMeanImage(self.mean)(image)
        return image

class AugmentBoxes(Processor):
    def __init__(self, image_size):
        super().__init__()
        self.to_image_coords = ToImageBoxCoordinates(image_size)
        self.expand = ExpandBoxes()
        self.random_crop = RandomSampleCrop()
        self.random_flip = RandomFlipBoxes()

    def call(self, boxes, image_shape):
        boxes = self.to_image_coords(boxes, image_shape)
        boxes = self.expand(boxes)
        boxes = self.random_crop(boxes)
        boxes = self.random_flip(boxes)
        return boxes

class DrawBoxData2D(Processor):
    def __init__(self, class_names, colors=None):
        super().__init__()
        self.draw = DrawBoxes2D(class_names, colors)

    def call(self, image, boxes):
        return self.draw(image, boxes)

class ShowBoxes(Processor):
    def __init__(self, image_size, prior_boxes, class_names, colors=None):
        super().__init__()
        self.resize = ResizeImage(image_size)
        self.decode = DecodeBoxes(prior_boxes)
        self.denormalize = DenormalizeBoxes(image_size)
        self.draw = DrawBoxes2D(class_names, colors)

    def call(self, image, encoded_boxes):
        image = self.resize(image)
        boxes = self.decode(encoded_boxes)
        boxes = self.denormalize(boxes)
        image = self.draw(image, boxes)
        show_image(image)
        return image

class AugmentDetection(Processor):
    def __init__(self, image_size, prior_boxes, num_classes, class_names, mean=None, training=True):
        super().__init__()
        self.load_image = LoadImage()
        self.training = training
        self.augment_image = AugmentImage(image_size, mean)
        self.augment_boxes = AugmentBoxes(image_size)
        self.preprocess_image = PreprocessImage(image_size, mean)
        self.preprocess_boxes = PreprocessBoxes(prior_boxes, num_classes)
        self.wrap = Wrap()

    def call(self, sample):
        image = self.load_image(sample['image_path'])
        boxes = sample['boxes']
        if self.training:
            image = self.augment_image(image)
            boxes = self.augment_boxes(boxes, image.shape)
        image = self.preprocess_image(image)
        encoded_boxes = self.preprocess_boxes(boxes)
        return self.wrap({'image': image, 'boxes': encoded_boxes})

if __name__ == '__main__':
    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load model and data
    model = paz.models.mini_SSD()
    dataset = VOC('train', 'VOCdevkit')
    prior_boxes = create_prior_boxes(model, dataset)
    num_classes = len(dataset.class_names)
    image_size = (300, 300)
    mean = [123.68, 116.779, 103.939]

    augment_detection = AugmentDetection(
        image_size, prior_boxes, num_classes, dataset.class_names, mean, training=True)
    show_boxes = ShowBoxes(image_size, prior_boxes, dataset.class_names)

    for sample in dataset:
        processed = augment_detection(sample)
        image = processed['image']
        boxes = processed['boxes']
        show_boxes(image, boxes)
```