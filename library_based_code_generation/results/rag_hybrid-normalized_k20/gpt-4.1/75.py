```python
import tensorflow as tf
import numpy as np
import paz
from paz.backend.image import resize_image, show_image
from paz.backend.boxes import denormalize_boxes
from paz.pipelines import Processor
from paz.processors import (
    MatchBoxes, EncodeBoxes, ToOneHotVector, ResizeImage, SubtractMeanImage,
    NormalizeImage, RandomContrast, RandomBrightness, RandomSaturation,
    RandomHue, BlendRandomCropBackground, ToImageBoxCoordinates,
    ExpandBoxes, RandomSampleCrop, RandomFlipBoxes, DrawBoxes2D,
    DecodeBoxes, DenormalizeBoxes, LoadImage, Wrap, Compose
)
from paz.datasets import VOC
from paz.models import SSD300

class PreprocessBoxes(Processor):
    def __init__(self, prior_boxes, num_classes, match_threshold=0.5):
        super(PreprocessBoxes, self).__init__()
        self.match_boxes = MatchBoxes(prior_boxes, match_threshold)
        self.encode_boxes = EncodeBoxes(prior_boxes)
        self.to_one_hot = ToOneHotVector(num_classes)

    def call(self, boxes):
        matched_boxes = self.match_boxes(boxes)
        encoded_boxes = self.encode_boxes(matched_boxes)
        one_hot_boxes = self.to_one_hot(encoded_boxes)
        return one_hot_boxes

class PreprocessImage(Processor):
    def __init__(self, image_size, mean=None, normalize=False):
        super(PreprocessImage, self).__init__()
        self.resize = ResizeImage(image_size)
        self.mean = mean
        self.normalize = normalize
        if mean is not None:
            self.subtract_mean = SubtractMeanImage(mean)
        else:
            self.subtract_mean = None

    def call(self, image):
        image = self.resize(image)
        if self.subtract_mean is not None:
            image = self.subtract_mean(image)
        if self.normalize:
            image = NormalizeImage()(image)
        return image

class AugmentImage(Processor):
    def __init__(self, image_size, backgrounds=None):
        super(AugmentImage, self).__init__()
        self.resize = ResizeImage(image_size)
        self.backgrounds = backgrounds
        self.blend_background = BlendRandomCropBackground(backgrounds) if backgrounds else None
        self.random_contrast = RandomContrast()
        self.random_brightness = RandomBrightness()
        self.random_saturation = RandomSaturation()
        self.random_hue = RandomHue()

    def call(self, image):
        image = self.resize(image)
        if self.blend_background:
            image = self.blend_background(image)
        image = self.random_contrast(image)
        image = self.random_brightness(image)
        image = self.random_saturation(image)
        image = self.random_hue(image)
        return image

class AugmentBoxes(Processor):
    def __init__(self, image_size):
        super(AugmentBoxes, self).__init__()
        self.to_image_box = ToImageBoxCoordinates(image_size)
        self.expand = ExpandBoxes()
        self.random_crop = RandomSampleCrop()
        self.random_flip = RandomFlipBoxes()

    def call(self, boxes, image_shape):
        boxes = self.to_image_box(boxes, image_shape)
        boxes = self.expand(boxes)
        boxes = self.random_crop(boxes)
        boxes = self.random_flip(boxes)
        return boxes

class DrawBoxData2D(Processor):
    def __init__(self, class_names, colors=None):
        super(DrawBoxData2D, self).__init__()
        self.draw_boxes = DrawBoxes2D(class_names, colors)

    def call(self, image, boxes):
        return self.draw_boxes(image, boxes)

class ShowBoxes(Processor):
    def __init__(self, image_size, prior_boxes, class_names, mean=None):
        super(ShowBoxes, self).__init__()
        self.resize = ResizeImage(image_size)
        self.decode = DecodeBoxes(prior_boxes)
        self.denormalize = DenormalizeBoxes(image_size)
        self.draw = DrawBoxes2D(class_names)
        self.mean = mean

    def call(self, image, boxes):
        image = self.resize(image)
        boxes = self.decode(boxes)
        boxes = self.denormalize(boxes)
        image = self.draw(image, boxes)
        if self.mean is not None:
            image = image + self.mean
            image = np.clip(image, 0, 255).astype(np.uint8)
        show_image(image)
        return image

class AugmentDetection(Processor):
    def __init__(self, image_size, prior_boxes, num_classes, class_names,
                 mean=None, backgrounds=None, training=True):
        super(AugmentDetection, self).__init__()
        self.load_image = LoadImage()
        self.training = training
        self.augment_image = AugmentImage(image_size, backgrounds)
        self.augment_boxes = AugmentBoxes(image_size)
        self.preprocess_image = PreprocessImage(image_size, mean, normalize=True)
        self.preprocess_boxes = PreprocessBoxes(prior_boxes, num_classes)
        self.wrap = Wrap()

    def call(self, sample):
        image = self.load_image(sample['image'])
        boxes = sample['boxes']
        if self.training:
            image = self.augment_image(image)
            boxes = self.augment_boxes(boxes, image.shape)
        image = self.preprocess_image(image)
        boxes = self.preprocess_boxes(boxes)
        return self.wrap({'image': image, 'boxes': boxes})

if __name__ == '__main__':
    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load model and data
    image_size = (300, 300)
    num_classes = 21
    class_names = VOC.class_names
    prior_boxes = paz.models.ssd.prior_boxes.SSD300(image_size, num_classes).prior_boxes
    mean = np.array([123., 117., 104.])
    dataset = VOC('train')

    # Processors
    augment_detection = AugmentDetection(
        image_size, prior_boxes, num_classes, class_names, mean=mean, training=True)
    show_boxes = ShowBoxes(image_size, prior_boxes, class_names, mean=mean)

    # Apply processors to dataset
    for sample in dataset:
        processed = augment_detection(sample)
        image = processed['image']
        boxes = processed['boxes']
        show_boxes(image, boxes)
```