import tensorflow as tf

from constants import *
from utils.slim.preprocessing.inception_preprocessing \
    import preprocess_image


def perform_prediction(model):
    def _evaluation_fn(image_id, image, label):
        image = preprocess_image(
            image, IMAGE_SIZE, IMAGE_SIZE,
            is_training=False, cropping=False)
        image_expanded = tf.expand_dims(image, 0)
        image_rescaled = tf.cast(255. * (image + 1.) / 2., tf.uint8)

        prediction = tf.argmax(model.fprop(image_expanded)['probs'], 1)
        prediction = prediction[0]

        return image_id, image_rescaled, label, prediction

    return _evaluation_fn
