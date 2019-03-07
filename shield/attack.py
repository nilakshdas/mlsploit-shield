import tensorflow as tf

from constants import *
from utils.slim.preprocessing.inception_preprocessing \
    import preprocess_image


def perform_attack(model, attack_class, attack_options):
    attack = attack_class(model)

    def _attack_fn(image_id, image, label):
        image = preprocess_image(
            image, IMAGE_SIZE, IMAGE_SIZE,
            is_training=False, cropping=False)
        image_expanded = tf.expand_dims(image, 0)

        image_attacked = attack.generate(image_expanded, **attack_options)
        image_attacked = image_attacked[0]
        image_attacked = tf.cast(255. * (image_attacked + 1.) / 2., tf.uint8)

        return image_id, image_attacked, label

    return _attack_fn
