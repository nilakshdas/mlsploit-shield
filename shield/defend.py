import tensorflow as tf


def perform_defense(defense_fn, defense_options,
                    is_tf_defense=False,
                    apply_cropping=True):

    def _defense_fn(image_id, image, label):
        if apply_cropping:
            image = tf.image.central_crop(
                image, central_fraction=0.875)

        if is_tf_defense:
            image = defense_fn(image, **defense_options)
        else:
            image = tf.py_func(
                lambda x: (
                    lambda x_, o: defense_fn(x_, **o))
                (x, defense_options),
                [image], (tf.uint8,))

        return image_id, image, label

    return _defense_fn
