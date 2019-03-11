from collections import OrderedDict
import csv
import json
import os
import shutil
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from tempfile import mkdtemp
from zipfile import ZipFile

import numpy as np
from PIL import Image
import tensorflow as tf

from utils.metering import AccuracyMeter


LABELS_CSV_FILENAME = 'labels.csv'
PREDICTIONS_CSV_FILENAME = 'predictions.csv'


def _generate_raw_data_from_zip(input_filepath):
    assert input_filepath.endswith('.zip')

    with ZipFile(input_filepath, 'r') as input_zip:
        labels_csv = input_zip.read(LABELS_CSV_FILENAME)
        csv_reader = csv.reader(labels_csv.splitlines())
        for image_filename, label in csv_reader:
            # assert image_filename.endswith('.png'), \
            #     'Input files should be PNG'

            image_id = image_filename
            image_data = input_zip.read(image_filename)

            buffer = StringIO()
            buffer.write(image_data)
            buffer.seek(0)

            image = Image.open(buffer)
            image.convert('RGB')
            image = np.array(image)
            image = image[...,:3]

            label = int(label.strip())

            yield image_id, image, label


def create_dataset_from_zipfile(input_filepath):
    dataset = tf.data.Dataset.from_generator(
        _generate_raw_data_from_zip,
        (tf.string, tf.uint8, tf.int64),
        (tf.TensorShape([]),
         tf.TensorShape([None, None, 3]),
         tf.TensorShape([])),
        args=(input_filepath,))

    return dataset


def save_output_zipfile(
        iterator, output_file, output_dir,
        has_prediction=False, sess=None):

    output_zip_filepath = os.path.join(output_dir, output_file)

    if sess is None:
        sess = tf.get_default_session()
    assert sess is not None

    all_labels, all_predictions = OrderedDict(), OrderedDict()
    accuracy = AccuracyMeter()

    if has_prediction:
        image_id, image, label, prediction = iterator.get_next()

    else:
        image_id, image, label = iterator.get_next()
        prediction = tf.zeros(1, dtype=tf.uint8)[0]

    temp_output_dir = mkdtemp()
    with ZipFile(output_zip_filepath, 'w') as output_zip:
        while True:
            try:
                image_id_eval, image_eval, label_eval, prediction_eval = \
                    sess.run([image_id, image, label, prediction])

                if len(image_eval.shape) == 4:
                    image_eval = image_eval[0]

                all_labels[image_id_eval] = label_eval
                all_predictions[image_id_eval] = prediction_eval

                accuracy.offer([prediction_eval], [label_eval],
                               ids=[image_id_eval])

                temp_image_filepath = \
                    os.path.join(temp_output_dir, image_id_eval)

                im = Image.fromarray(image_eval)
                im.save(temp_image_filepath)

                output_zip.write(temp_image_filepath, image_id_eval)

            except tf.errors.OutOfRangeError:
                break

        temp_labels_filepath = \
            os.path.join(temp_output_dir, LABELS_CSV_FILENAME)

        with open(temp_labels_filepath, 'w') as temp_labels_file:
            writer = csv.writer(temp_labels_file)
            for k, v in all_labels.items():
                writer.writerow([k, v])
        output_zip.write(temp_labels_filepath, LABELS_CSV_FILENAME)

    preds_filepath = None
    if has_prediction:
        preds_filepath = os.path.join(
            output_dir, output_file+'-'+PREDICTIONS_CSV_FILENAME)
        with open(preds_filepath, 'w') as preds_file:
            writer = csv.writer(preds_file)
            for k, v in all_predictions.items():
                writer.writerow([k, v])

    shutil.rmtree(temp_output_dir)

    return preds_filepath, accuracy
