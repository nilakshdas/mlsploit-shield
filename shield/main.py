from __future__ import print_function

import json
import os

from addict import Dict
import tensorflow as tf

from attack import perform_attack
from constants import *
from data import create_dataset_from_zipfile, save_output_zipfile
from defend import perform_defense
from evaluate import perform_prediction
import opts


def _load_context(context_filepath):
    context = json.load(open(context_filepath))

    # assert all(input_file.endswith('.zip')
    #            for input_file in context['files']), \
    #     'Input files should be .zip files'

    # identify action and action type
    if 'attack' in context['name']:
        context['action'] = 'attack'

        func_name = context['name']
        attack = func_name[func_name.find('[')+1: func_name.find(']')]
        attack = {'FGSM': 'fgsm',
                  'DeepFool': 'df',
                  'CarliniWagner': 'cwl2'
                  }[attack]
        context['action_type'] = attack

        # for FGSM, recalculate 'eps' from 'epsilon'
        if attack == 'fgsm':
            eps = context['options']['epsilon']
            context['options']['eps'] = 2. * eps / 255.
            del context['options']['epsilon']

    elif 'defend' in context['name']:
        context['action'] = 'defend'
        func_name = context['name']
        defense = func_name[func_name.find('[')+1: func_name.find(']')]
        defense = {'JPEG': 'jpeg',
                   'SLQ': 'slq',
                   'MedianFilter': 'median_filter',
                   'TV-Bregman': 'tv_bregman'
                   }[defense]
        context['action_type'] = defense

    elif 'evaluate' in context['name']:
        context['action'] = 'evaluate'
        context['action_type'] = None

    else:
        raise ValueError('Unknown action in input.json')

    # identify model
    if 'model' in context['options']:
        context['model'] = {'ResNet50-v2': 'resnet_50_v2',
                            'Inception-v4': 'inception_v4'
                            }[context['options']['model']]
    else:
        context['model'] = None

    print('Performing %s with %s...' % (
        context['action'],
        (context['action_type']
         if context['action_type'] is not None
         else context['model'])))

    return Dict(context)


def main():
    context = _load_context(CONTEXT_FILEPATH)

    model_name = context.model
    has_prediction = (context.action == 'evaluate')

    output_data = dict()
    output_data['name'] = context.name
    output_data['files'] = list()
    output_data['tags'] = list()
    output_data['files_modified'] = list()
    output_data['files_extra'] = list()

    for input_file, tags in zip(context.files, context.tags):
        if 'ftype' in tags and tags['ftype'] == 'predictions':
            continue

        assert input_file.endswith('.zip'), \
            'Input files should be .zip files'

        graph = tf.Graph()
        with graph.as_default():
            dataset = create_dataset_from_zipfile(
                os.path.join(INPUT_DIR, input_file))

            model = None
            if model_name is not None:
                model = opts.model_class_map[model_name](tf.placeholder(
                    tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3)))

            if context.action == 'attack':
                attack = context.action_type
                attack_class = opts.attack_class_map[attack]
                attack_options = opts.attack_options_map[attack].copy()
                attack_options.update(context.options)

                dataset = dataset.map(
                    perform_attack(model, attack_class, attack_options))

            elif context.action == 'defend':
                defense = context.action_type
                defense_fn = opts.defense_fn_map[defense]
                is_tf_defense = defense in opts.tf_defenses
                defense_options = opts.defense_options_map[defense].copy()
                defense_options.update(context.options)

                dataset = dataset.map(
                    perform_defense(defense_fn, defense_options,
                                    is_tf_defense=is_tf_defense,
                                    apply_cropping=True))

            elif context.action == 'evaluate':
                dataset = dataset.map(perform_prediction(model))

            iterator = dataset.make_initializable_iterator()

            sess = tf.Session(
                graph=graph,
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    gpu_options=tf.GPUOptions(
                        per_process_gpu_memory_fraction=1.,
                        allow_growth=True)))

            with sess.as_default():
                tf.local_variables_initializer().run()
                tf.global_variables_initializer().run()

                sess.run(iterator.initializer)

                if model_name is not None:
                    model_checkpoint_path = os.path.join(
                        CHECKPOINTS_DIR, opts.model_checkpoint_map[model_name])
                    model.load_weights(model_checkpoint_path, sess=sess)

                preds_filepath, accuracy = save_output_zipfile(
                    iterator, input_file,
                    OUTPUT_DIR, sess=sess,
                    has_prediction=has_prediction)

        if has_prediction:
            preds_file = os.path.basename(preds_filepath)
            output_data['files'].append(preds_file)
            output_data['files_extra'].append(preds_file)
            output_data['tags'].append({'accuracy': accuracy.evaluate()})

        else:
            output_data['files'].append(input_file)
            output_data['files_modified'].append(input_file)
            output_data['tags'].append({})

        tf.reset_default_graph()

    with open(OUTPUT_JSON_FILEPATH, 'w') as output_json_file:
        json.dump(output_data, output_json_file)


if __name__ == '__main__':
    main()
