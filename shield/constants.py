import os as _os

BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
CHECKPOINTS_DIR = _os.path.join(BASE_DIR, 'scratch', 'checkpoints')
INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

CONTEXT_FILEPATH = _os.path.join(INPUT_DIR, 'input.json')
OUTPUT_ZIP_FILEPATH = _os.path.join(OUTPUT_DIR, 'output.zip')
OUTPUT_JSON_FILEPATH = _os.path.join(OUTPUT_DIR, 'output.json')

IMAGE_SIZE = 299
