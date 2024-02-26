import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'physionet.org', 'files', 'challenge-2017', '1.0.0')
TRAINING_DIR = os.path.join(DATA_DIR, 'training')
TESTING_DIR = os.path.join(DATA_DIR, 'validation')
FIGURE_DIR = os.path.join(BASE_DIR, 'figures')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')

