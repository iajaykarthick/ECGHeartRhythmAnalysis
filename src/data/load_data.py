import os
import scipy.io as sio
from tqdm import tqdm
from src.config import DATA_DIR, TRAINING_DIR

def load_dataset(data_path):
    
    labels_file_path = os.path.join(DATA_DIR, 'REFERENCE-v3.csv')
    
    # read the labels file and get the patient ids and their corresponding labels
    with open(labels_file_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip().split(',') for label in labels]
        
    # ecg data files are inside subdirectories; label[0] is the patient id and patient_id[:3] is the subdirectory in which the ecg data is stored for that patient
    dataset = []
    for label in tqdm(labels):
        patient_id = label[0]
        label = label[1]
        hea_file = os.path.join(data_path, patient_id[:3], patient_id + '.hea')
        ecg_file = os.path.join(data_path, patient_id[:3], patient_id + '.mat')
        dataset.append(
            {
                'patient_id': patient_id,
                'label': label,
                'hea_file': hea_file,
                'ecg_file': ecg_file
            }
        )
    
    return dataset


def load_ecg(ecg_file):
    ecg = sio.loadmat(ecg_file)['val'].squeeze()
    return ecg


def parse_header(header_file):
    with open(header_file, 'r') as file:
        lines = file.readlines()

    header_info = {}
    
    # First line
    parts = lines[0].strip().split()
    header_info['record_name'] = parts[0]
    header_info['n_leads'] = int(parts[1])
    header_info['sample_rate'] = int(parts[2])
    header_info['n_samples'] = int(parts[3])
    header_info['datetime'] = parts[4] + ' ' + parts[5]

    # Second line
    parts = lines[1].strip().split()
    header_info['file_name'] = parts[0]
    header_info['signal_details'] = ' '.join(parts[1:])

    return header_info
