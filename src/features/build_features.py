import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import stats
import neurokit2 as nk
from tqdm import tqdm
from src.data.load_data import parse_header
from sklearn.base import BaseEstimator, TransformerMixin


class ECGFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.build_features(X)
    
    def fit_transform(self, X, y=None):
        return self.build_features(X)
    
    @staticmethod
    def build_features(dataset):
        features = []

        for data_dict in tqdm(dataset, desc='Extracting features', total=len(dataset)):
            ecg_signal = data_dict['ecg_signal']
            patient_id = data_dict['patient_id']
            label = data_dict['label']
            
            header = parse_header(data_dict['hea_file'])
            sampling_rate = header['sample_rate']
            
            try:
                feature_dict = ECGFeatureExtractor.extract_features(ecg_signal, sampling_rate)
                feature_dict['label'] = label
                features.append(feature_dict)
            except ValueError as e:
                if str(e) == "NeuroKit error: the window cannot contain more data points than the time series. Decrease 'scale'.":
                    continue
                if str(e) == "cannot convert float NaN to integer":
                    continue
                else:
                    plt.plot(ecg_signal)
                    plt.title(f'Patient {patient_id} ECG {label}')
                    plt.show()
                    raise
            except ZeroDivisionError as e:
                continue
            except Exception as e:
                plt.plot(ecg_signal)
                plt.title(f'Patient {patient_id} ECG {label}')
                plt.show()
                raise
        
        features_df = pd.DataFrame(features)
        features_df.drop(columns=['HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'LF', 'HF', 'LF_HF_ratio'], inplace=True)
        
        return features_df
    
    @staticmethod
    def extract_features(ecg_signal, fs):
        # ECG processing to find R-peaks and segment the signal
        _, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
        
        # hrv features
        hrv_features = nk.hrv_time(info['ECG_R_Peaks'], sampling_rate=fs)
        hrv_features = hrv_features.to_dict('records')[0]
        
        # R-R Interval features
        rri = np.diff(info['ECG_R_Peaks']) / fs * 1000 # convert to ms
        rr_features = {
            'RR_mean': np.mean(rri),
            'RR_std': np.std(rri),
            'Irregularity_index': np.sum(np.abs(np.diff(rri)) > 50) / len(rri)
        }
        
        # Frequency Domain Features
        f, Pxx = sig.welch(ecg_signal, fs=fs)
        lf = np.trapz(Pxx[(f >= 0.04) & (f <= 0.15)])  # Low frequency power
        hf = np.trapz(Pxx[(f >= 0.15) & (f <= 0.4)])   # High frequency power
        freq_features = {
            'LF': lf,
            'HF': hf,
            'LF_HF_ratio': lf / hf if hf > 1e-10 else np.nan
        }

        # Statistical Features
        stat_features = {
            'Skewness': stats.skew(ecg_signal),
            'Kurtosis': stats.kurtosis(ecg_signal)
        }

        # Combine all features
        features = {
            **hrv_features,
            **rr_features,
            **freq_features,
            **stat_features
        }
        
        return features
