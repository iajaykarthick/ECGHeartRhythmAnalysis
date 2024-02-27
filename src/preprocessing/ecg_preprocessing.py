import numpy as np
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt, lfilter

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.load_data import load_ecg, parse_header

class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, fs, filter_order=6, low_cut=0.5, high_cut=40, zero_phase=True):
        self.fs = fs
        self.filter_order = filter_order
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.zero_phase = zero_phase
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return BandpassFilter.bandpass_filter(X, self.fs, self.filter_order, self.low_cut, self.high_cut, self.zero_phase)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    @staticmethod
    def bandpass_filter(signal, fs, filter_order=6, low_cut=0.5, high_cut=40, zero_phase=True):
        """
        Apply a bandpass filter to an ECG signal with an option to choose zero-phase filtering.

        Parameters:
        - signal (numpy array): The ECG signal.
        - fs (float): Sampling frequency of the ECG signal.
        - filter_order (int): Order of the filter.
        - low_cut (float): Low cutoff frequency for the high-pass filter.
        - high_cut (float): High cutoff frequency for the low-pass filter.
        - zero_phase (bool): If True, use zero-phase filtering (sosfiltfilt). If False, use forward filtering (lfilter).

        Returns:
        - numpy array: Filtered ECG signal.
        """
        
        if fs <= high_cut * 2:
            raise ValueError("Sampling frequency is too low for the selected high cutoff frequency.")

        nyquist_freq = 0.5 * fs
        low = low_cut / nyquist_freq
        high = high_cut / nyquist_freq
        
        # Design the filter
        if zero_phase:
            sos = butter(filter_order, [low, high], btype='band', output='sos', analog=False)
        else:
            b, a = butter(filter_order, [low, high], btype='band', analog=False)
            
        # Apply the filter
        if len(np.shape(signal)) == 2:
            # Multi-channel ECG signal
            [num_samples, num_channels] = np.shape(signal)
            fsig = np.zeros([num_samples, num_channels])
            for i in range(num_channels):
                if zero_phase:
                    fsig[:, i] = sosfiltfilt(sos, signal[:, i])
                else:
                    fsig[:, i] = lfilter(b, a, signal[:, i])
        elif len(np.shape(signal)) == 1:
            # Single-channel ECG signal
            if zero_phase:
                fsig = sosfiltfilt(sos, signal)
            else:
                fsig = lfilter(b, a, signal)
            
        return fsig


class NormalizeECG(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return NormalizeECG.normalize_ecg(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    @staticmethod
    def normalize_ecg(signal):
        """
        Normalize an ECG signal using Min-Max normalization.

        Parameters:
        - signal (numpy array): The ECG signal.

        Returns:
        - numpy array: Normalized ECG signal.
        """
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        return normalized_signal
    
    
class SegmentECG(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, overlap_size):
        self.window_size = window_size
        self.overlap_size = overlap_size
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return SegmentECG.segment_ecg(X, self.window_size, self.overlap_size)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    @staticmethod
    def segment_ecg(signal, window_size, overlap_size):
        """
        Segment an ECG signal into fixed-size windows.

        Parameters:
        - signal (numpy array): The ECG signal.
        - window_size (int): The size of each window in samples.
        - overlap_size (int): The size of overlap between consecutive windows in samples.

        Returns:
        - list of numpy arrays: List of segmented ECG windows.
        """
        segments = []
        start = 0
        end = window_size
        while end <= len(signal):
            segment = signal[start:end]
            segments.append(segment)
            start += window_size - overlap_size
            end = start + window_size
        return segments


class ECGPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, overlap_size):
        self.window_size = window_size
        self.overlap_size = overlap_size

    def fit(self, X, y=None):
        return self

    def transform(self, dataset):
        processed_data = []

        for data_dict in tqdm(dataset, desc='Preprocessing ECG signals', total=len(dataset)):
            ecg = load_ecg(data_dict['ecg_file'])
            header = parse_header(data_dict['hea_file'])
            fs = header['sample_rate']

            pipeline = Pipeline([
                ('bandpass_filter', BandpassFilter(fs)),
                ('normalize_ecg', NormalizeECG()),
                ('segment_ecg', SegmentECG(window_size=self.window_size, overlap_size=self.overlap_size))
            ])

            segmented_ecg = pipeline.fit_transform(ecg)
            data_dict['segmented_ecg'] = segmented_ecg
            processed_data.append(data_dict)

        return processed_data

