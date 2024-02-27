import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(signal, fs, filter_order=6, low_cut=0.5, high_cut=40):
    """
    Apply a bandpass filter to an ECG signal.

    Parameters:
    - signal (numpy array): The ECG signal.
    - fs (float): Sampling frequency of the ECG signal.
    - filter_order (int): Order of the filter.
    - low_cut (float): Low cutoff frequency for the high-pass filter.
    - high_cut (float): High cutoff frequency for the low-pass filter.

    Returns:
    - numpy array: Filtered ECG signal.
    """
    
    if fs <= high_cut * 2:
        raise ValueError("Sampling frequency is too low for the selected high cutoff frequency.")

    nyquist_freq = 0.5 * fs
    low = low_cut / nyquist_freq
    high = high_cut / nyquist_freq
    
    # Apply the bandpass filter
    sos = butter(filter_order, [low, high], btype='band', output='sos', analog=False)
    
    if len(np.shape(signal)) == 2:
        # Multi-channel ECG signal
        [num_samples, num_channels] = np.shape(signal)
        fsig = np.zeros([num_samples, num_channels])
        for i in range(num_channels):
            fsig[:,i] = sosfiltfilt(sos, signal[:,i])
    elif len(np.shape(signal)) == 1:
        # Single-channel ECG signal
        fsig = sosfiltfilt(sos, signal)
        
    return fsig
