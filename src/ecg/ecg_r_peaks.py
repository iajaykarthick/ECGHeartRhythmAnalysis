import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_r_peaks(ecg_signal, sampling_rate, plot=False):
    # Normalize the ECG signal
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # Find peaks (R-peaks are usually the highest peaks in an ECG)
    # The height and distance parameters can be adjusted based on your specific ECG characteristics
    peaks, _ = find_peaks(ecg_signal, height=np.max(ecg_signal)*0.5, distance=sampling_rate*0.2)

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(ecg_signal)
        plt.plot(peaks, ecg_signal[peaks], "x")
        plt.title("Detected R-peaks in ECG")
        plt.show()

    return peaks
