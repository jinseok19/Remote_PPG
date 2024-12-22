import cv2
import numpy as np
from scipy import signal

from timer import Timer


class rPPG:
    MAX_FPS = 30
    BPM_BAND = (42, 180)

    DURATION_RAW = 5
    DURATION_VISUALIZE = 5
    DURATION_ACCUMULATE_BPM = 1

    RAW_SIZE = MAX_FPS * DURATION_RAW
    VISUALIZE_SIZE = MAX_FPS * DURATION_VISUALIZE

    def __init__(self):
        # Reset buffers
        self.reset()

    def reset(self):
        self.raw = []
        self.visualize = []
        self.bpm_buffer = []
        self.bpm = 0

    def process(self, frame, sp, ep):
        # Calculate signal value
        crop = frame[sp[1]: ep[1], sp[0]: ep[0], ...]
        mask = self._get_skin_mask(crop)
        val = self._calculate_ppg_value(crop, mask)

        # Append buffer
        self.raw.append(val)
        self.raw = self.raw[-self.RAW_SIZE:]

        # Processing
        if len(self.raw) == self.RAW_SIZE:
            fps = Timer.get_fps()

            # Refine signal
            raw_signal = np.array(self.raw[-int(self.DURATION_RAW * fps):]).transpose()
            detrended = self._detrend_signal(raw_signal, fps)
            bandpassed = self._filter_bandpass(-detrended, fps, self.BPM_BAND)

            # Calculate bpm
            bpm = self._get_bpm(bandpassed, fps)
            self.bpm_buffer.append(bpm)
            self.bpm_buffer = self.bpm_buffer[-int(self.DURATION_ACCUMULATE_BPM * fps):]

            # Visualization
            self.visualize.append(bandpassed[-1])
            self.visualize = self.visualize[-self.VISUALIZE_SIZE:]

            return self.visualize
        else:
            return [0] * self.VISUALIZE_SIZE

    def _get_skin_mask(self, image, n_constant=0):
        try:
            low = np.array([0, 133, 77], np.uint8)
            high = np.array([235, 173, 127], np.uint8)

            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, low, high)
            mask[mask == 255] = 1

            return mask
        except Exception as e:
            return None

    def _calculate_ppg_value(self, image, mask):
        try:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            _, cr, cb = cv2.split(ycrcb)

            if not isinstance(mask, type(None)):
                n_pixels = image.shape[0] * image.shape[1]
            else:
                n_pixels = max(1, np.sum(mask))
                cr[mask == 0] = 0
                cb[mask == 0] = 0
            return (np.sum(cr) + np.sum(cb)) / n_pixels
        except Exception as e:
            return self.raw[-1] if len(self.raw) > 0 else 0.0

    def _detrend_signal(self, arr, wsize):
        try:
            if not isinstance(wsize, int):
                wsize = int(wsize)
            norm = np.convolve(np.ones(len(arr)), np.ones(wsize), mode='same')
            mean = np.convolve(arr, np.ones(wsize), mode='same') / norm
            return (arr - mean) / (mean + 1e-15)
        except ValueError:
            return arr

    def _filter_bandpass(self, arr, srate, band):
        try:
            nyq = 60 * srate / 2
            coef_vector = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
            return signal.filtfilt(*coef_vector, arr)
        except ValueError:
            return arr

    def _get_bpm(self, arr, fps):
        try:
            # Hamming 윈도우 적용
            windowed_arr = arr * np.hanning(len(arr))
            signal_len = len(windowed_arr)

            # 제로패딩 신호 생성
            pad_factor = max(1.0, (60 * fps) / signal_len)
            n_padded = int(len(windowed_arr) * pad_factor)

            # FFT 수행
            fft = np.fft.rfft(windowed_arr, n=n_padded)
            f = np.fft.rfftfreq(n_padded, d=1 / fps)

            # 주파수 스펙트럼
            frequency_spectrum = np.abs(fft)

            # confidence 계산
            fundamental_peak = np.argmax(frequency_spectrum)

            bpm = int(f[fundamental_peak] * 60)
            bpm = np.clip(bpm, self.BPM_BAND[0], self.BPM_BAND[1]).item()
            return bpm
        except (ValueError, FloatingPointError):
            return 0

    def get_bpm(self):
        if Timer.check_sec_ppg():
            if len(self.bpm_buffer) > 0:
                sorted_bpm = np.sort(self.bpm_buffer)
                bpm_len = len(sorted_bpm) // 3
                self.bpm = round(sorted_bpm[bpm_len: -bpm_len].mean())

                return self.bpm
        return self.bpm