import numpy as np
import cv2


def draw_signal(signal, width=500, height=150, peaks=None, frame=None, scale=None, ret_scale=False):
    # Create empty frame
    if frame is None:
        frame = np.zeros((height, width, 3), np.uint8)
    else:
        height, width = frame.shape[:2]

    # Signal preprocessing
    try:
        np_signal = np.array(signal)
        if scale is not None:
            min_val, max_val = scale
        else:
            max_val, min_val = np_signal.max(), np_signal.min()
        diff_val = max_val - min_val
        np_signal = np_signal if diff_val == 0 else (np_signal - np_signal.min()) / diff_val
    except:
        if ret_scale:
            return frame, (min_val, max_val)
        else:
            return frame

    # Draw signal
    width_offset = width / np_signal.shape[0]
    for i in range(np_signal.shape[0] - 1):
        sx = i * width_offset
        sy = height - (np_signal[i] * height)
        ex = (i + 1) * width_offset
        ey = height - (np_signal[(i + 1)] * height)
        cv2.line(frame, (int(sx), int(sy)), (int(ex), int(ey)), (0, 255, 0), 3)

        if (peaks is not None) and (i in peaks):
            cv2.circle(frame, (int((sx+ex)/2), int((sy+ey)/2)), 5, (0, 0, 255), -1)
    if ret_scale:
        return frame, (min_val, max_val)
    else:
        return frame


def show_signal(name, signal, width=500, height=150, peaks=None):
    frame = draw_signal(signal, width, height, peaks=peaks)
    cv2.imshow(name, frame)
