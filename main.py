# pyinstaller -D --clean --noconfirm --log-level WARN main.spec

import cv2
import numpy as np

from detector.landmark import Detector
from detector.rppg import rPPG
from detector.rrsp import rRSP
from timer import Timer
from utils import draw_signal


def draw_result(frame, signal, name, bpm, rect, rect_color=(0, 0, 255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, rect[0], rect[1], rect_color, 3)

    bpm_w = int(w / 3)

    signal_frame = draw_signal(signal, width=w-bpm_w)
    bpm_frame = np.zeros((150, bpm_w, 3), np.uint8)

    cv2.putText(bpm_frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bpm_frame, "%03d" % bpm, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
    frame = np.vstack((frame, np.hstack((signal_frame, bpm_frame))))

    return frame


def main():
    # Initialize modules
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = Detector()
    rppg = rPPG()
    rrsp = rRSP('model/onnx_model.onnx')

    # Set flags
    use_ppg = True
    use_rsp = True
    use_detect = True

    # Set frame
    frame_name = 'Remote PPG & RSP'
    cv2.namedWindow(frame_name)

    while True:
        # Set time
        Timer.set_time_stamp()

        # Get frame
        ret, frame = cap.read()
        if not ret:
            break
        visualize_frame = frame.copy()

        # Calculate landmark
        if use_detect:
            ret = detector.process(frame)
            if not ret:
                cv2.putText(visualize_frame, "No face detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(frame_name, visualize_frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                continue

        if ret:
            if use_ppg:
                # Get landmark
                face_sp, face_ep = detector.get_face_rect()

                # PPG processing
                rppg_signal = rppg.process(frame, face_sp, face_ep)
                rppg_bpm = rppg.get_bpm()

                # Visualize
                visualize_frame = draw_result(visualize_frame, rppg_signal, "rPPG", rppg_bpm, (face_sp, face_ep), (0, 255, 255))

            if use_rsp:
                # Get torso landmark
                torso_sp, torso_ep = detector.get_torso_rect()

                # RSP processing
                rrsp_signal = rrsp.process(frame, torso_sp, torso_ep)
                rrsp_bpm = rrsp.get_bpm()

                # Visualize
                visualize_frame = draw_result(visualize_frame, rrsp_signal, "rRSP", rrsp_bpm, (torso_sp, torso_ep), (0, 0, 255))

            # FPS
            cv2.putText(visualize_frame, "%02d fps" % round(Timer.get_fps()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Close event
        try:
            if cv2.getWindowProperty(frame_name, 0) < 0:
                break
        except:
            break

        cv2.imshow(frame_name, visualize_frame)
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == ord('1'):
            use_ppg, use_rsp = True, True
        elif key == ord('2'):
            use_ppg, use_rsp = True, False
            rrsp.reset()
        elif key == ord('3'):
            use_ppg, use_rsp = False, True
            rppg.reset()
        elif key == ord(' '):
            use_detect = not use_detect

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()

