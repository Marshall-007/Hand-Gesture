"""
Head-Controlled Mouse — Accessibility Mouse Control
Uses MacBook camera + MediaPipe Face Mesh head tracking to move the cursor.
Designed for people with limited mobility.

Controls:
    - Head movement (nose position): Move cursor
    - Close one eye: Left click
    - Both eyes blink (long): Pause/unpause tracking

Performance optimizations:
    - Reduced processing resolution
    - Efficient landmark subset (nose + face bounds)
    - Optimized smoothing with deque

Requirements:
    - macOS Camera permission
    - macOS Accessibility permission (System Settings → Privacy & Security → Accessibility)

Usage:
    python3 eye_mouse.py
    Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import argparse
from collections import deque

# Performance: disable PyAutoGUI delays
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# Key landmark indices for iris and eye
# Left eye indices
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33

# Right eye indices
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Head tracking landmarks
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


def get_args():
    parser = argparse.ArgumentParser(description="Eye-Controlled Mouse (Accessibility)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--sensitivity-x", type=float, default=1.5,
                        help="Horizontal sensitivity (default: 1.5)")
    parser.add_argument("--sensitivity-y", type=float, default=1.8,
                        help="Vertical sensitivity (default: 1.8)")
    parser.add_argument("--smoothing", type=int, default=8,
                        help="Smoothing window size — higher = smoother but more latency (default: 8)")
    parser.add_argument("--blink-threshold", type=float, default=0.21,
                        help="Eye aspect ratio threshold for blink detection (default: 0.21)")
    parser.add_argument("--click-cooldown", type=float, default=0.5,
                        help="Seconds between clicks (default: 0.5)")
    parser.add_argument("--cam-width", type=int, default=640,
                        help="Camera capture width (default: 640)")
    parser.add_argument("--cam-height", type=int, default=480,
                        help="Camera capture height (default: 480)")
    return parser.parse_args()


def eye_aspect_ratio(landmarks, top_idx, bottom_idx, inner_idx, outer_idx):
    """
    Calculate Eye Aspect Ratio (EAR) to detect blinks.
    EAR drops significantly when eye is closed.
    """
    top = np.array([landmarks[top_idx].x, landmarks[top_idx].y])
    bottom = np.array([landmarks[bottom_idx].x, landmarks[bottom_idx].y])
    inner = np.array([landmarks[inner_idx].x, landmarks[inner_idx].y])
    outer = np.array([landmarks[outer_idx].x, landmarks[outer_idx].y])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(inner - outer)

    if horizontal == 0:
        return 0.3  # default open
    return vertical / horizontal


def get_head_relative_position(landmarks, left_idx, right_idx, top_idx, bottom_idx, nose_idx):
    """
    Get nose position relative to face bounds (0-1 range).
    This normalizes head movement for cursor control.
    """
    nose = landmarks[nose_idx]
    left = landmarks[left_idx]
    right = landmarks[right_idx]
    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]

    left_x = min(left.x, right.x)
    right_x = max(left.x, right.x)
    top_y = min(top.y, bottom.y)
    bottom_y = max(top.y, bottom.y)

    face_width = right_x - left_x
    face_height = bottom_y - top_y

    if abs(face_width) < 1e-6:
        rel_x = 0.5
    else:
        rel_x = (nose.x - left_x) / face_width

    if abs(face_height) < 1e-6:
        rel_y = 0.5
    else:
        rel_y = (nose.y - top_y) / face_height

    return np.clip(rel_x, 0.0, 1.0), np.clip(rel_y, 0.0, 1.0)


class SmoothBuffer:
    """Moving average buffer for smooth cursor movement."""

    def __init__(self, size=5):
        self.buffer_x = deque(maxlen=size)
        self.buffer_y = deque(maxlen=size)

    def update(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        return np.mean(self.buffer_x), np.mean(self.buffer_y)


def main():
    args = get_args()

    screen_w, screen_h = pyautogui.size()

    # Camera setup
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: Cannot access camera.")
        print("Go to System Settings → Privacy & Security → Camera and allow access.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Smoothing buffers
    gaze_smoother = SmoothBuffer(args.smoothing)

    # State
    last_click_time = 0
    tracking_active = True
    both_blink_start = 0
    prev_x, prev_y = screen_w // 2, screen_h // 2

    # Calibration center (neutral head position = center of screen)
    center_x, center_y = 0.5, 0.5
    center_set = False

    print(f"Screen: {screen_w}x{screen_h}")
    print(f"Camera: {args.cam_width}x{args.cam_height}")
    print(f"Sensitivity: X={args.sensitivity_x}, Y={args.sensitivity_y}")
    print(f"Smoothing window: {args.smoothing}")
    print(f"Blink threshold: {args.blink_threshold}")
    print()
    print("Controls:")
    print("  Move head            → Move cursor")
    print("  Close one eye        → Left click")
    print("  Both eyes blink 1s   → Pause/Resume tracking")
    print()
    print("Press 'q' in the camera window to quit.")
    print("Press 'c' to recalibrate center (look at screen center first).")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB (MediaPipe expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Optimize: mark as not writeable for performance
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            status = "No face"
            left_ear = 0.3
            right_ear = 0.3

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark

                # Calculate Eye Aspect Ratios for blink detection
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                                            LEFT_EYE_INNER, LEFT_EYE_OUTER)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                            RIGHT_EYE_INNER, RIGHT_EYE_OUTER)

                left_blink = left_ear < args.blink_threshold
                right_blink = right_ear < args.blink_threshold

                # Both eyes blink for 1 second → toggle tracking
                if left_blink and right_blink:
                    if both_blink_start == 0:
                        both_blink_start = time.time()
                    elif time.time() - both_blink_start > 1.0:
                        tracking_active = not tracking_active
                        both_blink_start = 0
                        time.sleep(0.5)  # Prevent rapid toggling
                        status = "PAUSED" if not tracking_active else "RESUMED"
                        continue
                else:
                    # Single eye blinks for clicking
                    current_time = time.time()
                    if both_blink_start > 0 and (time.time() - both_blink_start) < 0.5:
                        # Was a brief both-blink, treat as regular blink
                        pass
                    both_blink_start = 0

                    if current_time - last_click_time > args.click_cooldown:
                        if left_blink != right_blink:
                            pyautogui.click()
                            last_click_time = current_time
                            status = "CLICK"

                # Head tracking (only when tracking is active and eyes are open)
                if tracking_active and not (left_blink and right_blink):
                    head_x, head_y = get_head_relative_position(
                        landmarks, LEFT_CHEEK, RIGHT_CHEEK,
                        FOREHEAD, CHIN, NOSE_TIP
                    )

                    if not center_set:
                        center_x, center_y = head_x, head_y
                        center_set = True

                    # Map gaze to screen coordinates
                    # Gaze is ~0.3-0.7 range, center at ~0.5
                    offset_x = (head_x - center_x) * args.sensitivity_x
                    offset_y = (head_y - center_y) * args.sensitivity_y

                    target_x = screen_w / 2 + offset_x * screen_w
                    target_y = screen_h / 2 + offset_y * screen_h

                    # Smooth
                    smooth_x, smooth_y = gaze_smoother.update(target_x, target_y)

                    # Clamp with buffer
                    smooth_x = int(np.clip(smooth_x, 5, screen_w - 5))
                    smooth_y = int(np.clip(smooth_y, 5, screen_h - 5))

                    try:
                        pyautogui.moveTo(smooth_x, smooth_y)
                    except Exception:
                        pass
                    prev_x, prev_y = smooth_x, smooth_y

                    status = "Tracking" if tracking_active else "PAUSED"
                elif not tracking_active:
                    status = "PAUSED"

                # Draw nose point and face bounds for feedback
                h, w = frame.shape[:2]
                nose_px = int(landmarks[NOSE_TIP].x * w)
                nose_py = int(landmarks[NOSE_TIP].y * h)
                cv2.circle(frame, (nose_px, nose_py), 3, (0, 255, 0), -1)

                left_px = int(landmarks[LEFT_CHEEK].x * w)
                right_px = int(landmarks[RIGHT_CHEEK].x * w)
                top_py = int(landmarks[FOREHEAD].y * h)
                bottom_py = int(landmarks[CHIN].y * h)

                x1, x2 = sorted([left_px, right_px])
                y1, y2 = sorted([top_py, bottom_py])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # Display status
            color = (0, 255, 0) if tracking_active else (0, 0, 255)
            cv2.putText(frame, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Cursor: ({prev_x}, {prev_y})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"L-EAR: {left_ear:.2f}  R-EAR: {right_ear:.2f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if not tracking_active:
                cv2.putText(frame, "PAUSED - Blink both eyes 1s to resume", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Eye Mouse Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Recalibrate: set current head position as center
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    hx, hy = get_head_relative_position(
                        landmarks, LEFT_CHEEK, RIGHT_CHEEK,
                        FOREHEAD, CHIN, NOSE_TIP
                    )
                    center_x, center_y = hx, hy
                    center_set = True
                    print(f"Recalibrated center: ({center_x:.3f}, {center_y:.3f})")

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
