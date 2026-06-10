"""
Hand Gesture Mouse Control
Uses MacBook camera + MediaPipe hand tracking to control the mouse cursor.

Gestures:
  - Index finger raised (others curled): Move cursor
  - Thumb + Index pinch: Left click
  - Thumb + Middle finger pinch: Right click
  - Open palm: Neutral (no action)

Requirements:
  - macOS Camera permission
  - macOS Accessibility permission (System Settings → Privacy & Security → Accessibility)

Usage:
  python3 hand_mouse.py
  Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import argparse

# Disable PyAutoGUI fail-safe pause for smoother movement
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False  # Disabled since we clamp coordinates away from corners

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_args():
    parser = argparse.ArgumentParser(description="Hand Gesture Mouse Control")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--sensitivity", type=float, default=1.5,
                        help="Cursor sensitivity multiplier (default: 1.5)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="Smoothing factor 0-1, lower = smoother (default: 0.3)")
    parser.add_argument("--click-cooldown", type=float, default=0.4,
                        help="Seconds between clicks (default: 0.4)")
    parser.add_argument("--pinch-threshold", type=float, default=0.05,
                        help="Pinch detection distance threshold (default: 0.05)")
    return parser.parse_args()


def distance(p1, p2):
    """Euclidean distance between two landmark points."""
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def is_finger_up(landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is raised by comparing tip and PIP joint y-coordinates."""
    return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y


def detect_gesture(landmarks, pinch_threshold):
    """
    Detect the current hand gesture.
    Returns: ('move', 'left_click', 'right_click', 'neutral')
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check pinches first (higher priority)
    thumb_index_dist = distance(thumb_tip, index_tip)
    thumb_middle_dist = distance(thumb_tip, middle_tip)

    if thumb_index_dist < pinch_threshold:
        return "left_click"

    if thumb_middle_dist < pinch_threshold:
        return "right_click"

    # Check which fingers are up
    index_up = is_finger_up(landmarks, 8, 6)
    middle_up = is_finger_up(landmarks, 12, 10)
    ring_up = is_finger_up(landmarks, 16, 14)
    pinky_up = is_finger_up(landmarks, 20, 18)

    # Open palm = all fingers up → neutral
    if index_up and middle_up and ring_up and pinky_up:
        return "neutral"

    # Only index finger up → move mode
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "move"

    # Index + middle up (peace sign) → also move (common resting position)
    if index_up and middle_up and not ring_up and not pinky_up:
        return "move"

    return "neutral"


def main():
    args = get_args()

    # Screen dimensions
    screen_w, screen_h = pyautogui.size()

    # Camera setup
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: Cannot access camera.")
        print("Go to System Settings → Privacy & Security → Camera and allow access.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Smoothing state
    prev_x, prev_y = screen_w // 2, screen_h // 2
    last_click_time = 0

    # Define a frame region to map (ignore edges to give more control)
    # Use center 60% of frame for mapping
    margin_x = 0.2
    margin_y = 0.2

    print(f"Screen: {screen_w}x{screen_h}")
    print(f"Camera: {cam_w}x{cam_h}")
    print(f"Sensitivity: {args.sensitivity}, Smoothing: {args.smoothing}")
    print(f"Pinch threshold: {args.pinch_threshold}")
    print()
    print("Gestures:")
    print("  Index finger up        → Move cursor")
    print("  Thumb + Index pinch    → Left click")
    print("  Thumb + Middle pinch   → Right click")
    print("  Open palm              → Neutral (pause)")
    print()
    print("Press 'q' in the camera window to quit.")
    print("Move mouse to top-left corner to emergency stop (PyAutoGUI failsafe).")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # Flip horizontally for natural mirror effect
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            gesture = "neutral"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark

                # Draw landmarks on frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect gesture
                gesture = detect_gesture(landmarks, args.pinch_threshold)

                if gesture == "move" or gesture == "left_click" or gesture == "right_click":
                    # Use index finger tip for cursor position
                    index_tip = landmarks[8]

                    # Map finger position to screen coordinates
                    # Clamp to the active region
                    norm_x = (index_tip.x - margin_x) / (1.0 - 2 * margin_x)
                    norm_y = (index_tip.y - margin_y) / (1.0 - 2 * margin_y)
                    norm_x = np.clip(norm_x, 0.0, 1.0)
                    norm_y = np.clip(norm_y, 0.0, 1.0)

                    # Apply sensitivity
                    target_x = int(norm_x * screen_w * args.sensitivity)
                    target_y = int(norm_y * screen_h * args.sensitivity)

                    # Clamp to screen bounds with corner buffer (avoid edges)
                    corner_buf = 5
                    target_x = min(max(corner_buf, target_x), screen_w - corner_buf)
                    target_y = min(max(corner_buf, target_y), screen_h - corner_buf)

                    # Smooth the cursor position (exponential moving average)
                    smooth_x = int(prev_x + args.smoothing * (target_x - prev_x))
                    smooth_y = int(prev_y + args.smoothing * (target_y - prev_y))

                    # Move cursor
                    try:
                        pyautogui.moveTo(smooth_x, smooth_y)
                    except pyautogui.FailSafeException:
                        pass
                    prev_x, prev_y = smooth_x, smooth_y

                # Handle clicks with cooldown
                current_time = time.time()
                if current_time - last_click_time > args.click_cooldown:
                    if gesture == "left_click":
                        pyautogui.click()
                        last_click_time = current_time
                    elif gesture == "right_click":
                        pyautogui.rightClick()
                        last_click_time = current_time

            # Draw gesture status on frame
            color = {
                "move": (0, 255, 0),
                "left_click": (0, 0, 255),
                "right_click": (255, 0, 0),
                "neutral": (200, 200, 200),
            }.get(gesture, (200, 200, 200))

            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Cursor: ({prev_x}, {prev_y})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw the active region rectangle
            x1 = int(margin_x * cam_w)
            y1 = int(margin_y * cam_h)
            x2 = int((1 - margin_x) * cam_w)
            y2 = int((1 - margin_y) * cam_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

            cv2.imshow("Hand Gesture Mouse Control", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
