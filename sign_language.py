"""
Sign Language Translator
Uses MacBook camera + MediaPipe hand tracking to recognize ASL (American Sign Language)
alphabet letters and display translations in real-time.

Controls:
  - Show hand signs → Detected letter appears on screen
  - Hold sign steady for 1s → Letter added to word
  - Open palm (5 fingers) → Space (next word)
  - Fist (0 fingers) → Backspace
  - Press 'c' → Clear all text
  - Press 'q' → Quit

Requirements:
  - macOS Camera permission

Usage:
  python3 sign_language.py
  Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_args():
    parser = argparse.ArgumentParser(description="Sign Language Translator")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--hold-time", type=float, default=0.8,
                        help="Seconds to hold a sign before it registers (default: 0.8)")
    parser.add_argument("--confidence", type=float, default=0.8,
                        help="Required consistency ratio during hold (0-1, default: 0.8)")
    parser.add_argument("--cam-width", type=int, default=640,
                        help="Camera capture width (default: 640)")
    parser.add_argument("--cam-height", type=int, default=480,
                        help="Camera capture height (default: 480)")
    return parser.parse_args()


def distance(p1, p2):
    """Euclidean distance between two landmark points."""
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def is_finger_extended(landmarks, tip_id, pip_id, mcp_id, is_thumb=False):
    """Check if a finger is extended."""
    if is_thumb:
        # Thumb: compare x-distance from palm base
        return abs(landmarks[tip_id].x - landmarks[0].x) > abs(landmarks[pip_id].x - landmarks[0].x)
    else:
        return landmarks[tip_id].y < landmarks[pip_id].y


def get_finger_states(landmarks):
    """Get extended state of each finger: [thumb, index, middle, ring, pinky]."""
    thumb = is_finger_extended(landmarks, 4, 3, 2, is_thumb=True)
    index = is_finger_extended(landmarks, 8, 6, 5)
    middle = is_finger_extended(landmarks, 12, 10, 9)
    ring = is_finger_extended(landmarks, 16, 14, 13)
    pinky = is_finger_extended(landmarks, 20, 18, 17)
    return [thumb, index, middle, ring, pinky]


def fingers_touching(landmarks, id1, id2, threshold=0.05):
    """Check if two fingertips are close together."""
    return distance(landmarks[id1], landmarks[id2]) < threshold


def get_hand_angle(landmarks):
    """Get the angle of the hand (wrist to middle finger MCP)."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    angle = np.arctan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)
    return np.degrees(angle)


def recognize_asl_letter(landmarks):
    """
    Recognize ASL alphabet letters from hand landmarks.
    Returns the detected letter or '?' if unrecognized.
    """
    fingers = get_finger_states(landmarks)
    thumb, index, middle, ring, pinky = fingers
    num_extended = sum(fingers)

    # Distances between fingertips
    thumb_index_dist = distance(landmarks[4], landmarks[8])
    thumb_middle_dist = distance(landmarks[4], landmarks[12])
    thumb_ring_dist = distance(landmarks[4], landmarks[16])
    thumb_pinky_dist = distance(landmarks[4], landmarks[20])
    index_middle_dist = distance(landmarks[8], landmarks[12])

    # Finger tip to palm distances
    palm_center = landmarks[9]  # middle finger MCP as palm reference

    # A: Fist with thumb to the side (thumb out, others curled)
    if thumb and not index and not middle and not ring and not pinky:
        # Check thumb is to the side, not tucked
        if landmarks[4].y < landmarks[3].y:
            return 'A'

    # B: All fingers up, thumb tucked across palm
    if not thumb and index and middle and ring and pinky:
        if index_middle_dist < 0.05:
            return 'B'

    # C: Curved hand (fingers together, curved)
    if num_extended >= 3:
        if thumb and index and middle:
            if thumb_index_dist > 0.05 and thumb_index_dist < 0.12:
                if not ring and not pinky:
                    return 'C'

    # D: Index up, others form circle with thumb
    if index and not middle and not ring and not pinky:
        if thumb_middle_dist < 0.05:
            return 'D'

    # E: All fingers curled, thumb tucked
    if num_extended == 0:
        return 'E'

    # F: Index and thumb form circle, other fingers up
    if not index and middle and ring and pinky:
        if thumb_index_dist < 0.04:
            return 'F'

    # G: Index pointing sideways, thumb parallel
    if index and thumb and not middle and not ring and not pinky:
        hand_angle = get_hand_angle(landmarks)
        if -45 < hand_angle < 45:
            return 'G'

    # H: Index and middle pointing sideways
    if index and middle and not ring and not pinky:
        hand_angle = get_hand_angle(landmarks)
        if -45 < hand_angle < 45:
            return 'H'

    # I: Only pinky up
    if not thumb and not index and not middle and not ring and pinky:
        return 'I'

    # K: Index and middle up, spread apart, thumb between
    if index and middle and not ring and not pinky:
        if index_middle_dist > 0.06:
            return 'K'

    # L: L-shape (index up, thumb out to side)
    if thumb and index and not middle and not ring and not pinky:
        if thumb_index_dist > 0.1:
            return 'L'

    # M: Thumb under three fingers (fist, thumb between ring and pinky)
    if not thumb and not index and not middle and not ring and not pinky:
        if landmarks[4].y > landmarks[8].y:
            return 'M'

    # O: All fingers form circle with thumb
    if num_extended >= 2:
        if thumb_index_dist < 0.04 and thumb_middle_dist < 0.06:
            return 'O'

    # R: Index and middle crossed
    if index and middle and not ring and not pinky:
        if index_middle_dist < 0.03:
            return 'R'

    # U: Index and middle up together
    if index and middle and not ring and not pinky and not thumb:
        if index_middle_dist < 0.05:
            return 'U'

    # V: Peace sign (index and middle spread)
    if index and middle and not ring and not pinky:
        if index_middle_dist > 0.05:
            return 'V'

    # W: Index, middle, ring up and spread
    if index and middle and ring and not pinky and not thumb:
        return 'W'

    # X: Index finger hooked/bent
    if not thumb and not middle and not ring and not pinky:
        if landmarks[8].y > landmarks[6].y and landmarks[8].y < landmarks[5].y:
            return 'X'

    # Y: Thumb and pinky out, others curled
    if thumb and not index and not middle and not ring and pinky:
        return 'Y'

    # 5/Open palm → treat as space
    if num_extended == 5:
        return ' '

    return '?'


def main():
    args = get_args()

    # Camera setup
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: Cannot access camera.")
        print("Go to System Settings → Privacy & Security → Camera and allow access.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    print(f"Camera: {args.cam_width}x{args.cam_height}")
    print(f"Hold time: {args.hold_time}s | Confidence: {args.confidence:.0%}")
    print()
    print("ASL Alphabet Signs Supported:")
    print("  A, B, C, D, E, F, G, H, I, K, L, M, O, R, U, V, W, X, Y")
    print()
    print("Controls:")
    print("  Show ASL sign     → Letter detected")
    print("  Hold sign steady  → Letter added to text")
    print("  Open palm (5)     → Space")
    print("  Fist              → Detected as 'E' (or backspace if held)")
    print("  Press 'c'         → Clear text")
    print("  Press 'q'         → Quit")
    print()

    # State
    current_letter = '?'
    last_letter = '?'
    letter_start_time = 0
    registered = False
    text = ""
    sentence_history = []
    detection_history = []  # track detections during hold period
    no_hand_frames = 0  # count frames with no hand to require re-entry

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            detected = '?'

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark

                # Require hand to leave and re-enter after registering
                if no_hand_frames < 5 and registered:
                    # Hand stayed in frame after last registration — wait for it to leave
                    current_letter = '?'
                    no_hand_frames = 0
                else:
                    no_hand_frames = 0

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Recognize the sign
                    detected = recognize_asl_letter(landmarks)

                    # Track detections for confidence scoring
                    if detected != '?':
                        detection_history.append(detected)
                    else:
                        detection_history.append(None)

                    # Only keep last 2 seconds of history
                    max_history = 60  # ~2s at 30fps
                    if len(detection_history) > max_history:
                        detection_history = detection_history[-max_history:]

                    # Track how long the same letter is held
                    if detected == last_letter and detected != '?':
                        elapsed = time.time() - letter_start_time
                        if elapsed >= args.hold_time and not registered:
                            # Check confidence: what % of recent frames had this letter
                            recent = detection_history[-int(args.hold_time * 30):]
                            if recent:
                                match_count = sum(1 for d in recent if d == detected)
                                confidence = match_count / len(recent)
                            else:
                                confidence = 0

                            if confidence >= args.confidence:
                                # Register the letter
                                if detected == ' ':
                                    text += ' '
                                else:
                                    text += detected
                                registered = True
                                detection_history.clear()
                                print(f"  → Added: '{detected}' (conf: {confidence:.0%})  |  Text: {text}")
                    else:
                        # New letter detected
                        last_letter = detected
                        letter_start_time = time.time()
                        registered = False

                    current_letter = detected

                    # Draw progress bar for hold time
                    if detected != '?' and not registered:
                        elapsed = time.time() - letter_start_time
                        progress = min(elapsed / args.hold_time, 1.0)
                        bar_width = 200
                        bar_x = 10
                        bar_y = 140
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                                      (50, 50, 50), -1)
                        cv2.rectangle(frame, (bar_x, bar_y),
                                      (bar_x + int(bar_width * progress), bar_y + 20),
                                      (0, 255, 0) if progress < 1.0 else (0, 200, 255), -1)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20),
                                      (255, 255, 255), 1)

                        # Show confidence
                        recent = detection_history[-int(args.hold_time * 30):]
                        if recent:
                            match_count = sum(1 for d in recent if d == detected)
                            conf = match_count / len(recent)
                            cv2.putText(frame, f"Conf: {conf:.0%}", (220, 157),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            else:
                no_hand_frames += 1
                if no_hand_frames > 10:
                    # Hand left the frame — reset for next sign
                    current_letter = '?'
                    last_letter = '?'
                    registered = False
                    detection_history.clear()

            # Display current detection
            h, w = frame.shape[:2]

            # Big letter display
            letter_color = (0, 255, 0) if current_letter != '?' else (100, 100, 100)
            display_char = current_letter if current_letter != ' ' else 'SPC'
            cv2.putText(frame, display_char, (w - 100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, letter_color, 4)

            # Status
            status = "REGISTERED" if registered else "Hold steady..." if current_letter != '?' else "Show a sign"
            status_color = (0, 200, 255) if registered else (255, 255, 255)
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Translated text display (bottom of frame)
            cv2.rectangle(frame, (0, h - 60), (w, h), (40, 40, 40), -1)
            cv2.putText(frame, "Text:", (10, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Show last 30 characters
            display_text = text[-30:] if len(text) > 30 else text
            cv2.putText(frame, display_text + "_", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Letter guide (top)
            cv2.putText(frame, "ASL: A B C D E F G H I K L O R U V W X Y", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow("Sign Language Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                text = ""
                print("  → Text cleared")
            elif key == ord("b") or key == 8:  # backspace
                text = text[:-1]
                print(f"  → Backspace  |  Text: {text}")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"Final text: {text}")
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
