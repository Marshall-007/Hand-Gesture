# Hand Gesture Mouse Control

Control your Mac's mouse cursor using hand gestures detected through your webcam.

## Gestures

| Gesture | Action |
|---------|--------|
| Index finger raised (others curled) | Move cursor |
| Thumb + Index finger pinch | Left click |
| Thumb + Middle finger pinch | Right click |
| Open palm (all fingers up) | Neutral / pause |

## Setup

### 1. Install Python 3 (if not already installed)

```bash
brew install python
```

### 2. Create virtual environment & install dependencies

```bash
cd "Hand Gesture "
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Grant macOS Permissions

You need **two** permissions:

1. **Camera**: macOS will prompt you on first run. Click "Allow".
2. **Accessibility**: Required for mouse control. Go to:
   - **System Settings → Privacy & Security → Accessibility**
   - Add and enable your Terminal app (Terminal.app, iTerm2, or VS Code)

Without Accessibility permission, the app can detect gestures but cannot move the mouse.

## Usage

```bash
source venv/bin/activate
python3 hand_mouse.py
```

Press `q` in the camera window to quit.

### Options

```
--cam 0              Camera index (default: 0)
--sensitivity 1.5    Cursor speed multiplier (default: 1.5)
--smoothing 0.3      Smoothing factor 0-1, lower = smoother (default: 0.3)
--click-cooldown 0.4 Seconds between clicks (default: 0.4)
--pinch-threshold 0.05  Distance threshold for pinch detection (default: 0.05)
```

### Examples

```bash
# Slower, smoother cursor
python3 hand_mouse.py --sensitivity 1.0 --smoothing 0.2

# Faster, more responsive
python3 hand_mouse.py --sensitivity 2.0 --smoothing 0.5

# More forgiving pinch detection
python3 hand_mouse.py --pinch-threshold 0.07
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Cannot access camera" | System Settings → Privacy & Security → Camera → Allow |
| Cursor doesn't move | System Settings → Privacy & Security → Accessibility → Enable your terminal |
| Too jittery | Lower `--smoothing` (e.g., 0.15) |
| Cursor too slow | Increase `--sensitivity` (e.g., 2.0) |
| Accidental clicks | Increase `--click-cooldown` or decrease `--pinch-threshold` |
| Emergency stop | Move mouse to top-left corner of screen (PyAutoGUI failsafe) |

## How It Works

1. OpenCV captures frames from your webcam
2. MediaPipe Hands detects 21 hand landmarks in real-time
3. Custom logic interprets landmark positions into gestures
4. PyAutoGUI moves the cursor / performs clicks
5. Exponential moving average smooths cursor movement to prevent jitter
