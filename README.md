# 🎨 CvPainting – AI-Powered Air Drawing with Hand Tracking

Draw in the air using your finger! This computer vision application uses **MediaPipe** for hand tracking and **OpenCV** to create a real-time drawing canvas. Just point your index finger and paint – no mouse, no tablet, no touchscreen needed.

> ✨ **Try it yourself** – Wave your hand, select colors with two fingers, and watch your digital art come to life.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| ✋ **Hand Tracking** | Real-time finger detection using MediaPipe |
| 🖌️ **Air Drawing** | Draw by raising your index finger |
| 🎨 **10 Colors** | Blue, Green, Red, Yellow, Purple, Cyan, Magenta, Orange, Teal, Silver |
| 🧽 **Eraser Mode** | Switch to eraser with gray button or gesture |
| 📏 **Adjustable Thickness** | Press `+` / `-` to change brush/eraser size (5–50px) |
| 🖐️ **Gesture Controls** | Two fingers up = color selector, one finger up = draw |
| 🖼️ **Canvas Overlay** | Drawing appears semi-transparent over camera feed |
| 💾 **Persistent Canvas** | Keeps your drawing on screen until cleared |

---

## 🎮 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ **Index finger only** | Draw / Erase (follows finger movement) |
| ✌️ **Index + Middle fingers** | Select color / Eraser from top bar |
| ✊ **No fingers up** | Stop drawing (move without leaving marks) |

---

## 🎨 Color Selection

| Index | Color | RGB |
|-------|-------|-----|
| 0 | Blue | (255, 0, 0) |
| 1 | Green | (0, 255, 0) |
| 2 | Red | (0, 0, 255) |
| 3 | Yellow | (0, 255, 255) |
| 4 | Purple | (255, 0, 255) |
| 5 | Cyan | (0, 255, 255) |
| 6 | Magenta | (128, 0, 128) |
| 7 | Orange | (255, 165, 0) |
| 8 | Teal | (0, 128, 128) |
| 9 | Silver | (192, 192, 192) |
| Eraser | — | (0, 0, 0) |

---

## 📦 Installation

### Prerequisites

- Python 3.7+
- Webcam (built-in or external)
- OpenCV, MediaPipe, NumPy

### Quick Install

```bash
git clone https://github.com/IMApurbo/CvPainting.git
cd CvPainting
pip install -r requirements.txt
python Paint.py
```

### Required Dependencies

```txt
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.19.0
```

Or install manually:

```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 Usage

### Basic Operation

1. **Run the script**
   ```bash
   python Paint.py
   ```

2. **Position yourself** in front of the webcam

3. **Raise your index finger** (other fingers down) to start drawing

4. **Move your hand** – the tip of your index finger leaves a trail

5. **Raise two fingers** (index + middle) to open color selector

6. **Point to a color box** at the top of the screen to switch colors

7. **Press `+` / `-`** to adjust brush or eraser thickness

8. **Press `Esc`** to exit

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `+` | Increase brush/eraser thickness (max 50px) |
| `-` | Decrease brush/eraser thickness (min 5px) |
| `Esc` | Exit application |

---

## 🖥️ UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│ [B][G][R][Y][P][C][M][O][T][S][Eraser] ← Color selection bar │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                                                              │
│                    Camera Feed + Canvas                     │
│                    (50% transparency)                       │
│                                                              │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ Eraser Thickness: 20px                                      │
│ Drawing Thickness: 8px                                      │
│ Color: Blue                                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧠 How It Works

### Hand Tracking (MediaPipe)
- Detects 21 hand landmarks in real-time
- Identifies fingertip positions (INDEX_FINGER_TIP, MIDDLE_FINGER_TIP)
- Compares fingertip Y-coordinates with DIP joints to determine if a finger is raised

### Drawing Mechanism
```python
if index_finger_up and not middle_finger_up:
    cv2.line(canvas, prev_pos, current_pos, color, thickness)
```

### Color Selection
- Two-finger gesture activates selector mode
- System checks if fingertip coordinates overlap with color rectangle zones
- Updates `current_color` and `current_color_name`

### Canvas Overlay
```python
frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
```
- 50/50 blend of camera feed (live) and drawing canvas (persistent)

---

## 🔧 Configuration

Edit these variables in `Paint.py`:

```python
# Canvas resolution
canvas = np.zeros((480, 640, 3), dtype="uint8")

# Default thicknesses
eraser_thickness = 20   # Pixels (5-50 range)
drawing_thickness = 8   # Pixels (5-50 range)

# Color palette (BGR format)
colors = [(255,0,0), (0,255,0), ...]

# Camera index (0 = default, 1 = external)
cap = cv2.VideoCapture(1)  # Change to 0 for built-in webcam
```

---

## 🎯 Use Cases

| Application | Description |
|-------------|-------------|
| 🎨 **Virtual Art Class** | Teach drawing without physical supplies |
| 🧑‍🏫 **Presentation Tool** | Annotate slides by pointing at screen |
| 🧒 **Kids' Activity** | Mess-free digital finger painting |
| ♿ **Accessibility** | Draw without mouse or touchscreen |
| 🕹️ **Gesture Prototype** | Base for more advanced gesture interfaces |

---

## 🐛 Troubleshooting

### ❌ Camera won't open

```python
# Change camera index in Paint.py
cap = cv2.VideoCapture(0)   # Try 0, 1, 2
```

### ❌ Hand not detected

- Ensure good lighting
- Background should not be too busy
- Keep hand within camera frame
- Check that MediaPipe model loaded correctly

### ❌ Drawing lag

- Reduce canvas resolution (change `640x480` to `320x240`)
- Close other CPU-intensive applications
- Use external webcam with better FPS

### ❌ Thickness not changing

- Make sure you're pressing `+` / `-` on the keyboard
- Check that the terminal window has focus
- Eraser and brush have separate thickness values

---

## 🚀 Future Enhancements

Ideas for contribution:

- [ ] Clear canvas button (gesture or key)
- [ ] Undo / Redo functionality
- [ ] Save drawing as PNG
- [ ] Brush shapes (circle, square, spray)
- [ ] Multi-color gradient brush
- [ ] Fist gesture to activate eraser
- [ ] Zoom and pan on canvas
- [ ] Export drawing as SVG
- [ ] Support for left-handed mode
- [ ] FPS counter display

---

## 📁 File Structure

```
CvPainting/
├── Paint.py           # Main application
├── requirements.txt   # Dependencies
├── README.md          # This file
└── LICENSE            # MIT License
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

MIT License – see [LICENSE](LICENSE) file.

Copyright (c) 2025 **IMApurbo**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

---

## 📬 Contact

**Author:** IMApurbo  
**GitHub:** [https://github.com/IMApurbo](https://github.com/IMApurbo)  
**Project Link:** [https://github.com/IMApurbo/CvPainting](https://github.com/IMApurbo/CvPainting)

---

> *Draw like magic. Paint in the air. No cleanup required.* 🎨✨
```
