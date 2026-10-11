# Ambulance-Sound-Controlled Traffic Signal

A traffic-signal system that detects an **ambulance siren** from audio and holds the light green to let the ambulance through. Siren detection is done with traditional **audio + image processing**, and the signal is driven on an **Arduino** via `pyfirmata`.

---

## How it works

### 1. Siren detection (signal + image processing)

Detection is done by turning the audio into an image and looking for the siren's characteristic repeating pattern:

1. **Audio → spectrogram.** The input `.wav` is converted to a **mel spectrogram** (64 mel bands, 1024-point FFT, hop length 320) with `librosa`, then converted to decibels and saved as an image.
2. **Image processing.** The spectrogram is binarized (OpenCV), cropped to the region of interest, and **skeletonized** (`skimage`) to reduce the siren's energy bands to thin curves; contours are then extracted.
3. **Periodicity check.** The contour image is scanned column by column for the alternating peak/trough pattern of a siren, with constraints on spacing and pixel count. Two siren types are handled separately — **wail** (slow sweep) and **yelp** (fast sweep).

The detector returns whether an ambulance siren is present or not.

### 2. Traffic signal control (Arduino)

A three-LED traffic light (red / yellow / green) is wired to an Arduino running the Firmata firmware and driven from Python with `pyfirmata`:

- **Normal operation** cycles the lights on a fixed schedule — red → yellow → green → yellow.
- **On detection**, the green LED is held on while an ambulance siren is detected, giving the ambulance a clear path.

---

## Repository contents

| File | Description |
|------|-------------|
| `Arduino_Ambulance_Sound_Combined.py` | Full system — siren detection combined with Arduino LED control. |
| `Arduino_Traffic_Signal.py` | Standalone Arduino traffic-light cycle on a fixed schedule. |
| `Ambulance_Longer_Cycle.py` | Detection/signal variant with a longer signal cycle. |
| `Ambulance_Shorter_Cycle.py` | Detection/signal variant with a shorter signal cycle. |

---

## Hardware & dependencies

**Hardware**
- Arduino board 
- 3 LEDs — red, yellow, green 

**Python libraries**
- `pyfirmata` — Arduino serial control
- `librosa` — audio loading and mel spectrogram
- `opencv-python` (`cv2`) — image thresholding, cropping, contours
- `scikit-image` — skeletonization
- `matplotlib`, `numpy`
- `keyboard` — select which test audio file to run

---

## Running it

1. Upload the **StandardFirmata** sketch to the Arduino and connect it (update the serial port in the script — e.g. `COM3`).
2. Wire the red/yellow/green LEDs to digital pins 10 / 9 / 6.
3. Point the script at your test siren `.wav` file(s).
4. Run:
   ```sh
   python Arduino_Ambulance_Sound_Combined.py
   ```

---
