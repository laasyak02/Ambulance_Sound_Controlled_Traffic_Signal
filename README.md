# Ambulance-Sound-Controlled Traffic Signal

A traffic-signal system that detects an **ambulance siren** from audio and holds the light green to let the ambulance through. Siren detection is done with traditional **audio + image processing**, and the signal is driven on an **Arduino** via `pyfirmata`.

<p align="center">
  <img src="media/led_states.jpg" width="720" alt="Arduino traffic-signal prototype showing the red, yellow and green states">
  <br><i>The Arduino traffic-signal prototype - red, yellow and green states.</i>
</p>

*Built during a research internship at CSIR-National Aerospace Laboratories (NAL), June-July 2023.*

---

## How it works

### 1. Siren detection (signal + image processing)

Ambulance sirens sweep up and down in pitch in a regular, repeating pattern, mostly between about 650 and 1100 Hz, while most horns stay below 600 Hz. Detection turns the audio into an image and looks for that pattern:

1. **Frequency check.** If there is no sound above 600 Hz, the audio is classified as horn sound only.
2. **Audio → spectrogram.** The input `.wav` is converted to a **mel spectrogram** (64 mel bands, 1024-point FFT, hop length 320) with `librosa`, then to a log (dB) scale and saved as an image.
3. **Image processing.** The spectrogram is thresholded into a binary image (OpenCV), the region containing the siren is cropped out, and it is **skeletonized** (`skimage`) to reduce the siren's energy bands to single-pixel curves. Contours are then extracted and very small ones removed.
4. **Periodicity check.** The contour image is scanned for the alternating peaks and troughs of a siren. If they repeat for a fixed number of cycles, an ambulance siren is detected.

Two siren types are handled separately: the slow **wail** (needs about 15 s of audio) and the fast **yelp** (about 5 s).

<p align="center"><img src="media/pipeline_wail.png" width="800" alt="Wail siren detection pipeline"></p>
<p align="center"><img src="media/pipeline_yelp.png" width="800" alt="Yelp siren detection pipeline"></p>

### 2. Traffic signal control (Arduino)

A three-LED traffic light is wired to an Arduino Uno running the StandardFirmata firmware and driven from Python with `pyfirmata`:

- **Normal operation** cycles the lights on a fixed schedule: red (5 s) → yellow (2 s) → green (5 s) → yellow → red.
- **On detection**, the signal switches to green (or stays green) and holds it while the siren is heard. Once only horn sound remains, the cycle resumes from where it was interrupted.

<p align="center"><img src="media/circuit.png" width="420" alt="Circuit diagram: Arduino Uno with red, yellow and green LEDs"></p>
<p align="center"><i>Red, yellow and green LEDs on digital pins 10, 9 and 6, each through a 220 Ω resistor.</i></p>

---

## Results

The system was tested on recordings of both siren types, on sirens mixed with multiple horn sounds in the background (as in heavy traffic), and on horn sounds alone. Detecting a siren takes **under 4 seconds** from the input audio.

**Siren vs. horn.** The wail detector finds the repeating peaks and troughs and returns `True`; for horn sound alone, nothing is found above 600 Hz and it returns `False`.

<p align="center">
  <img src="media/wail_detected.png" width="640" alt="Wail detector output: peak and trough coordinates, True">
  <br>
  <img src="media/horn_rejected.png" width="640" alt="Horn-only input: False">
</p>

**Full system.** While the signal is red, a wail siren interrupts the cycle and switches it to green. When only horn sound is heard again, the cycle resumes from where it was interrupted (back to red).

<table align="center">
  <tr>
    <td align="center"><img src="media/combined_wail_to_green.png" width="400" alt="Red, siren detected, switches to green"><br><i>Siren detected → green</i></td>
    <td align="center"><img src="media/combined_horn_resume.png" width="400" alt="Horn only, cycle resumes at red"><br><i>Siren gone → cycle resumes</i></td>
  </tr>
</table>

---

## Repository contents

| File | Description |
|------|-------------|
| `Arduino_Ambulance_Sound_Combined.py` | Full system - siren detection (wail and yelp) combined with Arduino LED control. |
| `Arduino_Traffic_Signal.py` | Standalone Arduino traffic-light cycle on a fixed schedule. |
| `Ambulance_Longer_Cycle.py` | Detection for the longer-cycle **wail** siren. |
| `Ambulance_Shorter_Cycle.py` | Detection for the shorter-cycle **yelp** siren. |

---

## Hardware & dependencies

**Hardware**
- Arduino Uno R3
- 3 LEDs (red, yellow, green) with 220 Ω resistors, breadboard

**Python libraries**
- `pyfirmata` - Arduino serial control
- `librosa` - audio loading and mel spectrogram
- `opencv-python` (`cv2`) - image thresholding, cropping, contours
- `scikit-image` - skeletonization
- `matplotlib`, `numpy`
- `keyboard` - select which test audio file to run

---

## Running it

1. Upload the **StandardFirmata** sketch to the Arduino and connect it (update the serial port in the script, e.g. `COM3`).
2. Wire the red/yellow/green LEDs to digital pins 10 / 9 / 6.
3. Point the script at your test siren `.wav` file(s).
4. Run:
   ```sh
   python Arduino_Ambulance_Sound_Combined.py
   ```

---
