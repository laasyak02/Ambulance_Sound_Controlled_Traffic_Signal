# Ambulance-Sound-Controlled Traffic Signal

Built a traffic-signal system that detects an **ambulance siren** from audio and holds the light green to let the ambulance through. Siren detection is done with traditional **audio + image processing**, and the signal is driven on an **Arduino** via `pyfirmata`.

This project was carried out as part of an internship at CSIR-National Aerospace Laboratories (NAL), Bengaluru from June - July 2023. 
<p align="center">
  <img src="media/led_states.jpg" width="720" alt="Arduino traffic-signal prototype showing the red, yellow and green states">
  <br><i>The Arduino traffic-signal prototype - red, yellow and green states.</i>
</p>

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
<p align="center"><i><b>Wail siren</b> (slow sweep, about 5 s per cycle). (1) The log-mel spectrogram of the recording - the siren shows up as the repeating arcs in the upper band. (2) The spectrogram is thresholded into a binary image and cropped to the band containing the siren. (3) Skeletonization thins the band to a single-pixel curve. (4) Contours are extracted and small ones removed, leaving the rise-and-fall pattern whose peaks and troughs the detector counts.</i></p>

<p align="center"><img src="media/pipeline_yelp.png" width="800" alt="Yelp siren detection pipeline"></p>
<p align="center"><i><b>Yelp siren</b> (fast sweep, many short cycles in a 5 s clip). The same four steps: (1) log-mel spectrogram, where the yelp appears as a tight sawtooth; (2) thresholded and cropped; (3) skeletonized; (4) contours, whose closely spaced peaks and troughs are counted to confirm the siren.</i></p>

### 2. Traffic signal control (Arduino)

A three-LED traffic light is wired to an Arduino Uno running the StandardFirmata firmware and driven from Python with `pyfirmata`:

- **Normal operation** cycles the lights on a fixed schedule: red (5 s) → yellow (2 s) → green (5 s) → yellow → red.
- **On detection**, the signal switches to green (or stays green) and holds it while the siren is heard. Once only horn sound remains, the cycle resumes from where it was interrupted.

<p align="center"><img src="media/circuit.png" width="420" alt="Circuit diagram: Arduino Uno with red, yellow and green LEDs"></p>
<p align="center"><i>Red, yellow and green LEDs on digital pins 10, 9 and 6, each through a 220 Ω resistor.</i></p>

---

## Results

The system was tested on recordings of both siren types, on sirens mixed with multiple horn sounds in the background (as in heavy traffic), and on horn sounds alone. Detecting a siren takes **under 4 seconds** from the input audio.

**Siren vs. horn.** For a siren, the wail detector finds the repeating peaks and troughs and returns `True`. For a recording of horns alone, the mel spectrogram has no energy above 600 Hz, so the same detector script returns `False`.

<p align="center">
  <img src="media/wail_detected.png" width="640" alt="Wail detector output on a siren recording: peak and trough coordinates, True">
  <br><i>Wail detector on a siren recording: the peaks and troughs it found, then <code>True</code>.</i>
</p>

<table align="center">
  <tr>
    <td align="center"><img src="media/horn_spectrogram.png" width="380" alt="Mel spectrogram of a horn-only recording"></td>
    <td align="center" valign="middle"><img src="media/horn_rejected.png" width="400" alt="Detector output on the horn recording: False"></td>
  </tr>
</table>
<p align="center"><i>The same detector run on a horn-only recording: everything stays below 600 Hz, so it returns <code>False</code>.</i></p>

**Full system.** The signal cycles normally until a siren is heard, then switches to green and holds it. When only horn sound remains, the cycle resumes from where it was interrupted. The logs below are the system's terminal output, one line per signal update (the report shows the signal state as printed text).

<p align="center"><img src="media/full_siren_to_green.png" width="760" alt="Signal is red, a wail siren is detected, signal switches to green"></p>
<p align="center"><i>While the signal is red, a wail siren is detected and the signal switches to green.</i></p>

<p align="center"><img src="media/full_horn_resume.png" width="760" alt="Siren gone, only horn sound, cycle resumes at red"></p>
<p align="center"><i>Once only horn sound is heard, the cycle resumes where it was interrupted - back to red.</i></p>

Detection currently runs on pre-recorded audio clips. The next steps identified in the report are real-time detection from live microphones, more siren types and multiple signals, and running it on a standalone board such as a Raspberry Pi or Jetson Nano.

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

## References

- R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed.
- [Audio feature extraction notebook](https://github.com/PrabhjotKaurGosal/Helpful-scripts-for-MachineLearning/blob/main/AudioFeatureExtraction_vX.ipynb) - P. K. Gosal, *Helpful scripts for Machine Learning*
- [Retrieving pixel coordinates of a specific colour](https://answers.opencv.org/question/98980/how-to-retrieve-all-coordinates-of-pixels-of-specific-colour-in-an-image/) - OpenCV Q&A
- [Skeletonize](https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html) - scikit-image documentation
