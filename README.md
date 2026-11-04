# CS6524: Computer Vision — Lab 1: Silhouettes and Fourier Epicycles

**Immersive Software Engineering (ISE) — University of Limerick**  
**Instructor:** Juan Albarracín | **Module Lead:** James Patten  
**Deadline:** Sunday, November 8th at Midnight (Autumn Semester Week 2)

---

## 🎯 Objective

Understand the use of linear filters on images given a specific objective, and learn the mathematical nature of the Fourier transform.

---

## 📌 Context

An **epicycle** is a geometric model from ancient and medieval astronomy in which a planet moves along a small circle (the epicycle) whose center travels along a larger circle (the deferent) centered on or near the Earth. Used primarily to describe planetary orbits prior to Kepler's laws of planetary motion, epicycles nowadays serve as an intuitive visualization of the Fourier decomposition of complex 2D signals.

![Fourier Epicycles Concept Sketch](assets/fourier_epicycles_concept.png)
*Figure 1: Fourier Epicycles Concept Sketch showing fundamental frequency ($n=1$) and higher harmonics ($n=3$) tracing a complex path.*

### How Epicycles Work

In this model, stacked vectors rotate independently around their respective origins:

* **Radius:** Represents the amplitude ($\vert{}C_k\vert{}$) of a harmonic.
* **Rotational Speed & Direction:** Represents its frequency ($\omega_k$).
* **Initial Vector Orientation:** Represents its phase ($\theta_k$).
* **Tip-to-Tail Stacking:** Represents vector addition, i.e., the summation of complex harmonics.

According to Fourier theory, any periodic signal can be represented as an infinite sum of harmonics. By extending this concept to a continuous 2D perimeter $z(t) = x(t) + iy(t)$, a specific configuration of rotating epicycles can reproduce any arbitrary closed silhouette in 2D space.

In this laboratory, you will implement an algorithm that draws an object's silhouette using rotating epicycles. Your input will **not** be a pre-extracted set of boundary coordinates, but a raw natural image. You must first transform this image by applying sequential linear spatial filters (e.g., Gaussian smoothing and spatial derivative filters) to detect and extract the object's boundary before computing its Fourier descriptors.

---

## 🔄 Pre-processing Pipeline

> **Note:** The pipeline described below is just an example. The same sequence of operations may not work for every image you choose.

| Step | Visual Output | Description |
| :--- | :---: | :--- |
| **i) Raw Image** | ![Original Image](assets/step1_original.png) | Start with a standard color image. |
| **ii) Grayscale** | ![Grayscale Image](assets/step2_grayscale.png) | Transform the image into grayscale for convenience. |
| **iii) Blurring** | ![Blurred Image](assets/step3_blurred.png) | Apply Gaussian blur/smoothing to remove artifacts (e.g., top-left text). |
| **iv) Thresholding** | ![Thresholded Image](assets/step4_thresholded.png) | Apply a fixed threshold to segment the silhouette. |
| **v) Border Detection** | ![Border Detection Image](assets/step5_border_detection.png) | Apply derivative/border detection filters to extract object edges. |

> **Tip:** Don't worry if the silhouette is not a perfect closed shape—attaining absolute perfection is out of scope. Just ensure that the borders are thin and have as few gaps as possible.

---

## 📐 Silhouette Trajectory

Once you have identified the edge pixels corresponding to the silhouette, transform them into a 2D parametric curve $(x(t), y(t))$, where $x(t)$ is as close as possible to $x(t+1)$.

![Trajectory Sorting Sequence](assets/trajectory_sorting_sequence.png)
*Figure 2: Ordered progression of tracing the object's boundary pixel trajectory.*

You can use the following reference Python snippet to sort and resample your silhouette points:

```python
import numpy as np

# Unsorted pixel coordinates of the silhouette
points = np.column_stack((edge_x, edge_y))
ordered_pts = [points[0]]
unvisited = list(points[1:])

while unvisited:
    last_pt = ordered_pts[-1]
    # Calculate Euclidean distances to remaining unvisited edge pixels
    dists = np.sum((unvisited - last_pt) ** 2, axis=1)
    nearest_idx = np.argmin(dists)
    ordered_pts.append(unvisited.pop(nearest_idx))

ordered_pts = np.array(ordered_pts)

# Resampling to fixed number of uniform samples
num_samples = 10000
distances = np.sqrt(np.sum(np.diff(ordered_pts, axis=0) ** 2, axis=1))
cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
total_length = cumulative_dist[-1]

target_dists = np.linspace(0, total_length, num_samples, endpoint=False)
x_resampled = np.interp(target_dists, cumulative_dist, ordered_pts[:, 0])
y_resampled = np.interp(target_dists, cumulative_dist, ordered_pts[:, 1])
```

> **Note:** You may notice straight lines cutting across the object where the nearest-neighbor algorithm connects distant points. You are encouraged to refine this logic to achieve a smoother trajectory.

---

## 🌊 Fourier Epicycles

1. Represent the 2D trajectory as a 1D complex signal:

$$z[n] = x[n] + i \cdot y[n]$$

2. Compute the 1D Discrete Fourier Transform (DFT) $Z[u]$ using `numpy.fft` to extract amplitudes, phases, and frequencies of the harmonics.

3. Select the top $K$ harmonics and reconstruct the trajectory using the Inverse Fourier Transform:

$$z'[n] = \frac{1}{N} \sum_{u=0}^{K-1} F[u] \exp(2\pi i \, \omega[u] \, n)$$

where:
* $K$ is the number of selected harmonics.
* $F[u]$ contains the harmonic amplitudes/phases.
* $\omega[u]$ represents the corresponding frequencies.

![Harmonic Reconstruction Progression](assets/fourier_reconstruction_progression.png)
*Figure 3: Reconstruction of silhouette with increasing number of Fourier Harmonics ($K = 1, 3, 5, 20, 100, 1000$).*

> As $K$ increases, the reconstruction quality improves and closely matches the original silhouette.

---

## 📋 Reporting and Submission Rules

1. **Report:** Written in LaTeX following the **IEEE Conference Template** (maximum **3 pages**).
2. **Code:** Standalone script runnable directly from a command prompt.
   * Must show proof of working on **at least 5 different images**. Choose images containing a single high-contrast object. Plain shapes/textures are not allowed.
   * **Restricted Libraries:** You **cannot** use high-level image processing libraries like OpenCV or `scikit-image`. `PIL` is allowed only for reading/writing images.
   * **Allowed Libraries:** `NumPy`, `SciPy`, `PIL`, and `matplotlib`.
   * **Filtering Requirement:** Linear filtering must be performed using `scipy.ndimage.convolve` or `scipy.ndimage.generic_filter` (you must design filter kernels manually).
3. **IO Structure:** Your code should read images from an input folder and output results to an output folder.
4. **Configuration:** If individual images require unique processing options, accept parameters via a `.json` configuration file.

---

## 🎁 Bonus (+1.0 Grade Point)

Include an **MP4 animation** of the rendering process showing the rotating epicycles drawing the shape. Any animation library (e.g., `matplotlib.animation`, `ffmpeg`) is permitted for rendering the video.