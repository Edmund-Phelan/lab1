from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_transform(
    original: np.ndarray,
    greyscale: np.ndarray,
    blurred: np.ndarray,
    thresholded: np.ndarray,
    borders: np.ndarray,
    output_path: str | Path,
) -> None:
    panels = [
        (original, 'Original image', None),
        (greyscale, 'Greyscale', 'gray'),
        (blurred, 'Gaussian blur', 'gray'),
        (thresholded, 'Threshold', 'gray'),
        (borders, 'Detected borders', 'gray'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (image, title, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')

    axes.flat[-1].axis('off')
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _curve_axes(ax, shape):
    # Image coordinates: y grows downwards, so flip to match the picture
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)
    ax.set_aspect('equal')
    ax.axis('off')


def save_trajectory(x, y, shape, output_path) -> None:
    """10 frames showing the resampled boundary being traced from t=0 to t=T."""
    N = len(x)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        n = int(np.ceil((i + 1) * N / 10))
        xs, ys = x[:n], y[:n]
        if i == 9:  # last frame closes the contour
            xs, ys = np.append(xs, x[0]), np.append(ys, y[0])
        ax.plot(xs, ys, 'b-', lw=1)
        ax.plot(xs[-1], ys[-1], 'ro', ms=4)
        ax.set_title(f't = {n}/{N}' if i < 9 else 't = T (closed)')
        _curve_axes(ax, shape)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_final(recons, percents, shape, output_path) -> None:
    """20 subplots: reconstruction with 5%..100% of the dominant harmonics."""
    fig, axes = plt.subplots(4, 5, figsize=(15, 13))
    for ax, z, p in zip(axes.flat, recons, percents):
        z = np.append(z, z[0])
        ax.plot(z.real, z.imag, 'k-', lw=1)
        ax.set_title(f'{p}% harmonics')
        _curve_axes(ax, shape)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_animation(Z, freqs, shape, output_path, step=2, fps=30) -> None:
    """MP4 of the rotating epicycles (all harmonics, largest first) drawing the silhouette."""
    from matplotlib import animation
    try:  # ffmpeg bundled with pip's imageio-ffmpeg, falls back to a system ffmpeg
        import imageio_ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    N = len(Z)
    # Frequency 0 is the shape's centroid and never rotates: use it as the fixed base
    centre = Z[freqs == 0].sum() / N
    Z, freqs = Z[freqs != 0], freqs[freqs != 0]
    radii = np.abs(Z) / N
    circle = np.exp(1j * np.linspace(0, 2 * np.pi, 40))
    drawn = radii > 0.5

    fig, ax = plt.subplots(figsize=(6, 6 * shape[0] / shape[1]))
    _curve_axes(ax, shape)
    circles_line, = ax.plot([], [], color='0.7', lw=0.5)
    arms_line, = ax.plot([], [], 'b-', lw=0.8)
    trace_line, = ax.plot([], [], 'r-', lw=1.5)
    trace = []

    def frame(n):
        # Tip-to-tail sum of rotating vectors at time n
        vectors = Z / N * np.exp(2j * np.pi * freqs * n / N)
        joints = centre + np.concatenate(([0], np.cumsum(vectors)))
        trace.append(joints[-1])
        rings = joints[:-1][drawn, None] + radii[drawn, None] * circle
        rings = np.column_stack((rings, np.full(len(rings), np.nan))).ravel()
        circles_line.set_data(rings.real, rings.imag)
        arms_line.set_data(joints.real, joints.imag)
        t = np.array(trace)
        trace_line.set_data(t.real, t.imag)
        return circles_line, arms_line, trace_line

    frames = list(range(0, N, step)) + [N]  # finish back at the start point
    anim = animation.FuncAnimation(fig, frame, frames=frames, blit=True)
    anim.save(output_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
