"""Generate a checkerboard pattern PNG for screen-based camera calibration
(no printer needed -- display this full-screen on a monitor/laptop instead
of printing it on paper).

Usage:
    python generate_checkerboard.py [cols] [rows] [square_px]

Defaults to 9x6 inner corners (10x7 squares), matching the pattern size
assumed by calibrate_from_video.py and ex2_plan.md.
"""

import sys
from pathlib import Path

from PIL import Image

OUT_DIR = Path(__file__).parent.parent / "calibration"


def generate_checkerboard(cols=9, rows=6, square_px=160):
    """cols, rows = inner corner counts (OpenCV convention). Produces a
    (cols+1) x (rows+1) squares image with a one-square white quiet zone
    border (findChessboardCorners needs the border to detect the outer
    corners reliably).
    """
    n_squares_x = cols + 1
    n_squares_y = rows + 1
    margin = square_px

    width = n_squares_x * square_px + 2 * margin
    height = n_squares_y * square_px + 2 * margin

    img = Image.new("L", (width, height), color=255)
    pixels = img.load()

    for sy in range(n_squares_y):
        for sx in range(n_squares_x):
            if (sx + sy) % 2 == 0:
                x0 = margin + sx * square_px
                y0 = margin + sy * square_px
                for y in range(y0, y0 + square_px):
                    for x in range(x0, x0 + square_px):
                        pixels[x, y] = 0

    return img


if __name__ == "__main__":
    cols = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    square_px = int(sys.argv[3]) if len(sys.argv) > 3 else 160

    img = generate_checkerboard(cols, rows, square_px)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"checkerboard_{cols}x{rows}.png"
    img.save(out_path)

    print(f"Saved {img.width}x{img.height} checkerboard ({cols}x{rows} inner "
          f"corners) to {out_path}")
    print()
    print("Next steps:")
    print("  1. Open this PNG and display it FULL SCREEN on a laptop/monitor")
    print("     (image viewer or browser, whatever avoids letterboxing).")
    print("  2. Max out screen brightness; dim room lights to kill glare.")
    print("  3. Measure the width of ONE black or white square on the")
    print("     screen with a ruler, in mm. You'll pass this as")
    print("     --square-size-mm to calibrate_from_video.py.")
    print("  4. Record a calibration video with the phone in the SAME video")
    print("     mode/resolution/aspect ratio you'll use for the real street")
    print("     footage (EIS off). Slowly move/tilt the phone so the board")
    print("     appears at varied distances, angles, and screen positions")
    print("     (including near the frame edges) for ~20-30s.")
