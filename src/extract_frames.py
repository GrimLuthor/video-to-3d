"""Extract frames from a video at a fixed stride for pairwise pose chaining.

Deliberately does not use ex4's imageio-based frame loading (that path is
tied to ex4's optical-flow mosaicing code); this reads directly with
cv2.VideoCapture since it's the same library the rest of Stage 1 depends on.
"""

import cv2


def extract_frames(video_path, stride=5, max_frames=None):
    """Returns a list of BGR frames (numpy arrays), taking every `stride`-th
    decoded frame. Raises if the video can't be opened or yields nothing.

    NOTE: cv2.VideoCapture does not reliably apply phone-recorded rotation
    metadata (unlike video players). Always check frame.shape after
    extraction against the visual orientation of the source video before
    trusting any K matrix derived for a particular width/height.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return frames


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video_path> [stride]")
        sys.exit(1)

    video_path = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    frames = extract_frames(video_path, stride=stride)
    h, w = frames[0].shape[:2]
    print(f"Extracted {len(frames)} frames, each {w}x{h} (w x h), stride={stride}")
    print("Confirm this orientation/aspect matches what you expect before "
          "trusting any K derived for this resolution.")
