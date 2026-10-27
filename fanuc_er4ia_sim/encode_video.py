"""
Encode the PNG frames written by make_video.py into an mp4.

    ~/fanuc_er4ia_sim/.venv/bin/python encode_video.py --frames frames/ --out demo.mp4
"""

import argparse
import glob
import os


def load_dependencies():
    """Load encoder packages with an actionable dependency diagnostic."""
    try:
        import cv2
        import imageio_ffmpeg
        import numpy as np
    except ModuleNotFoundError as error:
        package = "imageio-ffmpeg" if error.name == "imageio_ffmpeg" else error.name
        raise RuntimeError(
            f"video encoding unavailable: missing Python package {package!r}; "
            "install the repository's declared requirements.txt dependencies"
        ) from error
    return imageio_ffmpeg, np, cv2


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="frames")
    parser.add_argument("--out", default="demo.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--check-dependencies", action="store_true")
    args = parser.parse_args(argv)
    imageio_ffmpeg, _np, cv2 = load_dependencies()
    if args.check_dependencies:
        print("video encoder dependencies available")
        return 0

    files = sorted(glob.glob(os.path.join(args.frames, "*.png")))
    if not files:
        raise RuntimeError(f"no PNGs in {args.frames}")
    first = cv2.imread(files[0])
    if first is None:
        raise RuntimeError(f"could not decode first PNG: {files[0]}")
    h, w = first.shape[:2]
    print(f"{len(files)} frames, {w}x{h} @ {args.fps} fps -> {args.out}")

    writer = imageio_ffmpeg.write_frames(
        args.out, (w, h), fps=args.fps, quality=8, macro_block_size=1
    )
    writer.send(None)
    for filename in files:
        frame = cv2.imread(filename)
        if frame is None:
            raise RuntimeError(f"could not decode PNG: {filename}")
        writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
    writer.close()
    print(
        f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, "
        f"{len(files)/args.fps:.1f} s)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
