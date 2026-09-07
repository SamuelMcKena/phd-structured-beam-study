"""Convert the older realistic setup artwork to a white-background poster derivative.

This is presentation-only image processing. It does not create or modify optical data.
The source artwork should be supplied explicitly; the output is a visual schematic only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert(source: Path, output: Path) -> None:
    image = Image.open(source).convert("RGB")
    rgb = np.asarray(image).copy()
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation_proxy = maxc - minc
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    # Neutral near-black background becomes white.
    background = (luminance < 28) & (saturation_proxy < 35)
    rgb[background] = 255

    # Neutral white labels/lines become dark on the new background.
    pale_neutral = (luminance > 165) & (saturation_proxy < 38)
    rgb[pale_neutral] = np.array([45, 55, 65], dtype=np.uint8)

    # Remap intermediate neutral greys so annotations remain visible on white.
    grey = (luminance >= 28) & (luminance <= 165) & (saturation_proxy < 22)
    values = luminance[grey]
    remapped = np.clip(190 - values * 0.75, 55, 175).astype(np.uint8)
    rgb[grey] = np.stack([remapped, remapped, remapped], axis=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
