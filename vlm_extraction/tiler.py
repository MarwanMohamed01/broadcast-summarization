"""Slice a wide panorama into VLM-digestible tiles.

Each tile is `config.TILE_WIDTH` px wide with `config.TILE_OVERLAP` px
overlap with its neighbours, so headlines straddling a tile boundary
are captured intact in at least one tile. Tiles are written to
`config.TILE_CACHE_DIR/<panorama_stem>/tile_NNN.png` and the function
is idempotent — already-cached tiles are reused.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from . import config

logger = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None  # slice panoramas are >100 M px


def tile_panorama(panorama_path: Path,
                  tile_width: int = config.TILE_WIDTH,
                  overlap: int = config.TILE_OVERLAP) -> list[Path]:
    """Return a list of tile paths (cached on disk). Idempotent.

    Tiles are 1-indexed in the filename (`tile_001.png`, ...).
    """
    if not panorama_path.exists():
        raise FileNotFoundError(panorama_path)

    cache_dir = config.TILE_CACHE_DIR / panorama_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(panorama_path) as im:
        width, height = im.size
        if width <= tile_width:
            # Panorama already fits — emit a single tile that's just a copy.
            single = cache_dir / "tile_001.png"
            if not single.exists():
                im.save(single)
            return [single]

        stride = max(1, tile_width - overlap)
        starts: list[int] = list(range(0, max(0, width - tile_width) + 1, stride))
        # Make sure the rightmost edge is covered.
        if not starts or starts[-1] + tile_width < width:
            starts.append(width - tile_width)

        tiles: list[Path] = []
        for i, x in enumerate(starts, start=1):
            tile_path = cache_dir / f"tile_{i:03d}.png"
            if not tile_path.exists():
                tile = im.crop((x, 0, x + tile_width, height))
                tile.save(tile_path)
            tiles.append(tile_path)
        logger.info("tiled %s -> %d tiles in %s",
                    panorama_path.name, len(tiles), cache_dir)
        return tiles


def tiles_for_slice(slice_name: str) -> list[Path]:
    """Concatenate tiles across all source panoramas for a named slice."""
    out: list[Path] = []
    for pano_path in config.PANORAMA_SOURCES[slice_name]:
        out.extend(tile_panorama(pano_path))
    return out
