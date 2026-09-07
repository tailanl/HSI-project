"""Shared explicit imports for the static scene-only algorithms.

    Model requests use the configured client's non-blocking reservation. The
    retained review context below performs no second locking or queuing.
"""
import copy
import itertools
import json
import math
import os
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

from hsi.common.artifacts import (
    MemoryContractError, artifact, digest, read_json as read, read_sealed,
    require, sha256, verified, write_once,
)


def exclusive_qwen():
    """Transport owns its one optional fail-fast reservation; no hidden queue."""
    return nullcontext()
