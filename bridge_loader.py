"""
Bridge dataset loader.

Returns episodes as (goal, frames_np, actions_np) where:
  frames_np  — (T, H, W, 3) uint8
  actions_np — (T, 7) float32  [x, y, z, rx, ry, rz, gripper]
"""

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

GCS_BASE   = "gs://gresearch/robotics/bridge/0.1.0"
N_SHARDS_TEST  = 512
N_SHARDS_TRAIN = 1024

# Approximate episodes per shard (measured: ~7)
EPISODES_PER_SHARD = 7


def _shard_uri(split: str, shard_idx: int) -> str:
    n = N_SHARDS_TEST if split == "test" else N_SHARDS_TRAIN
    return f"{GCS_BASE}/bridge-{split}.tfrecord-{shard_idx:05d}-of-{n:05d}"


def _parse_episode(raw: bytes):
    """
    Bridge RLDS TFRecords are stored as tf.train.Example (not SequenceExample).
    Each record is one episode; step-level data is packed as flat lists.

    Candidate field names (probe_schema.py will confirm which are present):
      goal  : 'language_instruction'  |  'steps/language_instruction'
      images: 'steps/observation/image_0'  |  'steps/observation/image'
      actions: 'steps/action'
    """
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    f = ex.features.feature

    # ── goal ──────────────────────────────────────────────────────────────────
    for key in ("language_instruction",
                "steps/language_instruction",
                "episode_metadata/language_instruction"):
        if key in f and f[key].bytes_list.value:
            goal = f[key].bytes_list.value[0].decode(errors="replace")
            break
    else:
        goal = ""

    # ── images ────────────────────────────────────────────────────────────────
    for key in ("steps/observation/image_0",
                "steps/observation/image",
                "observation/image_0"):
        if key in f and f[key].bytes_list.value:
            img_bytes = f[key].bytes_list.value
            break
    else:
        raise KeyError("No image field found in episode record")

    frames = np.stack([
        np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        for b in img_bytes
    ])  # (T, H, W, 3)

    # ── actions ───────────────────────────────────────────────────────────────
    n_steps = len(img_bytes)
    for key in ("steps/action", "action"):
        if key in f and f[key].float_list.value:
            act_list = list(f[key].float_list.value)
            actions = np.array(act_list, dtype=np.float32).reshape(n_steps, -1)
            break
    else:
        raise KeyError("No action field found in episode record")

    return goal, frames, actions


def iter_episodes(split: str = "test", max_episodes: int = 200, seed: int = 42):
    """
    Yield (goal, frames, actions) for up to max_episodes Bridge episodes.
    Shards are sampled randomly so episodes are diverse across tasks.
    """
    rng = np.random.default_rng(seed)
    n_shards = N_SHARDS_TEST if split == "test" else N_SHARDS_TRAIN
    shard_order = rng.permutation(n_shards)

    yielded = 0
    for shard_idx in shard_order:
        if yielded >= max_episodes:
            break
        uri = _shard_uri(split, int(shard_idx))
        try:
            ds = tf.data.TFRecordDataset(uri)
            for raw in ds:
                if yielded >= max_episodes:
                    break
                try:
                    yield _parse_episode(raw.numpy())
                    yielded += 1
                except Exception as e:
                    print(f"  [skip] parse error in shard {shard_idx}: {e}")
        except Exception as e:
            print(f"  [skip] shard {shard_idx}: {e}")
