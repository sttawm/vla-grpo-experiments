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
    Bridge stores each episode as a SequenceExample with all step data packed
    as flat lists in context features.

    Confirmed field names (from probe_schema.py):
      goal    : steps/observation/natural_language_instruction  (bytes, T values, all same)
      images  : steps/observation/image                         (bytes, T PNG-encoded frames)
      actions : steps/action/world_vector   (float, T*3) +
                steps/action/rotation_delta (float, T*3) +
                steps/action/open_gripper   (int64, T)   → concatenated to (T, 7)
    """
    ex = tf.train.SequenceExample()
    ex.ParseFromString(raw)
    f = ex.context.feature

    # ── goal (same string repeated T times — take first) ─────────────────────
    goal = f["steps/observation/natural_language_instruction"].bytes_list.value[0].decode(errors="replace").strip()

    # ── images ────────────────────────────────────────────────────────────────
    img_bytes = f["steps/observation/image"].bytes_list.value
    frames = np.stack([
        np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        for b in img_bytes
    ])  # (T, H, W, 3)

    # ── actions: xyz + rpy + gripper → (T, 7) ────────────────────────────────
    T = len(img_bytes)
    xyz     = np.array(f["steps/action/world_vector"].float_list.value,   dtype=np.float32).reshape(T, 3)
    rpy     = np.array(f["steps/action/rotation_delta"].float_list.value, dtype=np.float32).reshape(T, 3)
    gripper = np.array(f["steps/action/open_gripper"].int64_list.value,   dtype=np.float32).reshape(T, 1)
    actions = np.concatenate([xyz, rpy, gripper], axis=1)  # (T, 7)

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
