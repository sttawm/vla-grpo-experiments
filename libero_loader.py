"""
LIBERO-Long data loader for lerobot/libero_10_image.
Yields (goal: str, frames: list[np.ndarray uint8], actions: np.ndarray[T,7]).

Dataset has 379 total episodes (single HF "train" split).
Deterministic 200/50/129 episode-level train/val/test split (MASTER_SEED=0).
  train: 299 episodes (~80%) — used for gradient updates
  val:    40 episodes (~10%) — episode-disjoint from train, used for checkpoint selection
  test:   40 episodes (~10%) — held out until final evaluation
"""

import json
import numpy as np
from functools import lru_cache
from pathlib import Path

DATASET_ID     = "lerobot/libero_10_image"
TOTAL_EPISODES = 379
_MASTER_SEED   = 0
_TRAIN_SIZE    = 299
_VAL_SIZE      = 40   # episode-disjoint from train; remaining 40 → test

STATS_PATH = Path(__file__).parent / "results" / "libero_action_stats.json"


def _split_episode_ids() -> tuple[list[int], list[int], list[int]]:
    """Deterministic 200/50/129 episode split. Returns (train_ids, val_ids, test_ids)."""
    rng = np.random.RandomState(_MASTER_SEED)
    idx = list(range(TOTAL_EPISODES))
    rng.shuffle(idx)
    train_ids = idx[:_TRAIN_SIZE]
    val_ids   = idx[_TRAIN_SIZE:_TRAIN_SIZE + _VAL_SIZE]
    test_ids  = idx[_TRAIN_SIZE + _VAL_SIZE:]
    return train_ids, val_ids, test_ids


@lru_cache(maxsize=1)
def _task_map() -> dict[int, str]:
    """Load task_index → language description mapping from dataset metadata."""
    import os
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download(DATASET_ID, "meta/tasks.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    # df index = task string, df["task_index"] = int
    return {int(row["task_index"]): str(task) for task, row in df.iterrows()}


def _load_dataset_smart():
    """Load from local blob cache (streaming=True over disk) if available, else HF network stream.

    streaming=False builds an Arrow cache which needs ~2x the dataset disk space.
    Using streaming=True with local data_files reads parquet blobs directly — no extra space needed.
    """
    import os
    from pathlib import Path
    from datasets import load_dataset

    hf_home = Path(os.environ.get("HF_HOME", "/workspace/hf_cache"))
    snap_base = hf_home / "hub" / "datasets--lerobot--libero_10_image" / "snapshots"
    if snap_base.exists():
        snaps = [p for p in snap_base.iterdir() if p.is_dir()]
        if snaps:
            parquet_glob = str(snaps[0] / "data" / "**" / "*.parquet")
            return load_dataset("parquet", data_files={"train": parquet_glob},
                                split="train", streaming=True)
    return load_dataset(DATASET_ID, split="train", streaming=True)


def iter_episodes(split: str, max_episodes: int, seed: int = 42,
                  yield_ep_idx: bool = False):
    """
    Yield (goal, frames, actions) for up to max_episodes episodes.
    If yield_ep_idx=True, yields (ep_idx, goal, frames, actions) instead,
    where ep_idx is the absolute episode index in the dataset (stable across seeds).

    split: "train" → 299-episode train split (episode-disjoint from val);
           "val"   → 40-episode val split (episode-disjoint from train);
           "test"  → 40-episode held-out test split.
    seed:  determines ordering/subset within the split pool.
    frames:  list of np.ndarray uint8 (H, W, 3), one per timestep.
    actions: np.ndarray float32 (T, 7) — [x, y, z, roll, pitch, yaw, gripper].
    """
    import os
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    from datasets import load_dataset

    train_ids, val_ids, test_ids = _split_episode_ids()
    if split == "train":
        ep_ids = train_ids
    elif split == "val":
        ep_ids = val_ids
    else:
        ep_ids = test_ids

    rng = np.random.RandomState(seed)
    ep_ids = list(ep_ids)
    rng.shuffle(ep_ids)
    target_set = set(ep_ids[:max_episodes])

    task_map = _task_map()
    ds = _load_dataset_smart()
    ep_ds = ds.filter(
        lambda batch: [int(e) in target_set for e in batch["episode_index"]],
        batched=True, batch_size=2048,
    )

    cur_ep: int = -1
    cur_frames: list[np.ndarray] = []
    cur_actions: list[list[float]] = []
    cur_task_idx: int = 0
    yielded = 0

    for row in ep_ds:
        if yielded >= max_episodes:
            break
        ep_idx = int(row["episode_index"])

        if ep_idx != cur_ep:
            if cur_frames and yielded < max_episodes:
                goal = task_map.get(cur_task_idx, f"task_{cur_task_idx}")
                if yield_ep_idx:
                    yield cur_ep, goal, cur_frames, np.array(cur_actions, dtype=np.float32)
                else:
                    yield goal, cur_frames, np.array(cur_actions, dtype=np.float32)
                yielded += 1
            cur_ep = ep_idx
            cur_frames = []
            cur_actions = []
            cur_task_idx = int(row["task_index"])

        cur_frames.append(np.array(row["observation.images.image"], dtype=np.uint8))
        cur_actions.append(list(row["action"]))

    # Final flush
    if cur_frames and yielded < max_episodes:
        goal = task_map.get(cur_task_idx, f"task_{cur_task_idx}")
        if yield_ep_idx:
            yield cur_ep, goal, cur_frames, np.array(cur_actions, dtype=np.float32)
        else:
            yield goal, cur_frames, np.array(cur_actions, dtype=np.float32)


def compute_action_stats(save_path: Path = STATS_PATH) -> dict:
    """Compute per-dim mean/std over all train-split episodes. Saves to JSON."""
    import os
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

    train_ids, _, _ = _split_episode_ids()
    target_set = set(train_ids)

    ds = _load_dataset_smart()
    ds = ds.filter(
        lambda batch: [int(e) in target_set for e in batch["episode_index"]],
        batched=True, batch_size=2048,
    )
    all_actions: list[list[float]] = []
    n_ep = 0
    last_ep = -1

    print(f"Computing LIBERO action stats over {len(train_ids)} train episodes...")
    for row in ds:
        ep_idx = int(row["episode_index"])
        if ep_idx != last_ep:
            n_ep += 1
            last_ep = ep_idx
            if n_ep % 50 == 0:
                print(f"  ...{n_ep}/{len(train_ids)} episodes")
        all_actions.append(list(row["action"]))

    arr  = np.array(all_actions, dtype=np.float32)
    mean = arr.mean(axis=0).tolist()
    std  = [max(float(s), 1e-6) for s in arr.std(axis=0).tolist()]
    q01  = np.percentile(arr, 1,  axis=0).tolist()
    q99  = np.percentile(arr, 99, axis=0).tolist()
    mn   = arr.min(axis=0).tolist()
    mx   = arr.max(axis=0).tolist()

    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    print(f"Action stats over {len(arr)} frames ({n_ep} episodes):")
    for d, m, s, lo, hi in zip(dim_names, mean, std, q01, q99):
        print(f"  {d}: mean={m:+.4f}  std={s:.4f}  q01={lo:+.4f}  q99={hi:+.4f}")

    stats = {
        "mean": mean, "std": std,
        "q01": q01, "q99": q99,
        "min": mn, "max": mx,
        "n_frames": int(len(arr)),
    }
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved to {save_path}")
    return stats


def load_action_stats() -> dict:
    """Load cached action stats, computing from scratch on first call."""
    if not STATS_PATH.exists():
        return compute_action_stats()
    with open(STATS_PATH) as f:
        return json.load(f)
