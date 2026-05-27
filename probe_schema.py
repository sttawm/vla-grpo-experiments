"""
Run this once on the remote machine to verify Bridge TFRecord field names.
  python probe_schema.py [--split test] [--shard 0]
"""

import argparse
import tensorflow as tf

GCS_BASE       = "gs://gresearch/robotics/bridge/0.1.0"
N_SHARDS_TEST  = 512
N_SHARDS_TRAIN = 1024


def probe(split: str, shard_idx: int):
    n = N_SHARDS_TEST if split == "test" else N_SHARDS_TRAIN
    uri = f"{GCS_BASE}/bridge-{split}.tfrecord-{shard_idx:05d}-of-{n:05d}"
    print(f"Reading: {uri}\n")

    ds = tf.data.TFRecordDataset(uri)
    raw = next(iter(ds)).numpy()

    # Try SequenceExample first
    try:
        ex = tf.train.SequenceExample()
        ex.ParseFromString(raw)
        print("=== SequenceExample ===")
        print("Context features:")
        for k, v in ex.context.feature.items():
            kind = v.WhichOneof("kind")
            if kind == "bytes_list":
                n_vals = len(v.bytes_list.value)
                sample = v.bytes_list.value[0][:40] if n_vals else b""
                print(f"  {k!r:60s} bytes_list  len={n_vals}  sample={sample}")
            elif kind == "float_list":
                vals = list(v.float_list.value)
                print(f"  {k!r:60s} float_list  len={len(vals)}  sample={vals[:5]}")
            elif kind == "int64_list":
                vals = list(v.int64_list.value)
                print(f"  {k!r:60s} int64_list  len={len(vals)}  sample={vals[:5]}")
        print("\nFeature lists (sequence part):")
        for k, v in ex.feature_lists.feature_list.items():
            first = v.feature[0] if v.feature else None
            kind = first.WhichOneof("kind") if first else "?"
            print(f"  {k!r:60s} {kind}  n_steps={len(v.feature)}")
    except Exception as e:
        print(f"Not a valid SequenceExample: {e}")

    # Also try plain Example
    try:
        ex2 = tf.train.Example()
        ex2.ParseFromString(raw)
        print("\n=== Example ===")
        for k, v in ex2.features.feature.items():
            kind = v.WhichOneof("kind")
            if kind == "bytes_list":
                n_vals = len(v.bytes_list.value)
                sample = v.bytes_list.value[0][:40] if n_vals else b""
                print(f"  {k!r:60s} bytes_list  len={n_vals}")
            elif kind == "float_list":
                vals = list(v.float_list.value)
                print(f"  {k!r:60s} float_list  len={len(vals)}  sample={vals[:5]}")
            elif kind == "int64_list":
                vals = list(v.int64_list.value)
                print(f"  {k!r:60s} int64_list  len={len(vals)}  sample={vals[:5]}")
    except Exception as e:
        print(f"Not a valid Example: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()
    probe(args.split, args.shard)
