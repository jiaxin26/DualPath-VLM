"""
Diagnose AMBER data loading issues.

Usage:
  PYTHONPATH=. python experiments/diagnose_amber.py \
    --amber-path data/amber/ \
    --amber-image-dir data/amber/images/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amber-path", default="data/amber/")
    parser.add_argument("--amber-image-dir", default="data/amber/images/")
    args = parser.parse_args()

    print(f"AMBER path: {args.amber_path}")
    print(f"AMBER image dir: {args.amber_image_dir}")
    print()

    # 1. Check directory exists
    if not os.path.isdir(args.amber_path):
        print(f"ERROR: AMBER directory not found: {args.amber_path}")
        return

    # 2. List top-level contents
    print("=== AMBER directory contents ===")
    for item in sorted(os.listdir(args.amber_path)):
        full = os.path.join(args.amber_path, item)
        if os.path.isdir(full):
            n_files = len(os.listdir(full))
            print(f"  [DIR]  {item}/ ({n_files} items)")
        else:
            size = os.path.getsize(full)
            print(f"  [FILE] {item} ({size:,} bytes)")

    # 3. Find the data file
    candidates = [
        "data/query/query_discriminative.json",
        "query_discriminative.json",
        "discriminative.json",
        "amber_discriminative.jsonl",
    ]
    data_file = None
    for name in candidates:
        path = os.path.join(args.amber_path, name)
        if os.path.exists(path):
            data_file = path
            break

    if data_file is None:
        print(f"\nERROR: No AMBER data file found. Tried: {candidates}")
        # Search recursively
        print("\nSearching recursively for JSON files...")
        for root, dirs, files in os.walk(args.amber_path):
            for f in files:
                if f.endswith(('.json', '.jsonl')):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, args.amber_path)
                    size = os.path.getsize(full)
                    print(f"  Found: {rel} ({size:,} bytes)")
        return

    print(f"\n=== Data file: {data_file} ===")

    # 4. Load and inspect samples
    with open(data_file) as f:
        if data_file.endswith(".jsonl"):
            raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            raw_data = json.load(f)

    print(f"Total samples: {len(raw_data)}")

    # Type distribution
    types = {}
    for item in raw_data:
        t = item.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"Type distribution: {json.dumps(types, indent=2)}")

    # 5. Inspect first 3 samples
    print("\n=== First 3 samples ===")
    for i, item in enumerate(raw_data[:3]):
        print(f"\nSample {i}:")
        for k, v in item.items():
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"  {k}: {val_str}")

    # 6. Check image paths
    print("\n=== Image path analysis ===")
    sample_images = [item.get("image", "") for item in raw_data[:20]]

    for img in sample_images[:5]:
        print(f"  Raw image field: {img[:120]}...")

    # Check if images are URLs or local paths
    is_url = any(img.startswith("http") for img in sample_images if img)
    is_local = any(not img.startswith("http") for img in sample_images if img)
    print(f"\n  Contains URLs: {is_url}")
    print(f"  Contains local paths: {is_local}")

    # Try to find images
    print(f"\n=== Checking image directory: {args.amber_image_dir} ===")
    if os.path.isdir(args.amber_image_dir):
        img_files = os.listdir(args.amber_image_dir)
        print(f"  Found {len(img_files)} files")
        if img_files:
            print(f"  First 5: {img_files[:5]}")
    else:
        print(f"  Directory NOT FOUND")
        # Try common alternatives
        alts = [
            os.path.join(args.amber_path, "image"),
            os.path.join(args.amber_path, "images"),
            os.path.join(args.amber_path, "data", "image"),
            os.path.join(args.amber_path, "data", "images"),
        ]
        for alt in alts:
            if os.path.isdir(alt):
                n = len(os.listdir(alt))
                print(f"  FOUND alternative: {alt} ({n} files)")

    # 7. Test resolve
    print("\n=== Testing image resolution ===")
    for item in raw_data[:5]:
        img = item.get("image", "")
        # Try direct
        direct = os.path.join(args.amber_image_dir, img)
        # Try basename
        basename = os.path.join(args.amber_image_dir, os.path.basename(img))
        # Try without prefix
        exists_direct = os.path.exists(direct)
        exists_base = os.path.exists(basename)
        print(f"  image='{img[:80]}' -> direct={exists_direct}, basename={exists_base}")


if __name__ == "__main__":
    main()
