"""
Fakeddit Image Downloader  —  fixed version
Creator : Dhanush D  |  dhanushd1812@gmail.com  |  github.com/Drdhx

Usage:
    python image_downloader.py multimodal_train.tsv
    python image_downloader.py multimodal_validate.tsv
    python image_downloader.py multimodal_test_public.tsv
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Fakeddit image downloader (fixed)')
parser.add_argument('type', type=str, help='path to TSV file: multimodal_train.tsv etc.')
args = parser.parse_args()

if not os.path.exists(args.type):
    print(f"ERROR: File not found: {args.type}")
    sys.exit(1)

# ── Load TSV ──────────────────────────────────────────────────────────────────
print(f"Loading {args.type} ...")
df = pd.read_csv(args.type, sep='\t', on_bad_lines='skip')
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)
print(f"Rows loaded: {len(df):,}")

# ── Output folder ─────────────────────────────────────────────────────────────
os.makedirs("images", exist_ok=True)

# ── Counters ──────────────────────────────────────────────────────────────────
downloaded = 0
skipped    = 0
failed     = 0
already    = 0

# ── Request headers (mimic browser to avoid 403 blocks) ──────────────────────
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# ── Download loop ─────────────────────────────────────────────────────────────
pbar = tqdm(total=len(df), unit='img', dynamic_ncols=True)

for _, row in df.iterrows():
    pbar.update(1)

    # Skip rows with no image
    has_image  = str(row.get('hasImage', '')).strip().lower()
    image_url  = str(row.get('image_url', '')).strip()
    post_id    = str(row.get('id', '')).strip()

    if has_image != 'true' or image_url in ('', 'nan') or post_id == '':
        skipped += 1
        continue

    out_path = os.path.join('images', post_id + '.jpg')

    # Skip already downloaded
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        already += 1
        continue

    # Try downloading with retries
    for attempt in range(3):
        try:
            req = urllib.request.Request(image_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            # Only save if looks like an image (not an HTML error page)
            if len(data) > 500:
                with open(out_path, 'wb') as f:
                    f.write(data)
                downloaded += 1
            else:
                failed += 1
            break

        except urllib.error.HTTPError as e:
            # 404 / 403 / 410 — URL is dead, no point retrying
            failed += 1
            break
        except urllib.error.URLError:
            if attempt < 2:
                time.sleep(1)
            else:
                failed += 1
        except Exception:
            if attempt < 2:
                time.sleep(1)
            else:
                failed += 1

    # Live stats in progress bar
    pbar.set_postfix(ok=downloaded, fail=failed, skip=skipped, cached=already)

pbar.close()

# ── Summary ───────────────────────────────────────────────────────────────────
total_imgs = len([f for f in os.listdir('images') if f.endswith('.jpg')])
print("\n" + "=" * 50)
print("  DOWNLOAD SUMMARY")
print("=" * 50)
print(f"  Downloaded  : {downloaded:,}")
print(f"  Already had : {already:,}")
print(f"  Failed/dead : {failed:,}")
print(f"  No image    : {skipped:,}")
print(f"  Total in images/ folder : {total_imgs:,}")
print("=" * 50)
print("Done.")
