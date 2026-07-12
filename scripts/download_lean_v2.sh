#!/bin/bash
set -e

echo "--- Downloading Medyoussef Hard Negatives ---"
mkdir -p data/raw/medyoussef
ZIP_PATH="data/raw/medyoussef/fire_smoke_hardnegatives_complete.zip"

if [ ! -f "$ZIP_PATH" ]; then
    curl -k -L -s -o "$ZIP_PATH" "https://huggingface.co/datasets/medyoussef/fire-smoke-hardnegatives-int8/resolve/main/fire_smoke_hardnegatives_complete.zip"
fi

echo "Extracting Medyoussef..."
unzip -q -o "$ZIP_PATH" -d data/raw/medyoussef

echo "--- Downloading LibreYOLO Smoke Uvylj ---"
mkdir -p data/raw/libreyolo
python3 -c '
import json, os, subprocess
meta_path = "data/raw/hf_metadata/LibreYOLO_smoke-uvylj_api.json"
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    for s in meta.get("siblings", []):
        rf = s.get("rfilename", "")
        if rf.endswith(".jpg") or rf.endswith(".txt") or rf.endswith("yaml"):
            out = "data/raw/libreyolo/" + rf
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if not os.path.exists(out):
                url = "https://huggingface.co/datasets/LibreYOLO/smoke-uvylj/resolve/main/" + rf
                print(f"echo Downloading {rf}")
                print(f"curl -k -L -s -o \"{out}\" \"{url}\"")
' > download_libreyolo_cmds.sh

bash download_libreyolo_cmds.sh
rm download_libreyolo_cmds.sh

echo "Done downloading lean datasets."
