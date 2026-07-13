#!/bin/bash
mkdir -p data/raw/hf_metadata
cd data/raw/hf_metadata

echo "Downloading metadata..."

datasets=(
  "medyoussef/fire-smoke-hardnegatives-int8"
  "LibreYOLO/smoke-uvylj"
  "YingjieCheng/FireSmokeDetDatasets"
  "KienNgyuen/Fire-Smoke-Detection"
  "hiennguyen9874/fire-smoke-detection"
  "betasecond/jimei-fire-smoke-yolo-dataset"
)

for ds in "${datasets[@]}"; do
  safe_name=$(echo "$ds" | tr '/' '_')
  echo "Fetching $ds..."
  curl -k -s "https://huggingface.co/api/datasets/$ds" > "${safe_name}_api.json"
  curl -k -s "https://datasets-server.huggingface.co/info?dataset=$ds" > "${safe_name}_info.json"
done

echo "Done fetching metadata."
