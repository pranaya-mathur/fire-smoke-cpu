# HF Indoor Quality Report


- Dataset root: `/Users/mobcoderid-296/Desktop/fire-smoke/fire-smoke-cpu/data/raw/hf_kien_fire_smoke/extracted_indoor/Indoor Fire Smoke`
- Images discovered: `5000`
- Valid samples: `4915`
- Excluded: `85`
- Raw split counts: `{'test': 750, 'train': 3500, 'valid': 750}`
- Box counts by raw class id: `{'0': 3592, '1': 3343}`
- Canonical class mapping: `0=fire`, `1=smoke`
- Inferred group count: `1464`
- Groups with >=25 images: `{'hf_kien_indoor:video:280aos_mp4': 51, 'hf_kien_indoor:video:3_mp4': 38, 'hf_kien_indoor:video:6_mp4': 60, 'hf_kien_indoor:video:firevid29_mp4': 49, 'hf_kien_indoor:video:firevid36_mp4': 58, 'hf_kien_indoor:sequence:h': 60, 'hf_kien_indoor:sequence:img': 285, 'hf_kien_indoor:sequence:a': 155, 'hf_kien_indoor:sequence:b': 249, 'hf_kien_indoor:video:classk10_mp4': 40, 'hf_kien_indoor:video:classk11_mp4': 88, 'hf_kien_indoor:sequence:data': 71, 'hf_kien_indoor:video:fiemuc_mp4': 69, 'hf_kien_indoor:sequence:fire_tlm_1vv': 216, 'hf_kien_indoor:sequence:frame': 448, 'hf_kien_indoor:video:part1_mp4': 156, 'hf_kien_indoor:video:part2-trimmed_mp4': 71, 'hf_kien_indoor:video:part2_mp4': 51, 'hf_kien_indoor:video:part5_mp4': 26, 'hf_kien_indoor:video:part6_trim_mp4': 67, 'hf_kien_indoor:video:part7_mp4': 91, 'hf_kien_indoor:video:part8a_mp4': 27, 'hf_kien_indoor:video:part8b_mp4': 49}`

The provider's original train/valid/test split is recorded but not trusted for final training; the project split step rebuilds leakage-aware splits from canonical manifests.

License note: Roboflow YAML says CC BY 4.0, but commercial approval still requires review. This report is not legal advice.

See JSON report for exclusion details.
