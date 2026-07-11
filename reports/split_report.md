# Split Report

- Total samples: `4915`
- Train/val/test counts: `{'train': 3319, 'val': 772, 'test': 824}`
- Label counts: `{'train': Counter({'smoke_only': 1591, 'fire_only': 1359, 'fire_smoke': 369}), 'val': Counter({'fire_only': 536, 'smoke_only': 210, 'fire_smoke': 26}), 'test': Counter({'smoke_only': 519, 'fire_only': 261, 'fire_smoke': 44})}`
- Source counts: `{'train': Counter({'hf_kien_indoor': 3319}), 'val': Counter({'hf_kien_indoor': 772}), 'test': Counter({'hf_kien_indoor': 824})}`
- Exact duplicate split leakage checks: `0`
- Seed: `42`

Splits are group-aware using inferred source/video/filename groups. Exact and near-duplicate reports should be reviewed before final training.
