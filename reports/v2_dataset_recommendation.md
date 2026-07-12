# V2 Dataset Inclusion Recommendation

This policy was generated after reviewing Hugging Face candidate metadata and licenses.

## Included Datasets

1. **badsaarow/d-fire**
   - **Status**: INCLUDE_WITH_CAUTION
   - **Reason**: 21.5k samples of fire and smoke. Requires careful validation of negative presence and annotation quality since license is unknown.
   - **Expected Benefit**: Broad foundation for general fire/smoke.
   - **Expected Risk**: Potential annotation noise or synthetic bias.

2. **LibreYOLO/smoke-uvylj**
   - **Status**: INCLUDE
   - **Reason**: CC-BY-4.0 license, dedicated smoke annotations.
   - **Expected Benefit**: Acts as a smoke booster to address the 2.43x recall deficit seen in V1.
   - **Expected Risk**: Bounding box standards might not align perfectly with D-Fire.

3. **medyoussef/fire-smoke-hardnegatives-int8**
   - **Status**: INCLUDE_WITH_CAUTION
   - **Reason**: Dedicated hard negatives, but license is unknown.
   - **Expected Benefit**: Massive reduction in false alarms on indoor CCTV (which V1 completely lacked).
   - **Expected Risk**: Provenance/copyright risks for commercial use.

4. **KienNgyuen/Fire-Smoke-Detection**
   - **Status**: INCLUDE
   - **Reason**: This is the existing Dataset V1 source (supplemental positive).
   - **Expected Benefit**: Maintains baseline performance anchor.
   - **Expected Risk**: Should be capped in size so it doesn't dominate V2 diversity.

## Quarantined / Excluded Datasets

5. **YingjieCheng/FireSmokeDetDatasets**
   - **Status**: QUARANTINE
   - **Reason**: Annotations marked as 'unknown' format, requires manual parsing/conversion pipeline before it can be merged safely.

6. **hiennguyen9874/fire-smoke-detection**
   - **Status**: EXCLUDE
   - **Reason**: Size is 11.5 GB compressed (90k+ images). The dataset would dominate the split balancing and exhaust the 27GB free disk space limit. 

7. **betasecond/jimei-fire-smoke-yolo-dataset**
   - **Status**: QUARANTINE
   - **Reason**: Unknown license and duplicate risk. Needs further verification against D-Fire.
