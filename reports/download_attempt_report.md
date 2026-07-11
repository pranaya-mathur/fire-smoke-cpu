# Download Attempt Report

Timestamp UTC: `2026-07-11T15:49:20Z`

## D-Fire

- Official GitHub repository is already cloned at `data/raw/dfire/DFireDataset`.
- The actual image/label archive is hosted via official OneDrive links in the D-Fire README.
- Automated attempts:
  - OneDrive short link with and without `download=1`: HTTP `403`.
  - Official OneDrive shared-link API content endpoint: HTTP `401 BadContextToken`.
- Status: `AUTOMATED_DOWNLOAD_BLOCKED`.
- Required action: download through a browser-authenticated OneDrive session into `data/raw/dfire`.

## MS-FSDB

- Official Google Drive file ID: `14ylxaNBVmXjAFXt2h4lnyBe7xELhOVHc`.
- Automated attempts:
  - `gdown`: could not retrieve public link.
  - Google Drive `uc` endpoint: redirects to Google usercontent and returns HTTP `404`.
- Status: `AUTOMATED_DOWNLOAD_BLOCKED`.
- Required action: download through a browser-authenticated Google Drive session or have the provider enable public link access, then place the archive/extracted files in `data/raw/ms_fsdb`.

## MIVIA

- Official page is accessible.
- The page routes downloads through `/datasets-request/`, an account/request workflow.
- Status: `AUTOMATED_DOWNLOAD_BLOCKED`.
- Required action: complete the official MIVIA request/download flow in a browser and place videos in `data/raw/mivia`.

## FASDD

- DOI resolves to ScienceDB.
- Official ScienceDB `getAllUrl` exposed:
  `https://download.scidb.cn/download?fileId=62d56ad75336efe8bf53d28e&path=/V1/FASDD.zip&fileName=FASDD.zip`
- Archive size from headers: `36,018,644,602` bytes, about `33.5 GiB`.
- Local free disk from the hardware report: about `28.8 GiB`.
- Status: `DISCOVERED_NOT_DOWNLOADED`.
- Reason: insufficient local disk and FASDD is Phase 2 only.
