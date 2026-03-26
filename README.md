# AO25_visibility_analysis
Scripts used for analysing visibility data from AO2025 cruise onboard IB Oden and compare it to the forecaster's TAFs and ECMWF IFS model data.

- **`vis_functions.py`**:
    A utility library containing the core logic for TAF string decomposition, meteorological verification metrics, and specialized atmospheric plotting routines.
- **`vis_main.py`**:
    The main analysis pipeline used to ingest xarray/pandas datasets, perform regime-based filtering, and generate the final diagnostic figures and skill scores.



###  Work in progress.

To do:
- add other scripts (data pre processing + case analysis in Matlab)
- add link to reference data (e.g. Bolin centre + IFS)
- add paper reference/links
- fix headers in the scripts
