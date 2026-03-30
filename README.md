# AO25_visibility_analysis
Scripts used for analysing visibility data from AO2025 cruise onboard IB Oden and compare it to the forecaster's TAFs and ECMWF IFS model data.

- **`vis_functions.py`**:\
    Modular utility library containing the core logic for data parsing, statistical evaluation, and visualization.
    - Data Parsing: Features a custom TAF parser that converts raw string reports into structured objects, extracting visibility and probabilistic trends (TEMPO, PROB30/40).
    - Probabilistic Logic: Implements a weighted probability system (`assign_event_probabilities`) to convert TAF trends into a single event probability, allowing for "Safety-First" vs. "Optimistic" verification.
    - Verification Metrics: Functions to compute standard contingency table metrics including POD, FAR, CSI, and the Equitable Threat Score (ETS).
    - Visualization Suite: A collection of `matplotlib` wrappers for:
    - Time-series analysis: Overlaying TAF validity windows on NWP and observation data.
    - Reliability diagrams: Assessing the calibration of ensemble forecasts.
    - Talagrand (Rank) Histograms: Evaluating ensemble spread and bias.

- **`vis_main.py`**:\
The main analysis pipeline used to ingest xarray/pandas datasets, perform regime-based filtering, and generate the final diagnostic figures and skill scores. It is the execution entry point, managing the data pipeline and high-level analysis.
    - Data Management: Uses Dictionaries (`model_data`, `event_lib`) to store and iterate through multiple forecast sources (IFS, Ensemble means, TAF Forecasters) efficiently.
    - Preprocessing: Handles the alignment of Xarray datasets and Pandas dataframes, ensuring NWP outputs and point-observations are synchronized in time.
    - Sensitivity Analysis: Controls the global configuration for the evaluation, such as the `FOG_THRESH` (e.g., 1km) and the logical switch `HIGHER_THAN_FOG_THRESH` to toggle between verifying "Fog" events and "Clear" windows.
    - Execution Flow:
        1. Loads and cleans raw NWP (NetCDF) and Obs/TAF (CSV) data.
        2. Generates an "Evaluation Library" of boolean event hits/misses.
        3. Produces a summary performance table and triggers the diagnostic plots defined in the functions library.

###  Work in progress.

To do:
- add other scripts (data pre processing + case analysis in Matlab)
- add link to reference data (e.g. Bolin centre + IFS)
- add paper reference/links
- fix headers in the scripts
