# Maintainer: L.L. Donati - lorenzo.luca.donati@misu.su.se
# Scripts for "On the predictability of near-surface visibility over the Arctic Ocean"
# by Luise Schulte, Lorenzo Luca Donati, Vania Lopez Garcia, Linus Magnusson, Ian M. Brooks

# AI disclosure:
# AI was used to populate this script with comments and docstrings, and to
# assist in the structuring the plots in a more visually appealing way. The core logic, 
# data handling, and metric calculations were developed by the author.

#%% Imports
import vis_functions as vf
import importlib
importlib.reload(vf)
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

#%% 1. CONFIGURATION AND REGISTRY
# ---------------------------------------------------------
# All parameters, paths, and metadata governing the analysis.
# ---------------------------------------------------------
RUN_DIR="/Users/lodo0477/Documents/PhD/Research/Visibility study"
CONFIG = {
    'fog_thresh': 0.8,
    'higher_than_fog_thresh': False,
    'model_24h': False,
    'start_date': '2025-08-12 00:00',
    'end_date': '2025-09-16 00:00',
    'time_res': 'h',
    'paths': {
        'taf': f'{RUN_DIR}/AO25_TAFs.xlsx',
        'obs': f'{RUN_DIR}/obs_data/AO2025_MDF_20250812-20250915_hourly_quantiles_10minmin.nc',
        'ens': f'{RUN_DIR}/model_data/ifs_ens_oden_2025-08-11_2025-09-15_vis_day2.nc'
    },
    'obs_var': 'visas_10min'
}

PERIODS = [
    ('2025-08-12 00:00', '2025-08-16 12:00', 'Period 1'),
    ('2025-08-16 13:00', '2025-09-03 00:00', 'Period 2'),
    ('2025-09-03 01:00', '2025-09-16 00:00', 'Period 3'),
    (CONFIG['start_date'], CONFIG['end_date'], 'Entire Cruise')
]

# The registry defines HOW data is loaded and evaluated. 
# Types: 'det' (deterministic), 'pers' (persistence), 'ens_prob' (ensemble probability)
MODEL_REGISTRY = {
    'IFS': {
        'path': f'{RUN_DIR}/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v1.nc',
        'type': 'det', 'var': 'vis', 'color': 'blue'
    },
    'LowLvlMean': {
        'path': f'{RUN_DIR}/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v2.nc',
        'type': 'det', 'var': 'vis_from_hydro_lowLvlMean_7levels', 'color': 'tab:blue'
    },
    'Persist_10min': {
        'path': f'{RUN_DIR}/model_data/AO2025_20250812-20250915_persistence_forecast_v2.nc',
        'type': 'pers', 'var': 'persistence10m_minimum', 'color': 'black'
    },
    'Ens_P20':    {'type': 'ens_prob', 'thresh': 0.20, 'color': 'darkgreen'},
    'Ens_Median':    {'type': 'ens_prob', 'thresh': 0.50, 'color': 'tab:green'},
    'Ens_P80':    {'type': 'ens_prob', 'thresh': 0.80, 'color': 'lightgreen'}
}  # for high visibility, P20 is optimistic ("as long as 20% of memebrs say it's clear, then it's clear.")
   # on the other had, for low visibility it is pessimistic ("when even as low as 20% of memebrs say it's foggy, then it's foggy.")
FC_STYLES = {
    'base':         {'color': 'red',     'label': 'TAF (Base)'},
    'conservative': {'color': 'red',     'label': 'TAF ("Any")'},
    'first_half':   {'color': 'purple',  'label': 'TAF (First Half)'},
    'second_half':  {'color': 'magenta', 'label': 'TAF (Second Half)'}
}


#%% 2. DATA PIPELINE 
# ---------------------------------------------------------
# Extract, transform, and load data uniformly to time_vec.
# ---------------------------------------------------------
time_vec = pd.date_range(start=CONFIG['start_date'], end=CONFIG['end_date'], freq=CONFIG['time_res'], inclusive="both")
# model_data will contain the physical visibility values.
model_data = {}

# --- A. Load observations ---
with xr.open_dataset(CONFIG['paths']['obs'], decode_timedelta=True) as ds_obs:
    vis_obs = np.clip(ds_obs[CONFIG['obs_var']].to_series() * 1e-3, 0, 10).reindex(time_vec)

# --- B. Load TAFs ---
taf_table = pd.read_excel(CONFIG['paths']['taf'], header=1, sheet_name='Sheet1').dropna(subset=['TAF Oden']).reset_index(drop=True)
taf_table['Date'] = pd.to_datetime(taf_table['Date'])
mask = (taf_table['Date'] >= pd.to_datetime(CONFIG['start_date']).normalize()) & \
       (taf_table['Date'] <= pd.to_datetime(CONFIG['end_date']).normalize())

taf_eval = vf.df_TAF_gen(taf_table.loc[mask].reset_index(drop=True), time_vec, debug=False)
taf_eval = vf.calculate_scenarios(taf_eval)
taf_eval = vf.assign_event_probabilities(taf_eval, CONFIG['fog_thresh'], CONFIG['higher_than_fog_thresh'])

taf_eval['obs_vis'] = vis_obs
taf_eval['obs_event'] = (taf_eval['obs_vis'] > CONFIG['fog_thresh']).astype(float) if CONFIG['higher_than_fog_thresh'] else \
                       (taf_eval['obs_vis'] <= CONFIG['fog_thresh']).astype(float)

model_data.update({
    'TAF_Base': taf_eval['main_scenario'],
    'TAF_Pessimistic': taf_eval['worst_vis'],
    'TAF_Optimistic': taf_eval['best_vis']
})

# --- C. Load models via registry ---
event_val = 10.0 if CONFIG['higher_than_fog_thresh'] else 0.0
non_event_val = 0.0 if CONFIG['higher_than_fog_thresh'] else 10.0

# Pre-load ensemble data 
ens_aligned = None
prob_fog = None
if any(meta['type'].startswith('ens') for meta in MODEL_REGISTRY.values()):
    with xr.open_dataset(CONFIG['paths']['ens'], decode_timedelta=True) as ds_ens:
        ens_aligned = ds_ens.vis.clip(min=0, max=10000) * 1e-3
        prob_fog = (ens_aligned > CONFIG['fog_thresh']).mean(dim='number') if CONFIG['higher_than_fog_thresh'] else \
                   (ens_aligned <= CONFIG['fog_thresh']).mean(dim='number')
        prob_fog = prob_fog.to_series().reindex(time_vec)

for name, meta in MODEL_REGISTRY.items():
    m_type = meta['type']
    
    if m_type in ['det', 'pers']:
        with xr.open_dataset(meta['path'], decode_timedelta=True) as ds:
            series = ds[meta['var']].to_series().reindex(time_vec) * 1e-3
            model_data[name] = np.clip(series, 0, 10)
            
    elif m_type == 'ens_prob':
        model_data[name] = pd.Series(np.where(prob_fog >= meta['thresh'], event_val, non_event_val), index=time_vec)

# Generate style mapping expected by plotting functions
MODEL_STYLE = {name: meta['color'] for name, meta in MODEL_REGISTRY.items() if 'color' in meta}
MODEL_STYLE.update({
    'TAF_Base': 'red',
    'TAF_Pessimistic': 'purple',
    'TAF_Optimistic': 'magenta'
})

#%% 3. EVALUATION 
# ---------------------------------------------------------
# Compute binary verification metrics over defined periods and TAF halves
# for BOTH High-Visibility and Low-Visibility regimes.
# ---------------------------------------------------------

# Dictionaries to store multi-period results for both verification targets
matrix_results = {
    'high': [], # corresponds to higher_than_fog_thresh = True
    'low':  []  # corresponds to higher_than_fog_thresh = False
}

# Explicitly evaluate both threshold conditions sequentially
for regime in ['high', 'low']:
    is_high_target = (regime == 'high')
    
    # 1. Dynamically update the baseline observation events for this target
    taf_eval['obs_event'] = (taf_eval['obs_vis'] > CONFIG['fog_thresh']).astype(float) if is_high_target else \
                           (taf_eval['obs_vis'] <= CONFIG['fog_thresh']).astype(float)
    
    # 2. Dynamic thresholdg for numerical models and TAF interpretive choices
    truth, ev_lib = vf.get_evaluation_library(
        model_data, taf_eval['obs_vis'], 
        fog_thresh=CONFIG['fog_thresh'], 
        higher_than_fog_thresh=is_high_target
    )
    
    # Separate models from base forecaster dictionary entry
    models_lib = {k: v for k, v in ev_lib.items() if k != 'Forecaster'}
    
    # 3. Validity Masks Setup
    mask_valid = taf_eval['is_valid'] == True
    mask_1st   = taf_eval['is_valid_first_half'] == True
    mask_2nd   = taf_eval['is_valid_second_half'] == True
    
    # 4. Compute split metrics across all time windows
    for start_t, end_t, p_name in PERIODS:
        t_mask = (taf_eval.index >= start_t) & (taf_eval.index <= end_t)
        
        sub_periods = {
            'Full':        t_mask if CONFIG['model_24h'] else (t_mask & mask_valid),
            'First_Half':  t_mask & mask_1st,
            'Second_Half': t_mask & mask_2nd
        }
        
        period_splits = {}
        for split_name, current_mask in sub_periods.items():
            window_truth = truth.loc[current_mask]
            window_models = {k: v.loc[current_mask] for k, v in models_lib.items()}
            
            # Compute standard metrics dataframe (Rows: Models, Cols: POD, FAR, Hits, etc.)
            metrics_df = vf.compute_all_metrics(window_truth, window_models)
            
            # Compute ETS for each row using the existing contingency values
            metrics_df['ETS'] = metrics_df.apply(
                lambda row: vf.calculate_ets(
                    a=row['Hits'], 
                    b=row['False alarms'], 
                    c=row['Misses'], 
                    d=row['Correct negatives']
                ), 
                axis=1
            )
            period_splits[split_name] = metrics_df
            
        matrix_results[regime].append({
            'period': p_name,
            'splits': period_splits
        })

# High-Visibility Regime (enitre period)
# Note: Inverting prob_fog to represent probability of visibility > threshold
eval_mask_fc = (taf_eval.index >= CONFIG['start_date']) & (taf_eval.index <= CONFIG['end_date']) & mask_valid
prob_clear = 1.0 - prob_fog
obs_clear = (taf_eval['obs_vis'] > CONFIG['fog_thresh']).astype(float)
bs_ens_high = vf.compute_brier_score(prob_clear[eval_mask_fc], obs_clear[eval_mask_fc])
final_res_high = matrix_results['high'][-1]['splits']['Full']
print("\n=== EVALUATION SUMMARY (ENTIRE CRUISE - HIGH VISIBILITY WINDOW) ===")
print(f"Ensemble Brier Score (Clear): {bs_ens_high:.4f}\n")
print(final_res_high.to_string(float_format="%.3f"))
print("-" * 67)

# Low-Visibility Regime entire period)
obs_fog = (taf_eval['obs_vis'] <= CONFIG['fog_thresh']).astype(float)
bs_ens_low = vf.compute_brier_score(prob_fog[eval_mask_fc], obs_fog[eval_mask_fc])
final_res_low = matrix_results['low'][-1]['splits']['Full']
print("\n=== EVALUATION SUMMARY (ENTIRE CRUISE - LOW VISIBILITY WINDOW) ===")
print(f"Ensemble Brier Score (Fog):   {bs_ens_low:.4f}\n")
print(final_res_low.to_string(float_format="%.3f"))
print("=" * 67)

#%% 4. VISUALIZATION
# ---------------------------------------------------------
# Dispatch processed dual structures directly to the matrix plotting function.
# ---------------------------------------------------------

# Generate the 4x2 or 1x2 matrix oplot showing High Visibility (Col 0) and Low Visibility (Col 1)
fig, axs = vf.plot_multi_period_performance_matrix(
    results_high=matrix_results['high'],
    results_low=matrix_results['low'],
    period_names=["Entire cruise"],
    model_style_map=MODEL_STYLE,
    all_periods=False,
    insets=True,
    plot_halves=True
)

# 2. Metrics summary (example for entire period [3], considering both halves )
fig1, fig2 = vf.plot_metrics_summary(matrix_results["high"][2]["splits"]["Full"])
fig1.suptitle("Windows of opportunity"); fig2.suptitle("Windows of opportunity")
fig1, fig2 = vf.plot_metrics_summary(matrix_results["low"][2]["splits"]["Full"])
fig1.suptitle("Low visibility events"); fig2.suptitle("Low visibility events")

# 3. Ensemble diagnostics
vf.plot_reliability_diagram(prob_fog, taf_eval['obs_event'], n_bins=20)
vf.plot_talagrand_histogram(ens_aligned, taf_eval['obs_vis'])

# 4. Flexible Visibility Summary Meteogram 
# Programmatically parse only continuous/physical time series from the pipeline
# Filter data to TAF validity times only (set to NaN outside validity window)
vis_obs_filtered = vis_obs.copy()
vis_obs_filtered[~(taf_eval['is_valid'] == True)] = np.nan

meteo_filtered = {
    'Observations': (vis_obs_filtered, 'crimson', '-', 2.0, "o")
}
for name, meta in MODEL_REGISTRY.items():
    if meta['type'] in ['det', 'pers']:  # Filter out threshold step-functions
        linestyle = '--' if meta['type'] == 'det' else ':'
        if meta["type"]=='pers':
            linestyle="-"
        thick = 4 if meta['type'] == 'pers' else 1.5
        model_series = model_data[name].copy()
        model_series[~(taf_eval['is_valid'] == True)] = np.nan
        meteo_filtered[name] = (model_series, meta['color'], linestyle, thick, None)

# Call the profile summary over an interesting sub-window (e.g., Period 1)
fig_met, ax_met = vf.plot_vis_summary(
    df=taf_eval,
    series_dict=meteo_filtered,
    fog_thresh=CONFIG['fog_thresh'],
    start_date='2025-08-31',
    end_date='2025-09-15'
)
ax_met.set_title("Log-Scale Visibility Time Series Comparison (Sub-Window Test)", fontweight='bold')


#%% 5. SEEPS Computation and calculation

# Three visibility categories:
#   V <= 0.8 km
#   0.8 < V <= 2.0 km
#   V > 2.0 km
# ONE common climatological SEEPS matrix is estimated from the full-cruise observations and reused for every period.

SEEPS_THRESHOLDS = (0.8, 3.0)
seeps_t1, seeps_t2 = SEEPS_THRESHOLDS

# Construct three requested constant-category reference forecasts
# Useful to check if SEEPS is correctly computed:
# they should be zero for the netire period and different than zero for the individual ones

constant_low_value = 0.7
constant_mid_value = 0.5 * (seeps_t1 + seeps_t2)
constant_high_value = 10.0

# Backup data
seeps_model_data = model_data.copy()

# Also create a "perfect" forecast by copying the observations
seeps_model_data["Perfect"] = taf_eval["obs_vis"].copy()
seeps_model_data.update(
    {
        "Always_700m": pd.Series(
            constant_low_value,
            index=time_vec,
            dtype=float,
        ),
        "Always_Mid": pd.Series(
            constant_mid_value,
            index=time_vec,
            dtype=float,
        ),
        "Always_10000m": pd.Series(
            constant_high_value,
            index=time_vec,
            dtype=float,
        ),
    }
)


# Define ONE observational climatology for the scoring matrix.

seeps_common_valid = (
    (taf_eval.index >= CONFIG["start_date"])
    & (taf_eval.index <= CONFIG["end_date"])
    & (taf_eval["is_valid"] == True)
)

seeps_climatology_obs = taf_eval.loc[
    seeps_common_valid,
    "obs_vis"
].dropna()


# Calculate full-cruise climatological prob and matrix
_, seeps_score_matrix, seeps_probs = (
    vf.compute_seeps_visibility(
        obs=seeps_climatology_obs,
        forecasts={
            "_dummy": seeps_climatology_obs,
        },
        thresholds_km=SEEPS_THRESHOLDS,
        climatology_obs=seeps_climatology_obs,
    )
)

# Show matrix and p1 p2 p3:
print("\n=== VISIBILITY SEEPS CONFIGURATION ===")
print(
    f"Thresholds: "
    f"{seeps_t1 * 1000:.0f} m, "
    f"{seeps_t2 * 1000:.0f} m"
)
print("\nClimatological category probabilities:")
print(seeps_probs.to_string(float_format="%.4f"))
print("\nSEEPS error matrix:")
print(
    pd.DataFrame(
        seeps_score_matrix,
        index=["FC Low","FC Intermediate", "FC High"],
        columns=["OBS Low", "OBS Intermediate","OBS High"],
    ).to_string(float_format="%.3f")
)

# Evaluate each operational period.
#
# Each period uses:
#    - the observations inside that period
#    - the model forecats inside that period
#    - the SAME full-cruise climatological probabilities
#
# This means differences between periods represent forecast
# performance rather than changes in the scoring matrix

seeps_period_results = {}
seeps_sample_sizes = {}

for start_t, end_t, period_name in PERIODS:
    period_mask = (
        (taf_eval.index >= start_t)
        & (taf_eval.index <= end_t)
        & (taf_eval["is_valid"] == True)
    )
    obs_period = taf_eval.loc[
        period_mask,
        "obs_vis"
    ]
    forecasts_period = {
        name: series.reindex(taf_eval.index).loc[period_mask]
        for name, series in seeps_model_data.items()
    }

    period_result, _, _ = vf.compute_seeps_visibility(
        obs=obs_period,
        forecasts=forecasts_period,
        thresholds_km=SEEPS_THRESHOLDS,
        climatology_obs=seeps_climatology_obs,
    )

    seeps_period_results[period_name] = (
        period_result["SEEPS_skill"]
    )
    seeps_sample_sizes[period_name] = (
        period_result["N"]
    )

# Plot tables
# rows    -> periods
# columns -> model

seeps_skill_table = pd.DataFrame(
    seeps_period_results
).T
seeps_n_table = pd.DataFrame(
    seeps_sample_sizes
).T
print("\n=== VISIBILITY SEEPS SKILL ===")
print(seeps_skill_table.to_string(float_format="%.3f"))

print("\n=== VALID SAMPLE SIZE ===")
print(seeps_n_table.to_string(float_format="%.0f"))

#%% 6. SEEPS VISUALIZATION

SEEPS_MODEL_STYLE = MODEL_STYLE.copy()

SEEPS_MODEL_STYLE.update(
    {
        "Always_700m": "0.35",
        "Always_Mid": "0.55",
        "Always_10000m": "0.75",
    }
)

fig_seeps, ax_seeps = vf.plot_seeps_skill(
    skill_table=seeps_skill_table,
    thresholds_km=SEEPS_THRESHOLDS,
    model_style_map=SEEPS_MODEL_STYLE,
    highlight_period="Entire Cruise",
    periods_to_plot=None
)
# Cut axis for visualization purposes
ax_seeps.set_ylim(top=1.,bottom=-0.4)

#%%