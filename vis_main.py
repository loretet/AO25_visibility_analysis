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
CONFIG = {
    'fog_thresh': 0.8,
    'higher_than_fog_thresh': True,
    'model_24h': False,
    'start_date': '2025-08-12 00:00',
    'end_date': '2025-09-16 00:00',
    'time_res': 'h',
    'paths': {
        'taf': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/AO25_TAFs.xlsx',
        'obs': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/obs_data/AO2025_MDF_20250812-20250915_hourly_quantiles_10minmin.nc',
        'ens': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_ens_oden_2025-08-11_2025-09-15_vis_day2.nc'
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
        'path': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v1.nc',
        'type': 'det', 'var': 'vis', 'color': 'blue'
    },
    'LowLvlMean': {
        'path': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_diagnostic_lowLvlMean.nc',
        'type': 'det', 'var': 'vis', 'color': 'tab:blue'
    },
    'Persist_10min': {
        'path': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/AO2025_20250812-20250915_persistence_forecast_v2.nc',
        'type': 'pers', 'var': 'persistence10m_minimum', 'color': 'black'
    },
    'Ens_P20':    {'type': 'ens_prob', 'thresh': 0.20, 'color': 'darkgreen'},
    'Ens_P50':    {'type': 'ens_prob', 'thresh': 0.50, 'color': 'tab:green'},
    'Ens_P80':    {'type': 'ens_prob', 'thresh': 0.80, 'color': 'lightgreen'}
}

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

# --- A. Load Observations ---
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

# --- C. Load Models via Registry ---
event_val = 10.0 if CONFIG['higher_than_fog_thresh'] else 0.0
non_event_val = 0.0 if CONFIG['higher_than_fog_thresh'] else 10.0

# Pre-load ensemble data if requested by the registry
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
    
    # 2. Dynamic thresholding for numerical models and TAF interpretive choices
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
            period_splits[split_name] = vf.compute_all_metrics(window_truth, window_models)
            
        matrix_results[regime].append({
            'period': p_name,
            'splits': period_splits
        })

# --- Consolidated Console Outputs (Example: Entire Cruise, High-Vis) ---
eval_mask_fc = (taf_eval.index >= CONFIG['start_date']) & (taf_eval.index <= CONFIG['end_date']) & mask_valid
bs_ens = vf.compute_brier_score(prob_fog[eval_mask_fc], (taf_eval['obs_vis'] > CONFIG['fog_thresh'])[eval_mask_fc].astype(float))
final_res_high = matrix_results['high'][-1]['splits']['Full']

print("\n--- EVALUATION SUMMARY (ENTIRE CRUISE - HIGH VISIBILITY WINDOW) ---")
print(f"Ensemble Brier score: {bs_ens:.4f}\n")
print(final_res_high.to_string(float_format="%.3f"))


#%% 4. VISUALIZATION
# ---------------------------------------------------------
# Dispatch processed dual structures directly to the matrix plotting function.
# ---------------------------------------------------------

# Generate the 4x2 Matrix Plot showing High Visibility (Col 0) and Low Visibility (Col 1)
fig, axs = vf.plot_multi_period_performance_matrix(
    results_high=matrix_results['high'],
    results_low=matrix_results['low'],
    period_names=[d[2] for d in PERIODS],
    model_style_map=MODEL_STYLE,
    all_periods=True
)
plt.savefig("performance_matrix_full.pdf")

# 2. Metrics Summary (example for entire period, considering both halves )
fig1, fig2 = vf.plot_metrics_summary(matrix_results["high"][0]["splits"]["Full"])
fig1.suptitle("Windows of opportunity"); fig2.suptitle("Windows of opportunity")
fig1, fig2 = vf.plot_metrics_summary(matrix_results["low"][0]["splits"]["Full"])
fig1.suptitle("Low visibility events"); fig2.suptitle("Low visibility events")

# 3. Ensemble Diagnostics
vf.plot_reliability_diagram(prob_fog, taf_eval['obs_event'], n_bins=20)
vf.plot_talagrand_histogram(ens_aligned, taf_eval['obs_vis'])

#%%

