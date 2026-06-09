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

FC_STYLES = {
    'base':         {'color': 'red',     'label': 'TAF (Base)'},
    'conservative': {'color': 'red',     'label': 'TAF ("Any")'},
    'first_half':   {'color': 'purple',  'label': 'TAF (First Half)'},
    'second_half':  {'color': 'magenta', 'label': 'TAF (Second Half)'}
}

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


#%% 2. DATA PIPELINE (ETL)
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


#%% 3. EVALUATION 
# ---------------------------------------------------------
# Compute binary verification metrics over defined periods and TAF halves.
# ---------------------------------------------------------

# 1. Convert all continuous data in model_data into boolean events at once.
# ev_lib converts continuous visibility values into binary True/False events based on the fog threshold.
truth, ev_lib = vf.get_evaluation_library(
    taf_eval, model_data, taf_eval['obs_vis'], 
    fog_thresh=CONFIG['fog_thresh'], 
    higher_than_fog_thresh=CONFIG['higher_than_fog_thresh']
)

# Extract all thresholded series (Numerical Models + TAF variants)
models_lib = {k: v for k, v in ev_lib.items() if k != 'Forecaster'}
# multi_period_results will contain scalar verification scores across different time windows.
multi_period_results = []

# Define validity masks
mask_valid = taf_eval['is_valid'] == True
mask_1st   = taf_eval['is_valid_first_half'] == True
mask_2nd   = taf_eval['is_valid_second_half'] == True

for start_t, end_t, p_name in PERIODS:
    t_mask = (taf_eval.index >= start_t) & (taf_eval.index <= end_t)
    
    # Generate the evaluation masks for each split context
    sub_periods = {
        'Full':       t_mask if CONFIG['model_24h'] else (t_mask & mask_valid),
        'First_Half':  t_mask & mask_1st,
        'Second_Half': t_mask & mask_2nd
    }
    
    # Compute metrics for all elements in models_lib across each split
    period_splits = {}
    for split_name, current_mask in sub_periods.items():
        window_truth = truth.loc[current_mask]
        window_models = {k: v.loc[current_mask] for k, v in models_lib.items()}
        
        # vf.compute_all_metrics returns a DataFrame of scores for all models
        period_splits[split_name] = vf.compute_all_metrics(window_truth, window_models)
        
    multi_period_results.append({
        'period': p_name,
        'splits': period_splits
    })

# --- Console Outputs (Example: Entire Cruise) ---
eval_mask_fc = (taf_eval.index >= CONFIG['start_date']) & (taf_eval.index <= CONFIG['end_date']) & mask_valid
bs_ens = vf.compute_brier_score(prob_fog[eval_mask_fc], taf_eval['obs_event'][eval_mask_fc])

# Extract the 'Full' split for the final period (Entire Cruise)
final_res = multi_period_results[-1]['splits']['Full']

print("\n--- EVALUATION SUMMARY (ENTIRE CRUISE - FULL WINDOW) ---")
print(f"Ensemble Brier score: {bs_ens:.4f}\n")
print(final_res.to_string(float_format="%.3f"))
print(f"\n{'Model':<25} | {'ETS':<10} | Contingency [Hits, Misses, FA, CN]")
print("-" * 65)
for name, r in final_res.iterrows():
    ets = vf.calculate_ets(r["Hits"], r["False alarms"], r["Misses"], r["Correct negatives"])
    print(f"{name:<25} | {ets:<10.4f} | [{int(r['Hits'])}, {int(r['Misses'])}, {int(r['False alarms'])}, {int(r['Correct negatives'])}]")

# Console Outputs
eval_mask_fc = (taf_eval.index >= CONFIG['start_date']) & (taf_eval.index <= CONFIG['end_date']) & mask_valid
bs_ens = vf.compute_brier_score(prob_fog[eval_mask_fc], taf_eval['obs_event'][eval_mask_fc])

final_res = pd.concat([
    multi_period_results[-1]['models'], 
    pd.DataFrame([multi_period_results[-1]['fc_base']], index=['Forecaster_base']),
    pd.DataFrame([multi_period_results[-1]['fc_any']], index=['Forecaster_any'])
])

print("\n--- EVALUATION SUMMARY (ENTIRE CRUISE) ---")
print(f"Ensemble Brier score: {bs_ens:.4f}\n")
print(final_res.to_string(float_format="%.3f"))
print(f"\n{'Model':<25} | {'ETS':<10} | Contingency [Hits, Misses, FA, CN]")
print("-" * 65)
for name, r in final_res.iterrows():
    ets = vf.calculate_ets(r["Hits"], r["False alarms"], r["Misses"], r["Correct negatives"])
    print(f"{name:<25} | {ets:<10.4f} | [{int(r['Hits'])}, {int(r['Misses'])}, {int(r['False alarms'])}, {int(r['Correct negatives'])}]")


#%% 4. VISUALIZATION
# ---------------------------------------------------------
# Dispatch processed structures to visualization functions.
# ---------------------------------------------------------

# 1. Performance Diagram
vf.plot_multi_period_performance(
    results_list=multi_period_results,
    period_names=[d[2] for d in PERIODS],
    model_style_map=MODEL_STYLE,
    fc_style_map=FC_STYLES,
    higher_than_fog_thresh=CONFIG['higher_than_fog_thresh']
)

# 2. Metrics Summary
fig1, fig2 = vf.plot_metrics_summary(final_res)
fig1.suptitle("Metrics summary"); fig2.suptitle("Metrics summary")

# 3. Meteogram (Deterministic models only via dictionary comprehension)
vf.plot_ens_meteogram(
    prob_df=vf.calculate_stacked_probabilities(taf_eval), 
    model_dict={k: v for k, v in model_data.items() if 'Ens' not in k and 'TAF' not in k}, 
    vis_obs=taf_eval['obs_vis'], 
    start_date='2025-08-20', 
    end_date='2025-08-30'
)

# 4. Ensemble Diagnostics
if ens_aligned is not None:
    vf.plot_reliability_diagram(prob_fog, taf_eval['obs_event'], n_bins=20)
    vf.plot_talagrand_histogram(ens_aligned, taf_eval['obs_vis'])


# #%%

