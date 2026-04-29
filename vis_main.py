# Ref: L. Donati
# lorenzo.luca.donati@misu.su.se

# Scripts for "Paper title"
# by Authors...

#%% Imports
import vis_functions as vf
import importlib         # look for changes in vis_functions.py
importlib.reload(vf)     # without having to reload the kernel
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

#%% PREPROCESS DATASETS
# Flags:
preproc = False   # whether to preproc original dataset with different diagnostics (necesary only once)
debug   = False   # whether to print debugging lines

if preproc:
# Rename the different visibilities from the original diagnostic datasets to "vis"
# to be used with the functions in the script more easily.
    # 1. Load the dataset
    ds_base = xr.open_dataset('/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v1.nc',decode_times=True)
    # 2. Drop the original 'vis'
    ds_dropped = ds_base.drop_vars('vis')
    # 3. Create the three datasets
    ds_list = {
        'lowLvlMean': ds_dropped.rename({'vis_from_hydro_lowLvlMean': 'vis'}),
        'lowLvlSum':  ds_dropped.rename({'vis_from_hydro_lowLvlSum': 'vis'}),
        'lowestLvl':  ds_dropped.rename({'vis_from_hydro_lowestLvl': 'vis'})
    }
    for name, ds in ds_list.items():
        ds.to_netcdf(f"ifs_diagnostic_{name}.nc")

#%% SETTINGS AND PATHS
# Settings
FOG_THRESH = 0.8  # km  (0.8 km Cassel Aero threshold, 1 km WMO threshold)
HIGHER_THAN_FOG_THRESH = False  # if True, looks at windows of opportnity (high visibility). If False, looks at low vis. events
MODEL_24h = False # Whether to evaluate the full 24h forecast or just the TAF validity times:
                  #   True: the model gets evaluated over 24h, while the forecaster only on its active time
                  #   False: both model and forecaster are evaluated only on the TAFs validity window. Better imho

# Cruise period and resolution
START_DATE = '2025-08-12 00:00'
END_DATE = '2025-09-16 00:00'
TIME_RES = 'h'  # Analysis resolution (minutes, hours..) 

# File Paths (observation ALREADY preprocessed with hourly median/quantiles and divided in quantiles, from data repository)
TAF_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/AO25_TAFs.xlsx'
OBS_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/obs_data/AO2025_MDF_20250812-20250915_hourly_quantiles.nc'
visas = "visas_5min"

# Create model dictionary
MODEL_PATHS = {
    'IFS': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v1.nc',
    'lowLvlMean': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_diagnostic_lowLvlMean.nc',
}

PERS_PATH = "/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/AO2025_20250812-20250915_persistence_forecast.nc"

ENS_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_ens_oden_2025-08-11_2025-09-15_vis_day2.nc'

# Visual configuration
MODEL_STYLE = {
    'IFS': 'blue',
    'lowLvlMean': 'green',
    'Ens_Median': 'orange',
    'Ens_BestCase': 'red',  
    'Ens_WorstCase': 'purple',
    'Ens_P20': 'brown',
    'Ens_P30': 'pink',
    'pers_ds_5min': 'cyan',
    'pers_ds_median': 'olive'
}
FC_STYLES = {
    'base': {'color': 'black', 
             'marker': 'D', 
             'label': 'Forecaster (Base: 0.5)'},
    'conservative': {'color': 'darkgray', 
                     'marker': 'X', 
                     'label': 'Forecaster (All: 0.0)'},
    'first_half': {
        'color': 'magenta',
        'marker': '^',
        'label': 'Forecaster (First Half)'
    },
    'second_half': {
        'color': 'teal',
        'marker': 'v',
        'label': 'Forecaster (Second Half)'
    }
}

DATES = [
    ('2025-08-12 00:00', '2025-08-16 12:00', 'Period 1'),
    ('2025-08-16 13:00', '2025-09-03 00:00', 'Period 2'),
    ('2025-09-03 01:00', '2025-09-16 00:00', 'Period 3'),
    (START_DATE, END_DATE, 'Entire Cruise')
]

#%% BASELINE DATA PREP

# 0. Initialize Time Vector
time_vec = pd.date_range(start=START_DATE, end=END_DATE, freq=TIME_RES, inclusive="both")

# 1. Read models into the dictionary
model_data = {}
for name, path in MODEL_PATHS.items():
    # Process deterministic forecasts
    with xr.open_dataset(path, decode_timedelta=True) as ds:
        # Explicitly check for the variable
        if 'vis' not in ds:
            print(f"Error: Variable 'vis' not found in {path}. Available: {list(ds.data_vars)}")
        # Ensure we are interpolating the specific dataset object
        series = ds.vis.to_series().reindex(time_vec) * 1e-3
        model_data[name] = np.clip(series, 0, 10)
        if debug:
            print(f"Processed {name} with values ranging {model_data[name].min():.2f} to {model_data[name].max():.2f}")

# Process ensemble forecasts
with xr.open_dataset(ENS_PATH, decode_timedelta=True) as ds_ens:
    # 1. Align ensemble members to your timeline (time, number)
    # Convert to km and clip for consistency
    ens_aligned = ds_ens.vis.clip(min=0, max=10000) * 1e-3
    
    # 2. Extract "Main Scenario" (Median)
    model_data['Ens_Median'] = ens_aligned.median(dim='number').to_series().reindex(time_vec)
    
    # 3. Extract "Worst Case" (Minimum member)
    model_data['Ens_WorstCase'] = ens_aligned.min(dim='number').to_series().reindex(time_vec)
    # 4. Probabilistic Triggers
    # Calculate fraction of members
    if HIGHER_THAN_FOG_THRESH:
        prob_fog = (ens_aligned > FOG_THRESH).mean(dim='number').to_series().reindex(time_vec)
    else:
        prob_fog = (ens_aligned <= FOG_THRESH).mean(dim='number').to_series().reindex(time_vec)
    
    # Create binary series: If prob > X%, we set vis to 0.0 (Fog), else 10.0 (Clear)
    event_value = 10.0 if HIGHER_THAN_FOG_THRESH else 0.0
    non_event_value = 0.0 if HIGHER_THAN_FOG_THRESH else 10.0

    model_data['Ens_P20'] = pd.Series(np.where(prob_fog > 0.20, event_value, non_event_value), index=time_vec)
    model_data['Ens_P30'] = pd.Series(np.where(prob_fog > 0.30, event_value, non_event_value), index=time_vec)
    model_data['Ens_BestCase'] = pd.Series(np.where(prob_fog > 0.98, event_value, non_event_value), index=time_vec)

# Process "persistent" forecaster data
pers_ds = xr.open_dataset(PERS_PATH, decode_timedelta=True)
model_data["pers_ds_median"] = np.clip(pers_ds.persistence_median.to_series() * 1e-3, 0, 10).reindex(time_vec)
model_data["pers_ds_5min"] =np.clip(pers_ds.persistence_5min.to_series() * 1e-3, 0, 10).reindex(time_vec)

# 2. Load and Process TAFs
taf_table = pd.read_excel(TAF_PATH, header=1, sheet_name='Sheet1').reset_index(drop=True)
taf_table = taf_table[taf_table['TAF Oden'].notna()]  # ensure no empty rows
taf_table['Date'] = pd.to_datetime(taf_table['Date'])
start_day_table = pd.to_datetime(START_DATE).normalize()
end_day_table = pd.to_datetime(END_DATE).normalize()
mask = (taf_table['Date'] >= start_day_table) & (taf_table['Date'] <= end_day_table)
taf_table = taf_table.loc[mask].reset_index(drop=True)

# START_DATE **must** match furst row of the Excel
df_eval = vf.df_TAF_gen(taf_table, time_vec, debug)
df_eval = vf.calculate_scenarios(df_eval)
df_eval = vf.assign_event_probabilities(df_eval, FOG_THRESH, HIGHER_THAN_FOG_THRESH)

# 3. Load and Align Observations
ds_obs = xr.open_dataset(OBS_PATH, decode_timedelta=True)

# Apply smoothing and align to time_vec for both 5min and 15min
vis_obs_5min = np.clip(ds_obs["visas_5min"].to_series() * 1e-3, 0, 10).reindex(time_vec)
vis_obs_15min = np.clip(ds_obs["visas_15min"].to_series() * 1e-3, 0, 10).reindex(time_vec)

# Add to evaluation dataframe (Keeping obs_vis for standard 5min fallback plots)
df_eval['obs_vis'] = vis_obs_5min 
df_eval['obs_vis_5min'] = vis_obs_5min
df_eval['obs_vis_15min'] = vis_obs_15min

if HIGHER_THAN_FOG_THRESH:
    df_eval['obs_event_5min'] = (df_eval['obs_vis_5min'] > FOG_THRESH).astype(float)
    df_eval['obs_event_15min'] = (df_eval['obs_vis_15min'] > FOG_THRESH).astype(float)
    df_eval['obs_event'] = df_eval['obs_event_5min']
else:
    df_eval['obs_event_5min'] = (df_eval['obs_vis_5min'] <= FOG_THRESH).astype(float)
    df_eval['obs_event_15min'] = (df_eval['obs_vis_15min'] <= FOG_THRESH).astype(float)
    df_eval['obs_event'] = df_eval['obs_event_5min']

# Check number of visibility observations with vis < 800m
if debug:
    cp,cm=0,0
    for i in vis_obs_15min:
        if i <= FOG_THRESH:
            cm +=1
        elif i > FOG_THRESH:
            cp +=1
    print(f"[15-min obs] Count of points with vis > {FOG_THRESH*1e3}m: {cp}")
    print(f"[15-min obs] Count of points with vis <= {FOG_THRESH*1e3}m: {cm}  [{cm*100/(cm+cp):.1f}% of the total]")
    cp,cm=0,0
    for i in vis_obs_5min:
        if i <= FOG_THRESH:
            cm +=1
        elif i > FOG_THRESH:
            cp +=1
    print(f"[5-min obs] Count of points with vis > {FOG_THRESH*1e3}m: {cp}")
    print(f"[5-min obs] Count of points with vis <= {FOG_THRESH*1e3}m: {cm}  [{cm*100/(cm+cp):.1f}% of the total]")

# Check data overlap
if debug: 
    valid_times = df_eval['is_valid'].sum()
    print(f"Total time units with valid TAFs: {valid_times}")
    if valid_times == 0:
        print("WARNING: No TAFs were mapped to the time vector. Check START_DATE/Index alignment.")
    if HIGHER_THAN_FOG_THRESH:
        mask5 = (vis_obs_5min > FOG_THRESH) & (df_eval['is_valid'] == True)
        mask15 = (vis_obs_15min > FOG_THRESH) & (df_eval['is_valid'] == True)
    else:   
        mask5 = (vis_obs_5min <= FOG_THRESH) & (df_eval['is_valid'] == True)
        mask15 = (vis_obs_15min <= FOG_THRESH) & (df_eval['is_valid'] == True)
    vis_count_5min = mask5.sum()
    vis_count_15min = mask15.sum()
    print(f"[15-min obs] Low visibility events in TAF validity window: {vis_count_15min}")  
    print(f"[5-min obs] Low visibility events in TAF validity window: {vis_count_5min}")  

# Debugging time 
if debug:
    print("### DEBUGGING time_vec ###")
    for name, series in model_data.items():
        print(name, series.index.min(), series.index.max(), len(series))
    print("time_vec:", time_vec.min(), time_vec.max(), len(time_vec))
    print("Obs 5 min:", vis_obs_5min.index.min(), vis_obs_5min.index.max(), len(vis_obs_5min))
    print("Obs 15 min:", vis_obs_15min.index.min(), vis_obs_15min.index.max(), len(vis_obs_15min))
    print("TAFs:", df_eval.index.min(), df_eval.index.max(), len(df_eval))

#%% MODEL EVALUATION

# Define analysis periods
DATES = [
    ('2025-08-12 00:00', '2025-08-16 12:00', 'Period 1'),
    ('2025-08-16 13:00', '2025-09-03 00:00', 'Period 2'),
    ('2025-09-03 01:00', '2025-09-16 00:00', 'Period 3'),
    (START_DATE, END_DATE, 'Entire Cruise')
]

# 1. Generate full event libraries for BOTH thresholds AND BOTH observation types
truth_5min, ev_lib_05_5min = vf.get_evaluation_library(df_eval, model_data, df_eval['obs_vis_5min'], p_thresh=0.5, fog_thresh=FOG_THRESH, higher_than_fog_thresh=HIGHER_THAN_FOG_THRESH)
truth_15min, ev_lib_05_15min = vf.get_evaluation_library(df_eval, model_data, df_eval['obs_vis_15min'], p_thresh=0.5, fog_thresh=FOG_THRESH, higher_than_fog_thresh=HIGHER_THAN_FOG_THRESH)

_, ev_lib_00_5min = vf.get_evaluation_library(df_eval, model_data, df_eval['obs_vis_5min'], p_thresh=0.0, fog_thresh=FOG_THRESH, higher_than_fog_thresh=HIGHER_THAN_FOG_THRESH)
_, ev_lib_00_15min = vf.get_evaluation_library(df_eval, model_data, df_eval['obs_vis_15min'], p_thresh=0.0, fog_thresh=FOG_THRESH, higher_than_fog_thresh=HIGHER_THAN_FOG_THRESH)

# Isolate models 
models_lib_5min = {k: v for k, v in ev_lib_05_5min.items() if k != 'Forecaster'}
models_lib_15min = {k: v for k, v in ev_lib_05_15min.items() if k != 'Forecaster'}

multi_period_results = []
mask_valid = df_eval['is_valid'] == True
mask_valid_first = df_eval['is_valid_first_half'] == True
mask_valid_second = df_eval['is_valid_second_half'] == True

for start_t, end_t, p_name in DATES:
    t_mask = (df_eval.index >= start_t) & (df_eval.index <= end_t)
    
    # Compute only in TAF validity times if MODEL_24h=False
    eval_mask_models = t_mask if MODEL_24h else t_mask & mask_valid
    eval_mask_fc = t_mask & mask_valid
    
    # Compute 5min metrics
    mod_window_lib_5 = {k: v.loc[eval_mask_models] for k, v in models_lib_5min.items()}
    res_models_5 = vf.compute_all_metrics(truth_5min.loc[eval_mask_models], mod_window_lib_5)
    res_fc_05_5 = pd.DataFrame(vf.get_metrics(ev_lib_05_5min['Forecaster'].loc[eval_mask_fc], truth_5min.loc[eval_mask_fc]), index=['Forecaster_05'])
    res_fc_00_5 = pd.DataFrame(vf.get_metrics(ev_lib_00_5min['Forecaster'].loc[eval_mask_fc], truth_5min.loc[eval_mask_fc]), index=['Forecaster_00'])
    res_fc_first_00_5 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_00_5min['Forecaster'].loc[t_mask & mask_valid_first],
            truth_5min.loc[t_mask & mask_valid_first]
        ),
        index=['Forecaster_FirstHalf_00']
    )
    res_fc_first_05_5 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_05_5min['Forecaster'].loc[t_mask & mask_valid_first],
            truth_5min.loc[t_mask & mask_valid_first]
        ),
        index=['Forecaster_FirstHalf_05']
    )
    res_fc_second_00_5 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_00_5min['Forecaster'].loc[t_mask & mask_valid_second],
            truth_5min.loc[t_mask & mask_valid_second]
        ),
        index=['Forecaster_SecondHalf_00']
    )

    res_fc_second_05_5 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_05_5min['Forecaster'].loc[t_mask & mask_valid_second],
            truth_5min.loc[t_mask & mask_valid_second]
        ),
        index=['Forecaster_SecondHalf_05']
    )

    # Compute 15min metrics
    mod_window_lib_15 = {k: v.loc[eval_mask_models] for k, v in models_lib_15min.items()}
    res_models_15 = vf.compute_all_metrics(truth_15min.loc[eval_mask_models], mod_window_lib_15)
    res_fc_05_15 = pd.DataFrame(vf.get_metrics(ev_lib_05_15min['Forecaster'].loc[eval_mask_fc], truth_15min.loc[eval_mask_fc]), index=['Forecaster_05'])
    res_fc_00_15 = pd.DataFrame(vf.get_metrics(ev_lib_00_15min['Forecaster'].loc[eval_mask_fc], truth_15min.loc[eval_mask_fc]), index=['Forecaster_00'])
    res_fc_first_00_15 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_00_15min['Forecaster'].loc[t_mask & mask_valid_first],
            truth_15min.loc[t_mask & mask_valid_first]
        ),
        index=['Forecaster_FirstHalf_00']
    )
    res_fc_first_05_15 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_05_15min['Forecaster'].loc[t_mask & mask_valid_first],
            truth_15min.loc[t_mask & mask_valid_first]
        ),
        index=['Forecaster_FirstHalf_05']
    )
    res_fc_second_00_15 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_00_15min['Forecaster'].loc[t_mask & mask_valid_second],
            truth_15min.loc[t_mask & mask_valid_second]
        ),
        index=['Forecaster_SecondHalf_00']
    )

    res_fc_second_05_15 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_05_15min['Forecaster'].loc[t_mask & mask_valid_second],
            truth_15min.loc[t_mask & mask_valid_second]
        ),
        index=['Forecaster_SecondHalf_05']
    )

    # Store for plotting
    multi_period_results.append({
        'models_5min': res_models_5,
        'models_15min': res_models_15,
        'fc_05_5min': res_fc_05_5.iloc[0],
        'fc_05_15min': res_fc_05_15.iloc[0],
        'fc_00_5min': res_fc_00_5.iloc[0],
        'fc_00_15min': res_fc_00_15.iloc[0],
        'fc_first_00_5min': res_fc_first_00_5.iloc[0],
        'fc_first_00_15min': res_fc_first_00_15.iloc[0],
        'fc_second_00_5min': res_fc_second_00_5.iloc[0],
        'fc_second_00_15min': res_fc_second_00_15.iloc[0],
        'fc_first_05_5min': res_fc_first_05_5.iloc[0],
        'fc_first_05_15min': res_fc_first_05_15.iloc[0],
        'fc_second_05_5min': res_fc_second_05_5.iloc[0],
        'fc_second_05_15min': res_fc_second_05_15.iloc[0],
    })

# Compute Brier score
bs_ens = vf.compute_brier_score(prob_fog[eval_mask_fc], df_eval['obs_event_5min'][eval_mask_fc])
print(f"Ensemble Brier score (obs processed in 5 min): {bs_ens:.4f}")
bs_ens = vf.compute_brier_score(prob_fog[eval_mask_fc], df_eval['obs_event_15min'][eval_mask_fc])
print(f"Ensemble Brier score (obs processed in 15 min): {bs_ens:.4f}")

# Extract final period (Entire Cruise) for text output (5-minute basis)
final_res_5min = pd.concat([
    multi_period_results[-1]['models_5min'], 
    pd.DataFrame([multi_period_results[-1]['fc_05_5min']], index=['Forecaster_05']),
    pd.DataFrame([multi_period_results[-1]['fc_00_5min']], index=['Forecaster_00'])
])
print("\n--- EVALUATION SUMMARY (ENTIRE CRUISE - 5min Obs) ---")
print(final_res_5min.to_string(float_format="%.3f"))
print(f"\n{'Model':<25} | {'ETS':<10} | Contingency [Hits, Misses, FA, CN]")
print("-" * 65)
for name in final_res_5min.index:
    r = final_res_5min.loc[name]
    ets = vf.calculate_ets(r["Hits"], r["False alarms"], r["Misses"], r["Correct negatives"])
    print(f"{name:<25} | {ets:<10.4f} | [{int(r['Hits'])}, {int(r['Misses'])}, {int(r['False alarms'])}, {int(r['Correct negatives'])}]")

final_res_15min = pd.concat([
    multi_period_results[-1]['models_15min'], 
    pd.DataFrame([multi_period_results[-1]['fc_05_15min']], index=['Forecaster_05']),
    pd.DataFrame([multi_period_results[-1]['fc_00_15min']], index=['Forecaster_00'])
])
print("\n--- EVALUATION SUMMARY (ENTIRE CRUISE - 15min Obs) ---")
print(final_res_15min.to_string(float_format="%.3f"))
print(f"\n{'Model':<25} | {'ETS':<10} | Contingency [Hits, Misses, FA, CN]")
print("-" * 65)
for name in final_res_15min.index:
    r = final_res_15min.loc[name]
    ets = vf.calculate_ets(r["Hits"], r["False alarms"], r["Misses"], r["Correct negatives"])
    print(f"{name:<25} | {ets:<10.4f} | [{int(r['Hits'])}, {int(r['Misses'])}, {int(r['False alarms'])}, {int(r['Correct negatives'])}]")

#%% PERFORMANCE VISUAL ANALYSIS

# 1. 2x2 Multi-Period Performance Diagram
P_NAMES = [d[2] for d in DATES]
fig_perf, axs_perf = vf.plot_multi_period_performance(
    results_list=multi_period_results,
    period_names=P_NAMES,
    model_style_map=MODEL_STYLE,
    fc_style_map =FC_STYLES
)

# 2. Visual Summary Bar Chart (Using Entire Cruise Data)
fig1,fig2=vf.plot_metrics_summary(final_res_5min)
[fig.suptitle("Metrics summary with 5-min obs") for fig in [fig1,fig2]]
fig1,fig2=vf.plot_metrics_summary(final_res_15min)
[fig.suptitle("Metrics summary with 15-min obs") for fig in [fig1,fig2]]
# 3. Meteogram for a specific interesting window
plot_start, plot_end = '2025-08-20', '2025-08-30'
prob_df = vf.calculate_stacked_probabilities(df_eval)
# Note: this takes into account ONLY deterministic forecasts
vf.plot_ens_meteogram(
    prob_df=prob_df, 
    model_dict=model_data, 
    vis_obs=df_eval['obs_vis'], 
    start_date='2025-08-20', 
    end_date='2025-08-30'
)

# 4. Compute reliability diagram for ensemble probabilistic forecasts
# "When the model predicts a 40% chance of fog, how often does it actually fog (observations)?"
# Above the diagonal line: under confident. Else over confident. Perfect reliability on the diagonal.
vf.plot_reliability_diagram(prob_fog, df_eval['obs_event'], n_bins=20) # varies a lot with different bins!

# 5: Talagrand diagram for ensemble forecatss analysis (rank histogram)
# Rank 0 dominant: model is "too clear"
# Horizontal line: perfect ensemble spread (y = 1/(N+1) where N is the number of members and +1 is the observations)). Uniform dist.
# x-axis: "How many members predicted a lower visibility than the observations?", "and how often?" (y-axis)
vf.plot_talagrand_histogram(ens_aligned, df_eval['obs_vis'])

#%% GENERAL PLOTS

# 1. Component check
vf.plot_taf_components(df_eval)

# 2. Zoom into a specific event (e.g., a fog episode on Aug 25)
vf.plot_taf_window(df_eval, FOG_THRESH, '2025-09-03', '2025-09-16')

# 3. Visual summary with TAF uncertainty
vf.plot_vis_summary(df_eval, df_eval['obs_vis'], 
                    model_data["IFS"], model_data["lowLvlMean"], FOG_THRESH, 
                    start_date="2025-09-05", end_date="2025-09-05")

# 4. PDFs of observations
periods = [
    (('2025-08-12 00:00', '2025-08-16 12:00'), 'Period 1'),
    (('2025-08-16 13:00', '2025-09-03 00:00'), 'Period 2'),
    (('2025-09-03 01:00', '2025-09-16 00:00'), 'Period 3')
]
quant_vars = ["visas_1min", "visas_5min", "visas_10min", "visas_15min", "visas_median"]
fig,_=vf.plot_visibility_pdfs_cdfs(ds_obs, time_vec, periods, quant_vars, FOG_THRESH)
fig.suptitle("PDFs and CDFs for the observations processed using different quantiles",size=16,y=1.01)

# 5. Ensemble spaghetti
vf.plot_ensemble_spaghetti(ens_aligned, df_eval['obs_vis'], '2025-08-25', '2025-08-27',fog_thresh=FOG_THRESH)

# %%
