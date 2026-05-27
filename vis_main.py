# Ref: L. Donati
# lorenzo.luca.donati@misu.su.se

# Scripts for "Paper title"
# by Authors...

### NOTE:
#   In case one wants to process data using 5- and 15-minute processed observations,
#   just pull the following commit from the repository:  3d4e1f5  ("cosmetic changes")

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
HIGHER_THAN_FOG_THRESH = True  # if True, looks at windows of opportnity (high visibility). If False, looks at low vis. events
MODEL_24h = False # Whether to evaluate the full 24h forecast or just the TAF validity times:
                  #   True: the model gets evaluated over 24h, while the forecaster only on its active time
                  #   False: both model and forecaster are evaluated only on the TAFs validity window. Better imho

# Cruise period and resolution
START_DATE = '2025-08-12 00:00'
END_DATE = '2025-09-16 00:00'
TIME_RES = 'h'  # Analysis resolution (minutes, hours..) 

# File Paths (observation ALREADY preprocessed with hourly median/quantiles and divided in quantiles, from data repository)
TAF_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/AO25_TAFs.xlsx'
OBS_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/obs_data/AO2025_MDF_20250812-20250915_hourly_quantiles_10minmin.nc'
visas = "visas_10min"

# Create model dictionary
MODEL_PATHS = {
    'IFS': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v1.nc',
    'LowLvlMean': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_diagnostic_lowLvlMean.nc',
}

PERS_PATH = "/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/AO2025_20250812-20250915_persistence_forecast_v2.nc"

ENS_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_ens_oden_2025-08-11_2025-09-15_vis_day2.nc'

# Visual configuration
MODEL_STYLE = {
    'IFS': 'blue',
    'LowLvlMean': 'tab:blue',
    'Ens_P30': 'darkgreen',
    # 'Ens_AllCases': 'red',  
    # 'Ens_OneCase': 'purple',
    # 'Ens_P20': 'brown',
    'Ens_P50': 'lightgreen',
    'Persist_10min': 'black',
}
FC_STYLES = {
    'base': {'color': 'red', 
             'marker': 'D', 
             'label': 'TAF (Base)'},
    'conservative': {'color': 'red', 
                     'marker': 'X', 
                     'label': 'TAF (All)'},
    'first_half': {
        'color': 'purple',
        'marker': '^',
        'label': 'TAF (First Half)'
    },
    'second_half': {
        'color': 'magenta',
        'marker': 'v',
        'label': 'TAF (Second Half)'
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
    model_data['Ens_P50'] = ens_aligned.median(dim='number').to_series().reindex(time_vec)
    
    # 3. Extract "Worst Case" (Minimum member)
    # model_data['Ens_OneCase'] = ens_aligned.min(dim='number').to_series().reindex(time_vec)
    # 4. Probabilistic Triggers
    # Calculate fraction of members
    if HIGHER_THAN_FOG_THRESH:
        prob_fog = (ens_aligned > FOG_THRESH).mean(dim='number').to_series().reindex(time_vec)
    else:
        prob_fog = (ens_aligned <= FOG_THRESH).mean(dim='number').to_series().reindex(time_vec)
    
    # Create binary series: If prob > X%, we set vis to 0.0 (Fog), else 10.0 (Clear)
    event_value = 10.0 if HIGHER_THAN_FOG_THRESH else 0.0
    non_event_value = 0.0 if HIGHER_THAN_FOG_THRESH else 10.0

    # model_data['Ens_P20'] = pd.Series(np.where(prob_fog >= 0.20, event_value, non_event_value), index=time_vec)
    model_data['Ens_P30'] = pd.Series(np.where(prob_fog >= 0.30, event_value, non_event_value), index=time_vec)
    # model_data['Ens_AllCases'] = pd.Series(np.where(prob_fog > 0.98, event_value, non_event_value), index=time_vec)

# Process "persistent" forecaster data
pers_ds = xr.open_dataset(PERS_PATH, decode_timedelta=True)
# model_data["pers_ds_median"] = np.clip(pers_ds.persistence_median.to_series() * 1e-3, 0, 10).reindex(time_vec)
# model_data["pers_ds_5min"] =np.clip(pers_ds.persistence_5min.to_series() * 1e-3, 0, 10).reindex(time_vec)
model_data["Persist_10min"] =np.clip(pers_ds.persistence10m_minimum.to_series() * 1e-3, 0, 10).reindex(time_vec)

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

# Apply smoothing and align to time_vec
vis_obs = np.clip(ds_obs[visas].to_series() * 1e-3, 0, 10).reindex(time_vec)

# Add to evaluation dataframe 
df_eval['obs_vis'] = vis_obs

if HIGHER_THAN_FOG_THRESH:
    df_eval['obs_event'] = (df_eval['obs_vis'] > FOG_THRESH).astype(float)
else:
    df_eval['obs_event'] = (df_eval['obs_vis'] <= FOG_THRESH).astype(float)

# Check number of visibility observations with vis < 800m
if debug:
    cp,cm=0,0
    for i in vis_obs:
        if i <= FOG_THRESH:
            cm +=1
        elif i > FOG_THRESH:
            cp +=1
    print(f"Count of points with vis > {FOG_THRESH*1e3}m: {cp}")
    print(f"Count of points with vis <= {FOG_THRESH*1e3}m: {cm}  [{cm*100/(cm+cp):.1f}% of the total]")

# Check data overlap
if debug: 
    valid_times = df_eval['is_valid'].sum()
    print(f"Total time units with valid TAFs: {valid_times}")
    if valid_times == 0:
        print("WARNING: No TAFs were mapped to the time vector. Check START_DATE/Index alignment.")
    if HIGHER_THAN_FOG_THRESH:
        mask = (vis_obs > FOG_THRESH) & (df_eval['is_valid'] == True)
    else:   
        mask = (vis_obs <= FOG_THRESH) & (df_eval['is_valid'] == True)
    vis_count = mask.sum()
    print(f"Low visibility events in TAF validity window: {vis_count}")  

# Debugging time 
if debug:
    print("### DEBUGGING time_vec ###")
    for name, series in model_data.items():
        print(name, series.index.min(), series.index.max(), len(series))
    print("time_vec:", time_vec.min(), time_vec.max(), len(time_vec))
    print("Obs:", vis_obs.index.min(), vis_obs.index.max(), len(vis_obs))
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
truth, ev_lib_05 = vf.get_evaluation_library(df_eval, model_data, df_eval['obs_vis'], p_thresh=0.5, fog_thresh=FOG_THRESH, higher_than_fog_thresh=HIGHER_THAN_FOG_THRESH)
_, ev_lib_00 = vf.get_evaluation_library(df_eval, model_data, df_eval['obs_vis'], p_thresh=0.0, fog_thresh=FOG_THRESH, higher_than_fog_thresh=HIGHER_THAN_FOG_THRESH)

# Isolate models 
models_lib = {k: v for k, v in ev_lib_05.items() if k != 'Forecaster'}

multi_period_results = []
mask_valid = df_eval['is_valid'] == True
mask_valid_first = df_eval['is_valid_first_half'] == True
mask_valid_second = df_eval['is_valid_second_half'] == True

for start_t, end_t, p_name in DATES:
    t_mask = (df_eval.index >= start_t) & (df_eval.index <= end_t)
    
    # Compute only in TAF validity times if MODEL_24h=False
    eval_mask_models = t_mask if MODEL_24h else t_mask & mask_valid
    eval_mask_fc = t_mask & mask_valid
    
    # Compute metrics
    mod_window_lib = {k: v.loc[eval_mask_models] for k, v in models_lib.items()}
    res_models = vf.compute_all_metrics(truth.loc[eval_mask_models], mod_window_lib)
    res_fc_05 = pd.DataFrame(vf.get_metrics(ev_lib_05['Forecaster'].loc[eval_mask_fc], truth.loc[eval_mask_fc]), index=['Forecaster_05'])
    res_fc_00 = pd.DataFrame(vf.get_metrics(ev_lib_00['Forecaster'].loc[eval_mask_fc], truth.loc[eval_mask_fc]), index=['Forecaster_00'])
    res_fc_first_00 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_00['Forecaster'].loc[t_mask & mask_valid_first],
            truth.loc[t_mask & mask_valid_first]
        ),
        index=['Forecaster_FirstHalf_00']
    )
    res_fc_first_05 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_05['Forecaster'].loc[t_mask & mask_valid_first],
            truth.loc[t_mask & mask_valid_first]
        ),
        index=['Forecaster_FirstHalf_05']
    )
    res_fc_second_00 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_00['Forecaster'].loc[t_mask & mask_valid_second],
            truth.loc[t_mask & mask_valid_second]
        ),
        index=['Forecaster_SecondHalf_00']
    )

    res_fc_second_05 = pd.DataFrame(
        vf.get_metrics(
            ev_lib_05['Forecaster'].loc[t_mask & mask_valid_second],
            truth.loc[t_mask & mask_valid_second]
        ),
        index=['Forecaster_SecondHalf_05']
    )

    # Store for plotting
    multi_period_results.append({
        'models': res_models,
        'fc_05': res_fc_05.iloc[0],
        'fc_00': res_fc_00.iloc[0],
        'fc_first_00': res_fc_first_00.iloc[0],
        'fc_second_00': res_fc_second_00.iloc[0],
        'fc_first_05': res_fc_first_05.iloc[0],
        'fc_second_05': res_fc_second_05.iloc[0],
    })

# Compute Brier score
bs_ens = vf.compute_brier_score(prob_fog[eval_mask_fc], df_eval['obs_event'][eval_mask_fc])
print(f"Ensemble Brier score: {bs_ens:.4f}")

# Extract final period (Entire Cruise) for text output (5-minute basis)
final_res = pd.concat([
    multi_period_results[-1]['models'], 
    pd.DataFrame([multi_period_results[-1]['fc_05']], index=['Forecaster_05']),
    pd.DataFrame([multi_period_results[-1]['fc_00']], index=['Forecaster_00'])
])
print("\n--- EVALUATION SUMMARY (ENTIRE CRUISE) ---")
print(final_res.to_string(float_format="%.3f"))
print(f"\n{'Model':<25} | {'ETS':<10} | Contingency [Hits, Misses, FA, CN]")
print("-" * 65)
for name in final_res.index:
    r = final_res.loc[name]
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
fig1,fig2=vf.plot_metrics_summary(final_res)
[fig.suptitle("Metrics summary") for fig in [fig1,fig2]]

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
                    model_data["IFS"], model_data["LowLvlMean"], FOG_THRESH, 
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
