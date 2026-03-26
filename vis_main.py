#%% imports
from datetime import datetime
import re 
import vis_functions as vf
import importlib         # look for changes in vis_functions without
importlib.reload(vf)     # having to reload the kernel
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import seaborn as sns

#%% PREPROCESS DATASETS
flag = 0
debug = 0
# Rename the different visibilities from the original diagnostic datasets to "vis"
# to be used with the functions in the script.
if flag:
    # 1. Load the dataset
    ds_base = xr.open_dataset('/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v1.nc',decode_times=True)
    # 2. Drop the original 'vis'
    ds_dropped = ds_base.drop_vars('vis')
    # 3. Create the three datasets
    # We use .rename() to map the diagnostic variable to 'vis'
    ds_list = {
        'lowLvlMean': ds_dropped.rename({'vis_from_hydro_lowLvlMean': 'vis'}),
        'lowLvlSum':  ds_dropped.rename({'vis_from_hydro_lowLvlSum': 'vis'}),
        'lowestLvl':  ds_dropped.rename({'vis_from_hydro_lowestLvl': 'vis'})
    }
    for name, ds in ds_list.items():
        ds.to_netcdf(f"ifs_diagnostic_{name}.nc")


#%% SETTINGS AND PATHS
# Settings
FOG_THRESH = 1  # km  (0.8 km Cassel Aero threshold, 1 km WMO threshold)
FC_THRESH = 0.5 # forecast threshold to check low vis event (0: low vis assumed even if only predicted by TEMPO group. 0.5: only BASE group)
MODEL_24h = False  # Whether to evaluate the full 24h forecast or just the TAF validity times:
                  #   True: the model gets evaluated over 24h, while the forecaster only on its active time
                  #   False: both model and forecaster are evaluated only on the TAFs validity window. Better imho

# Period 1
START_DATE = '2025-08-12 00:00'
# END_DATE = '2025-08-16 12:00'
# Period 2
# START_DATE = '2025-08-16 13:00'
# END_DATE = '2025-09-03 00:00'
# Period 3:
# START_DATE = '2025-09-03 01:00'
END_DATE = '2025-09-16 00:00'
TIME_RES = 'h'  # Analysis resolution (minutes, hours..)

# File Paths
TAF_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/AO25_TAFs.xlsx'
OBS_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/obs_data/AO2025_MDF_20250812-20250915_hourly_quantiles.nc'
visas = "visas_median"

# Create model dictionary
MODEL_PATHS = {
    'IFS': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_oper_oden_20250811_20250915_day2_new_visibility_diagnostic_v1.nc',
    'lowLvlMean': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_diagnostic_lowLvlMean.nc',
    'lowLvlSum': '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_diagnostic_lowLvlSum.nc',
}

ENS_PATH = '/Users/lodo0477/Documents/PhD/Research/Oden/Visibility study/model_data/ifs_ens_oden_2025-08-11_2025-09-15_vis_day2.nc'

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
        # print(f"Processed {name} with values ranging {model_data[name].min():.2f} to {model_data[name].max():.2f}")

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
    prob_fog = (ens_aligned < FOG_THRESH).mean(dim='number').to_series().reindex(time_vec)
    
    # Create binary series: If prob > X%, we set vis to 0.0 (Fog), else 10.0 (Clear)
    model_data['Ens_P20'] = pd.Series(np.where(prob_fog > 0.20, 0.0, 10.0), index=time_vec)
    model_data['Ens_P30'] = pd.Series(np.where(prob_fog > 0.30, 0.0, 10.0), index=time_vec)

# 2. Load and Process TAFs
taf_table = pd.read_excel(TAF_PATH, header=1, sheet_name='Sheet1').reset_index(drop=True)
taf_table = taf_table[taf_table['TAF Oden'].notna()]  # ensure no empty rows
taf_table['Date'] = pd.to_datetime(taf_table['Date'])
start_day_table = pd.to_datetime(START_DATE).normalize()
end_day_table = pd.to_datetime(END_DATE).normalize()
mask = (taf_table['Date'] >= start_day_table) & (taf_table['Date'] <= end_day_table)
taf_table = taf_table.loc[mask].reset_index(drop=True)

# START_DATE must match furst row of the Excel
df_eval = vf.df_TAF_gen(taf_table, time_vec, pd.Timestamp(START_DATE))
df_eval = vf.calculate_scenarios(df_eval)
df_eval = vf.assign_event_probabilities(df_eval, v_thresh=FOG_THRESH)

# 3. Load and Align Observations
ds_obs = xr.open_dataset(OBS_PATH, decode_timedelta=True)

# Apply smoothing and align to time_vec
vis_obs_series = ds_obs[visas].to_series() * 1e-3
vis_obs_series = np.clip(vis_obs_series, 0, 10).reindex(time_vec)

# Check number of visibility observations with vis < 800m
cp=0
cm=0
for i in vis_obs_series:
    if i <= FOG_THRESH:
        cm +=1
    elif i > FOG_THRESH:
        cp +=1

print(f"Count of points with vis > {FOG_THRESH*1e3}m: {cp}")
print(f"Count of points with vis <= {FOG_THRESH*1e3}m: {cm}  [{cm*100/(cm+cp):.1f}% of the total]")


# Check data overlap
valid_times = df_eval['is_valid'].sum()
print(f"Total time units with valid TAFs: {valid_times}")
if valid_times == 0:
    print("WARNING: No TAFs were mapped to the time vector. Check START_DATE/Index alignment.")

# Add to evaluation dataframe
df_eval['obs_vis'] = vis_obs_series
df_eval['obs_event'] = (df_eval['obs_vis'] < FOG_THRESH).astype(float)

# Debugging time 
if debug:
    print("### DEBUGGING time_vec ###")
    for name, series in model_data.items():
        print(name, series.index.min(), series.index.max(), len(series))
    print("time_vec:", time_vec.min(), time_vec.max(), len(time_vec))
    print("Obs:", vis_obs_series.index.min(), vis_obs_series.index.max(), len(vis_obs_series))
    print("TAFs:", df_eval.index.min(), df_eval.index.max(), len(df_eval))

#%% MODEL EVALUATION

# 1. Get the raw events (Initial 24h baseline for all)
truth_full, event_lib = vf.get_evaluation_library(df_eval, model_data, df_eval['obs_vis'],
                                                   p_thresh=FC_THRESH, fog_thresh=FOG_THRESH)

# Define the Forecaster's active window mask
mask = (df_eval['is_valid'] == True)

if not MODEL_24h:
    # --- SCENARIO: SYNCHRONIZED DAYTIME ONLY ---
    # Everything (Truth, Forecaster, Models) is restricted to 07:00-15:00
    truth = truth_full.loc[mask]
    
    # Apply mask to the event library (Handling both DataFrame and Dict)
    if isinstance(event_lib, pd.DataFrame):
        eval_lib = event_lib.loc[mask]
    else:
        eval_lib = {k: v.loc[mask] for k, v in event_lib.items()}

    results_df = vf.compute_all_metrics(truth, eval_lib)
    print(f"--- SYNCHRONIZED EVALUATION (TAF VALIDITY TIME ONLY) ---")

else:
    # --- SCENARIO: HYBRID OPERATIONAL ---
    # Forecaster is evaluated on 8h window | Models are evaluated on 24h window
    
    # A. Forecaster metrics (8h window)
    truth_forecaster = truth_full.loc[mask]
    forecaster_series = event_lib['Forecaster'].loc[mask]
    forecaster_metrics = vf.get_metrics(forecaster_series, truth_forecaster)
    forecaster_df = pd.DataFrame(forecaster_metrics, index=['Forecaster'])
    
    # B. Model metrics (Full 24h window)
    if isinstance(event_lib, pd.DataFrame):
        models_only_lib = event_lib.drop(columns=['Forecaster'])
    else:
        models_only_lib = {k: v for k, v in event_lib.items() if k != 'Forecaster'}
    
    model_results_df = vf.compute_all_metrics(truth_full, models_only_lib)
    
    # C. Combine
    results_df = pd.concat([model_results_df, forecaster_df])
    print(f"--- HYBRID EVALUATION (Models: 24h | Forecaster: 8h) ---")

if debug:
    print("---- SANITY CHECK ----")
    print("Truth sum:", truth.sum())

    for name, s in eval_lib.items():
        print(name, "sum:", s.sum(), "non-NaN:", s.notna().sum())
    print("### DEBUG overlap length ###")
    print("Truth events:", truth.sum())
    for name, series in eval_lib.items():
        aligned = truth.align(series, join='inner')
        print(name, "overlap length:", len(aligned[0].dropna()))

# 3. View Results
all_rows = sorted([r for r in results_df.index if r != 'Forecaster'])
custom_order = ['Forecaster'] + all_rows  # Forecaster on top when printing
results_df = results_df.reindex(custom_order)
print(results_df.to_string(float_format="%.3f"))

# vf.plot_ensemble_spaghetti(ens_aligned, df_eval['obs_vis'], '2025-08-25', '2025-08-27')

# 4. Brier Score calculation
bs_ens = vf.compute_brier_score(prob_fog, df_eval['obs_event'])
print(f"Ensemble Brier Score: {bs_ens:.4f}")

#%% Check fog events in [START_DATE , END_DATE]

fig, ax = plt.subplots(figsize=(16, 8))

# Plot observations and models
ax.plot(df_eval.index, df_eval['obs_vis'], label='Observed Vis (km)', color='black')
for model_name, model_series in model_data.items():
    if model_name.startswith("Ens_"):
        ax.plot(model_series.index, model_series, label=f'{model_name}', lw=0.7)

# Fog threshold
ax.axhline(FOG_THRESH, color='red', linestyle='--', label='Fog Threshold')

# Highlight TAF valid times
ax.fill_between(
    df_eval.index,
    0,
    df_eval['obs_vis'].max(),
    where=~df_eval['is_valid'],
    color='lightgray',
    alpha=0.5,
    label='No TAF'
)

# Highlight fog events (ONLY within mask)
fog_masked = (df_eval['obs_vis'] < FOG_THRESH) & (df_eval['is_valid'])

ax.scatter(
    df_eval.index[fog_masked],
    df_eval['obs_vis'][fog_masked],
    color='red',
    s=60,
    label='Fog events (used in metrics)',
    zorder=5
)

ax.set_title("Visibility with Fog Events and TAF Validity Window")
ax.set_ylabel("Visibility (km)")
ax.set_ylim(-0.1,2)
ax.legend(prop=dict(size=7),ncol=2,loc="upper left")
ax.grid(alpha=0.3)

if debug:
    model = event_lib['IFS']

    # Find where obs = event
    obs_events = truth[truth == True]

    print("Checking each observed fog event:\n")

    for t in obs_events.index:
        print(f"Time: {t}")
        print("Obs:", truth.loc[t])
        
        if t in model.index:
            print("Model:", model.loc[t])
        else:
            print("Model: MISSING TIMESTAMP")
        
        print("-" * 30)

#%% PERFORMANCE VISUAL ANALYSIS

# 1. Performance Diagram (Generalised)
# Extract POD/FAR from the results_df
# Below B=1: underforecasting, else overforecasting.
# High POD = good; High Success ratio = good; Points on top right corner = very good.
vf.plot_performance_diagram(
    pods=results_df['POD'], 
    fars=results_df['FAR'], 
    labels=results_df.index
)

# 2. Visual Summary Bar Chart
# vf.plot_metrics_summary(results_df)

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
vf.plot_taf_window(df_eval, FOG_THRESH, '2025-08-24', '2025-08-24')
#%%
# 3. Visual summary with TAF uncertainty
vf.plot_vis_summary(df_eval, df_eval['obs_vis'], model_data["IFS"], model_data["lowLvlMean"], FOG_THRESH, start_date="2025-08-30", end_date="2025-08-30")


##########################################################################
#%%#######################################################################

##########################################################################

#%% WORKSHOP IN LEEDS

#%% Plot PDFs of observations

periods = [
    (('2025-08-12 00:00', '2025-08-16 12:00'), 'Period 1', 'k'),
    (('2025-08-16 13:00', '2025-09-03 00:00'), 'Period 2', 'r'),
    (('2025-09-03 01:00', '2025-09-16 00:00'), 'Period 3', 'b')
]

quant_vars = ["visas_1min", "visas_5min", "visas_10min", "visas_15min", "visas_median"]

vf.plot_visibility_pdfs_cdfs(ds_obs, df_eval, time_vec, periods, quant_vars, FOG_THRESH)

# %%
