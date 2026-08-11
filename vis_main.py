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
    # ('2025-08-16 13:00', '2025-09-16 00:00', 'Period 2'),
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
        'path': f'{RUN_DIR}/model_data/ifs_diagnostic_lowLvlMean.nc',
        'type': 'det', 'var': 'vis', 'color': 'tab:blue'
    },
    'Persist_10min': {
        'path': f'{RUN_DIR}/model_data/AO2025_20250812-20250915_persistence_forecast_v2.nc',
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
    period_names=[d[2] for d in PERIODS],
    model_style_map=MODEL_STYLE,
    all_periods=True,
    insets=False
)

# 2. Metrics summary (example for entire period [3], considering both halves )
fig1, fig2 = vf.plot_metrics_summary(matrix_results["high"][2]["splits"]["Full"])
fig1.suptitle("Windows of opportunity"); fig2.suptitle("Windows of opportunity")
fig1, fig2 = vf.plot_metrics_summary(matrix_results["low"][2]["splits"]["Full"])
fig1.suptitle("Low visibility events"); fig2.suptitle("Low visibility events")

# 3. Ensemble diagnostics
vf.plot_reliability_diagram(prob_fog, taf_eval['obs_event'], n_bins=20)
vf.plot_talagrand_histogram(ens_aligned, taf_eval['obs_vis'])

#%%

# 3. Flexible Visibility Summary Meteogram 
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


# %% quick plot for period 1 and entire period in low visiobity

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

def plot_two_period_performance_comparison(results_low, model_style_map):  
    """
    Plots a simplified 1x2 horizontal performance matrix comparing the 
    'Entire Cruise' and 'Period 1' metrics side-by-side.
    """
    # Create markers for different data halves
    my_marker_1 = vf.get_text_marker("A")
    my_marker_2 = vf.get_text_marker("B")

    # 1. Setup Data Meshgrids for Background CSI
    x = np.linspace(0.001, 1, 100)
    y = np.linspace(0.001, 1, 100)
    SR_grid, POD_grid = np.meshgrid(x, y)
    CSI = 1 / (1/SR_grid + 1/POD_grid - 1)
    grid_data = (SR_grid, POD_grid, CSI)

    # Setup figure with 2 columns for plots and 1 narrow column for the shared colorbar
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.5), 
                            gridspec_kw={'width_ratios': [1, 1, 0.05]})
    
    plot_axs = axs[:2]   # The two main comparison axes
    cbar_ax  = axs[2]    # Shared colorbar axis

    # Inset configurations mapped to our 2 chosen view panels
    # Panel 0: Entire Cruise (Low-Vis), Panel 1: Period 1 (Low-Vis)
    inset_configs = {
        0: {'bounds': None,                     'xlim': [0, 0],       'ylim': [0, 0]},
        1: {'bounds': [0.08, 0.15, 0.30, 0.70], 'xlim': [0.80, 1.02], 'ylim': [0.23, 0.75]}
    }

    # Extract exactly what we need: 'Entire Cruise' is at index -1, 'Period 1' is at index 0
    selected_low_data = [results_low[-1], results_low[0]]
    display_names     = ['Entire Cruise (Low visibility events)', 'Period 1 (Low visibility events)']
    panel_labels      = ['a', 'b']

    contour_mappable = None

    for i in range(2):
        ax = plot_axs[i]
        splits = selected_low_data[i]['splits']

        # Style Panel 0 (Entire Cruise Summary) with a more prominent border
        if i == 0:
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)       
                spine.set_color('black')       
                spine.set_zorder(10)           
        
        # Draw the performance diagram background contours
        contour_mappable = vf.draw_perf_background(ax, grid_data, line_w=0.8, line_alpha=0.5, contour_alpha=0.2, show_text=True)

        target_axs = [ax]
        vf.draw_hatching(ax, alpha=0.7, borders=True)
        cfg = inset_configs[i]
        
        # Build zoom-in insets where configured
        if cfg["bounds"] is not None:
            ax_ins = ax.inset_axes(cfg['bounds'])
            ax_ins.set_xlim(cfg['xlim'])
            ax_ins.set_ylim(cfg['ylim'])
            ax_ins.set_aspect('auto')
            
            ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4))
                
            ax_ins.tick_params(axis='both', which='major', labelsize=8)
            vf.draw_hatching(ax_ins)
            vf.draw_perf_background(ax_ins, grid_data, line_w=0.6, line_alpha=0.3, contour_alpha=0.15, show_text=False)
            target_axs.append(ax_ins)
            ax.indicate_inset_zoom(ax_ins, edgecolor="grey", alpha=1, lw=0.7)

        # --- UNIFORM TRAJECTORY PLOTTING BLOCK ---
        if all(k in splits for k in ['Full', 'First_Half', 'Second_Half']):
            df_full = splits['Full']
            df_1st  = splits['First_Half']
            df_2nd  = splits['Second_Half']

            for model_name, color in model_style_map.items():
                if model_name in df_full.index and model_name in df_1st.index and model_name in df_2nd.index:
                    row_full = df_full.loc[model_name]
                    row_1st  = df_1st.loc[model_name]
                    row_2nd  = df_2nd.loc[model_name]

                    pt_full = (1 - row_full['FAR'], row_full['POD'])
                    pt_1st  = (1 - row_1st['FAR'],  row_1st['POD'])
                    pt_2nd  = (1 - row_2nd['FAR'],  row_2nd['POD'])

                    mrkr = "*" if model_name == "Persist_10min" else "o"
                    sz = 180 if model_name == "Persist_10min" else 120

                    for t_ax in target_axs:
                        # Split half A
                        t_ax.scatter(*pt_1st, s=sz*2.2, c=color, marker=my_marker_1, edgecolor=color, zorder=4, alpha=0.7)
                        t_ax.scatter(*pt_1st, s=sz*0.05, c=color, marker="o", edgecolor=color, zorder=4, alpha=0.35)
                        
                        # Split half B
                        t_ax.scatter(*pt_2nd, s=sz*2.2, c=color, marker=my_marker_2, edgecolor=color, zorder=4, alpha=0.7)
                        t_ax.scatter(*pt_2nd, s=sz*0.05, c=color, marker="o", edgecolor=color, zorder=4, alpha=0.35)
                        
                        # Total integrated metric window
                        t_ax.scatter(*pt_full, s=sz, c=color, marker=mrkr, edgecolor='black', zorder=5, alpha=0.8)
                        t_ax.plot([pt_1st[0], pt_full[0], pt_2nd[0]], [pt_1st[1], pt_full[1], pt_2nd[1]], 
                                  color=color, linestyle='-', linewidth=1.2, alpha=0.4, zorder=3)

        # Labels, layout limits and panel configuration
        ax.set_title(display_names[i], pad=20, fontweight='bold' if i == 0 else 'normal')
        
        p_letter = panel_labels[i]
        ax.text(0.05, 0.95, f"{p_letter})", transform=ax.transAxes, fontsize=14, fontweight='bold', 
                va='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=1))
        
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=13)

    # Set common Y-Label on leftmost plot only
    plot_axs[0].set_ylabel('Probability of Detection (POD)', fontsize=13)

    # 2. Rebuild Complete Legend
    legend_elements = []
    for l, c in model_style_map.items():
        mrkr = '*' if l == "Persist_10min" else 'o'
        legend_elements.append(Line2D([0], [0], color="none", markerfacecolor=c, label=l, marker=mrkr, markeredgecolor='black', markersize=10))
    
    legend_elements.extend([
        Line2D([0], [0], color='none', marker='o', markerfacecolor='white', markeredgecolor='black', label='Full Window Metric', markersize=10),
        Line2D([0], [0], color='none', marker=my_marker_1, markerfacecolor='white', markeredgecolor='black', label='First Half Window (A)', markersize=11),
        Line2D([0], [0], color='none', marker=my_marker_2, markerfacecolor='white', markeredgecolor='black', label='Second Half Window (B)', markersize=11),
    ])
    plot_axs[0].legend(handles=legend_elements, frameon=True, loc='lower right', prop={'size': 8}, ncols=4)

    # 3. Add single Colorbar
    fig.colorbar(contour_mappable, cax=cbar_ax).set_label('CSI', fontsize=10)
    
    plt.tight_layout()
    return fig, plot_axs


plot_two_period_performance_comparison(results_low=matrix_results['low'],model_style_map=MODEL_STYLE)


# ------------------------------------ #


def plot_low_high_vis_performance_comparison(results_low, results_high, model_style_map):  
    """
    Plots a simplified 1x2 horizontal performance matrix comparing the 
    'Entire Cruise' for both low and high visibility.
    """
    # Create markers for different data halves
    my_marker_1 = vf.get_text_marker("A")
    my_marker_2 = vf.get_text_marker("B")

    # 1. Setup Data Meshgrids for Background CSI
    x = np.linspace(0.001, 1, 100)
    y = np.linspace(0.001, 1, 100)
    SR_grid, POD_grid = np.meshgrid(x, y)
    CSI = 1 / (1/SR_grid + 1/POD_grid - 1)
    grid_data = (SR_grid, POD_grid, CSI)

    # Setup figure with 2 columns for plots and 1 narrow column for the shared colorbar
    fig, axs = plt.subplots(1, 3, figsize=(15, 6.5), 
                            gridspec_kw={'width_ratios': [1, 1, 0.05]})
    
    plot_axs = axs[:2]   # The two main comparison axes
    cbar_ax  = axs[2]    # Shared colorbar axis

    # Inset configurations mapped to our 2 chosen view panels
    # Panel 0: Entire Cruise (Low-Vis), Panel 1: Period 1 (Low-Vis)
    inset_configs = {
        0: {'bounds': None,                     'xlim': [0, 0],       'ylim': [0, 0]},
        1: {'bounds': [0.08, 0.18, 0.50, 0.56], 'xlim': [0.79, 1.0], 'ylim': [0.837, 1.01]}
    }

    # Extract exactly what we need: 'Entire Cruise' is at index -1, 'Period 1' is at index 0
    selected_low_data = [results_low[-1], results_high[-1]]
    display_names     = ['Low visibility events', 'High visibility events']
    panel_labels      = ['a', 'b']

    contour_mappable = None

    for i in range(2):
        ax = plot_axs[i]
        splits = selected_low_data[i]['splits']

        # Style Panel with a more prominent border
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)       
            spine.set_color('black')       
            spine.set_zorder(10)           
        
        # Draw the performance diagram background contours
        contour_mappable = vf.draw_perf_background(ax, grid_data, line_w=0.8, line_alpha=0.5, contour_alpha=0.2, show_text=True)

        target_axs = [ax]
        vf.draw_hatching(ax, alpha=0.7, borders=True)
        cfg = inset_configs[i]
        
        # Build zoom-in insets where configured
        if cfg["bounds"] is not None:
            ax_ins = ax.inset_axes(cfg['bounds'])
            ax_ins.set_xlim(cfg['xlim'])
            ax_ins.set_ylim(cfg['ylim'])
            ax_ins.set_aspect('auto')
            
            ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4))
                
            ax_ins.tick_params(axis='both', which='major', labelsize=8)
            vf.draw_hatching(ax_ins)
            vf.draw_perf_background(ax_ins, grid_data, line_w=0.6, line_alpha=0.3, contour_alpha=0.15, show_text=False)
            target_axs.append(ax_ins)
            ax.indicate_inset_zoom(ax_ins, edgecolor="grey", alpha=1, lw=0.7)

        # --- UNIFORM TRAJECTORY PLOTTING BLOCK ---
        if all(k in splits for k in ['Full', 'First_Half', 'Second_Half']):
            df_full = splits['Full']
            df_1st  = splits['First_Half']
            df_2nd  = splits['Second_Half']

            for model_name, color in model_style_map.items():
                if model_name in df_full.index and model_name in df_1st.index and model_name in df_2nd.index:
                    row_full = df_full.loc[model_name]
                    row_1st  = df_1st.loc[model_name]
                    row_2nd  = df_2nd.loc[model_name]

                    pt_full = (1 - row_full['FAR'], row_full['POD'])
                    pt_1st  = (1 - row_1st['FAR'],  row_1st['POD'])
                    pt_2nd  = (1 - row_2nd['FAR'],  row_2nd['POD'])

                    mrkr = "*" if model_name == "Persist_10min" else "o"
                    sz = 180 if model_name == "Persist_10min" else 120

                    for t_ax in target_axs:
                        # Split half A
                        t_ax.scatter(*pt_1st, s=sz*2.2, c=color, marker=my_marker_1, edgecolor=color, zorder=4, alpha=0.7)
                        t_ax.scatter(*pt_1st, s=sz*0.05, c=color, marker="o", edgecolor=color, zorder=4, alpha=0.35)
                        
                        # Split half B
                        t_ax.scatter(*pt_2nd, s=sz*2.2, c=color, marker=my_marker_2, edgecolor=color, zorder=4, alpha=0.7)
                        t_ax.scatter(*pt_2nd, s=sz*0.05, c=color, marker="o", edgecolor=color, zorder=4, alpha=0.35)
                        
                        # Total integrated metric window
                        t_ax.scatter(*pt_full, s=sz, c=color, marker=mrkr, edgecolor='black', zorder=5, alpha=0.8)
                        t_ax.plot([pt_1st[0], pt_full[0], pt_2nd[0]], [pt_1st[1], pt_full[1], pt_2nd[1]], 
                                  color=color, linestyle='-', linewidth=1.2, alpha=0.4, zorder=3)

        # Labels, layout limits and panel configuration
        ax.set_title(display_names[i], pad=20, fontweight='bold')
        
        p_letter = panel_labels[i]
        ax.text(0.05, 0.95, f"{p_letter})", transform=ax.transAxes, fontsize=14, fontweight='bold', 
                va='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=1))
        
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=13)

    # Set common Y-Label on leftmost plot only
    plot_axs[0].set_ylabel('Probability of Detection (POD)', fontsize=13)

    # 2. Rebuild Complete Legend
    legend_elements = []
    for l, c in model_style_map.items():
        mrkr = '*' if l == "Persist_10min" else 'o'
        legend_elements.append(Line2D([0], [0], color="none", markerfacecolor=c, label=l, marker=mrkr, markeredgecolor='black', markersize=10))
    
    legend_elements.extend([
        Line2D([0], [0], color='none', marker='o', markerfacecolor='white', markeredgecolor='black', label='Full Window Metric', markersize=10),
        Line2D([0], [0], color='none', marker=my_marker_1, markerfacecolor='white', markeredgecolor='black', label='First Half Window (A)', markersize=11),
        Line2D([0], [0], color='none', marker=my_marker_2, markerfacecolor='white', markeredgecolor='black', label='Second Half Window (B)', markersize=11),
    ])
    plot_axs[0].legend(handles=legend_elements, frameon=True, loc='lower right', prop={'size': 8}, ncols=4)

    # 3. Add single Colorbar
    fig.colorbar(contour_mappable, cax=cbar_ax).set_label('CSI', fontsize=10)
    
    plt.tight_layout()
    return fig, plot_axs


plot_low_high_vis_performance_comparison(results_low=matrix_results['low'],results_high=matrix_results['high'],model_style_map=MODEL_STYLE)





#%% SEEPS score   

# MULTI-CATEGORICAL EVALUATION (SEEPS)
# ---------------------------------------------------------
# Compute Generalized SEEPS Error Score across physical visibility profiles.
# Restructured for 2 Thresholds / 3 Categories (analogous to the Rodwell paper).
# ---------------------------------------------------------
def compute_dynamic_seeps_thresholds(obs_data, fixed_thresh1=0.8, clipping_limit=10.0, safe_margin=0.5, max_entropy=True):
    """
    Computes SEEPS thresholds following the original Rodwell et al. (2010) logic,
    adapted for visibility data with an upper clipping ceiling.
    
    Parameters:
    -----------
    obs_data : array-like
        Historical or climatological observation values of visibility (km).
    fixed_thresh1 : float
        The rigid lower threshold (e.g., 0.8 km for dense fog).
    clipping_ceiling : float
        The maximum value where visibility data is clipped (e.g., 10.0 km).
    safe_margin : float
        The distance below the clipping limit to force-cap the second threshold 
        if it hits the data plateau (prevents category degeneracy).
        
    Returns:
    --------
    thresholds : list
        A list containing [fixed_thresh1, calculated_thresh2]
    """
    obs = np.asarray(obs_data)
    
    # Step 1: Compute p1 (Climatological probability of the fixed category)
    p1 = np.mean(obs <= fixed_thresh1)
    
    # Step 2: Compute target quantile based on max entropy approach 
    # (c = 1): Clear and Moderate split the remaining space 50/50
    if max_entropy:
        target_quantile = (1.0 + p1) / 2.0
    else: # assume high visbility events twice more likely than medium visiblity events
        target_quantile = (1.0 + 2.0 * p1) / 3.0
    
    # Step 3: Compute the raw physical threshold from the empirical distribution
    calculated_thresh2 = np.quantile(obs, target_quantile)
    
    # Step 4: Guard against the upper clipping limit artifact
    max_allowable_thresh = clipping_limit - safe_margin
    if calculated_thresh2 >= clipping_limit:
        # This occurs if the unclipped/foggy data represents a very small fraction 
        # of the dataset, forcing the target percentile into the 10km plateau.
        calculated_thresh2 = max_allowable_thresh
        
    return [fixed_thresh1, calculated_thresh2]

def compute_generalized_seeps(obs_series, fct_series, thresholds=[0.8, 2.0], P_baseline=None):
    """Computes generalized SEEPS error score handling NaNs cleanly."""
    # Align and drop missing observations or forecasts
    df_temp = pd.DataFrame({'obs': obs_series, 'fct': fct_series}).dropna()
    if len(df_temp) == 0:
        return np.nan
        
    obs_arr = df_temp['obs'].values
    fct_arr = df_temp['fct'].values
    
    # Classify into 3 categories (indices 0, 1, 2) using 2 thresholds
    obs_cat = np.digitize(obs_arr, thresholds, right=True)
    fct_cat = np.digitize(fct_arr, thresholds, right=True)
    
    n_thresholds = len(thresholds)
    total_penalty = np.zeros_like(obs_arr, dtype=float)
    
    # Decompose the 3-category space into scoring matrices at each threshold
    for k in range(n_thresholds):
        obs_below = (obs_cat <= k)
        fct_below = (fct_cat <= k)
        
        p_k = P_baseline[k]
        penalty_false_alarm = 1.0 / (1.0 - p_k)
        penalty_miss = 1.0 / p_k
        
        total_penalty += np.where(obs_below & ~fct_below, penalty_false_alarm, 0.0)
        total_penalty += np.where(~obs_below & fct_below, penalty_miss, 0.0)
        
    return np.mean(total_penalty / n_thresholds)

# 1. Extract only physical deterministic/persistence profiles to verify
continuous_models = [name for name, meta in MODEL_REGISTRY.items() if meta['type'] in ['det', 'pers']]
continuous_models += ['TAF_Base', 'TAF_Pessimistic', 'TAF_Optimistic']

seeps_records = []

# 2. Step through identical verification periods and operational sub-splits
for start_t, end_t, p_name in PERIODS:
    t_mask = (taf_eval.index >= start_t) & (taf_eval.index <= end_t)
    
    sub_periods = {
        'Full':        t_mask if CONFIG['model_24h'] else (t_mask & mask_valid),
        'First_Half':  t_mask & mask_1st,
        'Second_Half': t_mask & mask_2nd
    }
    
    for split_name, current_mask in sub_periods.items():
        obs_window = taf_eval['obs_vis'].loc[current_mask]
        # Establish stable baseline climatology from all valid cruise observations
        seeps_threshs = compute_dynamic_seeps_thresholds(obs_window, fixed_thresh1=0.8, clipping_limit=10.0, max_entropy=True)
        valid_obs_all = taf_eval['obs_vis'].dropna().values
        obs_cat_all = np.digitize(valid_obs_all, seeps_threshs, right=True)
        n_cats = len(seeps_threshs) + 1  # Evaluates to 3 categories
        p_climatology = np.array([np.sum(obs_cat_all == c) / len(valid_obs_all) for c in range(n_cats)])
        p_climatology = np.clip(p_climatology, 1e-4, None)  # Prevent divide-by-zero
        p_climatology /= p_climatology.sum()
        P_baseline = np.cumsum(p_climatology)[:-1]
        
        for m_name in continuous_models:
            fct_window = model_data[m_name].loc[current_mask]
            error_score = compute_generalized_seeps(obs_window, fct_window, thresholds=seeps_threshs, P_baseline=P_baseline)
            
            seeps_records.append({
                'Period': p_name,
                'Split': split_name,
                'SEEPS thresh.': seeps_threshs,
                'Model': m_name,
                'SEEPS_Score': 1 - error_score
            })
seeps_df = pd.DataFrame(seeps_records)

print("\n=== MULTI-CATEGORICAL EVALUATION SUMMARY (SCORE = 1 - SEEPS) ===")
print("\n")
for _, _, pname in PERIODS:
    print(f"{pname}")
    print_data = seeps_df[(seeps_df['Period'] == pname) & (seeps_df['Split'] == 'Full')]
    print("Note: 1.0 = Perfect Forecast, 0.0 = Climatology Baseline Performance\n")
    # Access the first element of the 'SEEPS thresh.' column safely
    print(f"SEEPS threshold for the dataset: {print_data['SEEPS thresh.'].iloc[0]}")
    print("-" * 67)
    print(print_data[['Model', 'SEEPS_Score']].to_string(index=False, float_format="%.4f"))
    print("=" * 67)
    print("\n")


#%%

# 3c. SEEPS VISUALIZATION (WITH GROUP SEPARATORS)
# ---------------------------------------------------------
# Generate a grouped bar chart to compare inverted SEEPS scores
# with vertical lines separating each verification window.
# ---------------------------------------------------------

# 1. Filter for the 'Full' temporal split
df_plot = seeps_df[seeps_df['Split'] == 'Full'].copy()

# 2. Enforce chronological ordering of the periods along the X-axis
period_order = [p[2] for p in PERIODS]
df_plot['Period'] = pd.Categorical(df_plot['Period'], categories=period_order, ordered=True)
df_plot = df_plot.sort_values(['Period', 'Model'])

# 3. Pivot data to align scores neatly per model across rows of periods
pivot_df = df_plot.pivot(index='Period', columns='Model', values='SEEPS_Score')

# Filter and align columns to match your exact pipeline's color registry order
models_to_plot = [m for m in MODEL_STYLE.keys() if m in pivot_df.columns]
pivot_df = pivot_df[models_to_plot]

# 4. Initialize Plot Figure
fig, ax = plt.subplots(figsize=(11, 6))

n_periods = len(period_order)
n_models = len(models_to_plot)
total_group_width = 0.8
bar_width = total_group_width / n_models
x_indexes = np.arange(n_periods)

# 5. Iterate through each model and plot its respective bar group
for i, model_name in enumerate(models_to_plot):
    offsets = x_indexes - (total_group_width / 2) + (i + 0.5) * bar_width
    scores = pivot_df[model_name].values
    color = MODEL_STYLE.get(model_name, 'gray')
    
    # Render the primary bars
    bars = ax.bar(
        offsets, scores, 
        width=bar_width, 
        label=model_name, 
        color=color, 
        alpha=0.80, 
        edgecolor='none'
    )
    
    # Isolate the 'Entire Cruise' bar and apply a heavy operational border
    entire_cruise_bar = bars[-1]
    entire_cruise_bar.set_edgecolor('black')
    entire_cruise_bar.set_linewidth(2.2)
    entire_cruise_bar.set_alpha(1.0)

# 6. Plot Reference Lines and Theoretical Boundaries 
ax.axhline(0.0, color='crimson', linestyle='--', linewidth=1.5, label='Climatology Baseline (Skill = 0)')
ax.axhline(1.0, color='black', linestyle='-', linewidth=1.2, label='Perfect Forecast (Skill = 1)')

# --- Vertical lines to separate the different groups of columns ---
for x_sep in range(n_periods - 1):
    ax.axvline(
        x=x_sep + 0.5, 
        color='darkgray', 
        linestyle='-', 
        linewidth=1.2, 
        alpha=0.6, 
        zorder=1
    )
# Preserved user-defined operational highlighting block for the final summary window
ax.axvspan(x_sep + 0.5, n_periods - 0.5, color="yellow", alpha=0.2)

# 7. Chart Aesthetics and Labels
ax.set_title('Multi-Model SEEPS Skill Score Comparison Across Operational Windows (3-Category Model)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('SEEPS Skill Score (Positive Orientation)', fontsize=11, fontweight='bold')
ax.set_xlabel('Verification Evaluation Windows', fontsize=11, fontweight='bold')

ax.set_xticks(x_indexes)
ax.set_xticklabels(period_order, fontsize=10, fontweight='bold')

# Dynamically set ymin to catch negative skill values, ymax capped just above 1.0
ymin = min(pivot_df.min().min() - 0.1, -0.2)
ax.set_ylim(ymin, 1.1)
ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)

# Position the legend cleanly outside the main plotting coordinate canvas
ax.legend(
    bbox_to_anchor=(1.02, 1), 
    loc='upper left', 
    frameon=True, 
    facecolor='white', 
    edgecolor='gainsboro',
    title="Models / Frameworks"
)

# Text annotation adjusted for positive orientation upward direction
ax.text(
    0.02, 0.93, '↑ Higher Value (Higher Skill)', 
    transform=ax.transAxes, fontsize=9, style='italic', 
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

plt.tight_layout()

###############################################################################################################################################################################
#%%
import numpy as np
from sklearn.mixture import GaussianMixture

# 1. ROBUST GRID-BASED GMM THRESHOLD GENERATION
# ---------------------------------------------------------
valid_obs_all = taf_eval['obs_vis'].dropna().values

fixed_thresh1 = 0.8
remainder_data = valid_obs_all[valid_obs_all > fixed_thresh1]
log_data = np.log(remainder_data).reshape(-1, 1)

# Fit GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(log_data)

means = gmm.means_.flatten()
weights = gmm.weights_.flatten()
variances = gmm.covariances_.flatten()

def pdf_cluster0(x):
    return weights[0] * (1.0 / np.sqrt(2 * np.pi * variances[0])) * np.exp(-0.5 * ((x - means[0])**2 / variances[0]))

def pdf_cluster1(x):
    return weights[1] * (1.0 / np.sqrt(2 * np.pi * variances[1])) * np.exp(-0.5 * ((x - means[1])**2 / variances[1]))

# Create a fine physical grid from just above your fog limit to the instrument ceiling
grid_kms = np.linspace(fixed_thresh1 + 0.05, 10.0, 2000)
grid_log = np.log(grid_kms)

# Find the absolute difference between the two regime profiles across the grid
diff_grid = np.abs(pdf_cluster0(grid_log) - pdf_cluster1(grid_log))

# Direct Extraction: Where is the intersection/closest approach point?
optimal_index = np.argmin(diff_grid)
calculated_thresh2 = grid_kms[optimal_index]

# Secure operational boundaries
if calculated_thresh2 >= 9.5 or calculated_thresh2 <= fixed_thresh1:
    calculated_thresh2 = 4.0  # Operational fallback

seeps_threshs = [fixed_thresh1, calculated_thresh2]
print(f"--> GMM Identified Thresholds: {seeps_threshs[0]:.1f} km and {seeps_threshs[1]:.1f} km")

# Back-calculate empirical probabilities and the natural data constant 'c'
p1 = np.mean(valid_obs_all <= fixed_thresh1)
p2 = np.mean((valid_obs_all > fixed_thresh1) & (valid_obs_all <= calculated_thresh2))
p3 = np.mean(valid_obs_all > calculated_thresh2)
c_natural = p3 / max(p2, 1e-4)
print(f"--> Natural Dataset Constant (c = p3/p2): {c_natural:.2f}")

# %%
