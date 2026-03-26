import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve

def TAF_parser(taf_string):
    """
    Parses a raw TAF string into a structured TAF object.

    Parameters
    ----------
    taf_string : str
        The raw TAF text (e.g., 'TAF EBBR 121130Z...').

    Returns
    -------
    taf : metar_taf_parser.model.taf.Taf
        A structured TAF object containing visibility and trend information.
    """
    from metar_taf_parser.parser.parser import TAFParser

    clean_taf = taf_string.strip()
    if not clean_taf.startswith('TAF'):
        clean_taf = 'TAF ' + clean_taf
    
    taf = TAFParser().parse(clean_taf)

    # Trends (TEMPO, BECMG, etc.)
    base_vis = taf.visibility.distance  # in meters
    # print("Base visibility:", base_vis)
    for trend in taf.trends:
        trend_type = trend.type.name  # TEMPO, BECMG, etc.
        trend_vis = trend.visibility.distance if trend.visibility else None
        start_hour = trend.validity.start_hour
        end_hour = trend.validity.end_hour
        # print(f"{trend_type}: {trend_vis}, from {start_hour:02d}:00 to {end_hour:02d}:00")
    return taf

def df_TAF_gen(taf_table, time_vec, taf_day0):
    columns = ['base','tempo','becmg',"prob30","prob40", "main_vis", "is_valid"]
    df = pd.DataFrame(np.nan, index=time_vec, columns=columns)
    df['is_valid'] = False 

    for idx, row in taf_table.iterrows():
        try:
            raw_taf = str(row['TAF Oden'])
            if 'nan' in raw_taf.lower() or len(raw_taf) < 10: 
                continue 
            
            # This replaces the "taf_day0 + idx" logic
            # It reads your "8/17/2025" format directly
            taf_date = pd.to_datetime(row['Date']).normalize() 
            
            taf = TAF_parser(raw_taf)
            taf_start = taf_date + pd.Timedelta(hours=taf.validity.start_hour)
            taf_end = taf_date + pd.Timedelta(hours=taf.validity.end_hour)
            
            # Standard safety for any potential midnight crossings
            if taf_end <= taf_start:
                taf_end += pd.Timedelta(days=1)

            # --- SET VALIDITY MASK ---
            df.loc[taf_start:taf_end, 'is_valid'] = True

            def parse_vis_dist(dist_str):
                if not dist_str: 
                    return 10.0 # Standard assumption for CAVOK/Missing distance
                
                # Handle "9999" (10km+) or "CAVOK" strings specifically
                if "9999" in dist_str or "CAVOK" in dist_str:
                    return 10.0
                    
                # Remove non-numeric except decimal points
                num_part = ''.join(c for c in dist_str if c.isdigit() or c == '.')
                
                if not num_part: 
                    return 10.0 if '>' in dist_str else np.nan
                    
                # Standard conversion (meters to km)
                val = float(num_part) / 1000.0
                
                # If the value is very small (e.g. 0.01), it was likely already in km
                # This acts as a safety for mixed-unit parsing
                if val < 0.1 and float(num_part) > 0:
                    return float(num_part) 
                    
                return val

            base_viz = parse_vis_dist(taf.visibility.distance)
            
            df.loc[taf_start:taf_end, 'base'] = base_viz
            df.loc[taf_start:taf_end, 'main_vis'] = base_viz

            # --- HANDLE TRENDS ---
            for trend in taf.trends:
                start_t = taf_date + pd.Timedelta(hours=trend.validity.start_hour) 
                end_t = taf_date + pd.Timedelta(hours=trend.validity.end_hour)
                if end_t <= start_t: end_t += pd.Timedelta(days=1)
                
                t_vis = parse_vis_dist(trend.visibility.distance) if trend.visibility else None

                if trend.type.name == 'BECMG' and t_vis is not None:
                    # 1. Get the current visibility right before this trend starts
                    # We use 'ffill' logic to get the last known state in the 'main_vis' column
                    v_start = df.loc[:start_t, 'main_vis'].ffill().iloc[-1]
                    
                    if pd.isna(v_start): 
                        v_start = base_viz

                    # 2. Identify the range
                    target_indices = df.loc[start_t:end_t].index
                    
                    if len(target_indices) > 1:
                        # Create the linear transition
                        ramp = np.linspace(v_start, t_vis, len(target_indices))
                        df.loc[target_indices, 'main_vis'] = ramp
                    else:
                        # If the window is too small for the time_vec resolution, jump to target
                        df.at[start_t, 'main_vis'] = t_vis
                    
                    # Update the state for the remainder of the TAF 
                    # so the NEXT trend starts from this new height
                    df.loc[end_t:taf_end, 'main_vis'] = t_vis

                elif trend.type.name in ['TEMPO', 'PROB30', 'PROB40'] and t_vis is not None:
                    df.loc[start_t:end_t, trend.type.name.lower()] = t_vis
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            continue
    return df

def calculate_scenarios(df):
    """
    Computes the deterministic 'Main' scenario and the probabilistic 'Best/Worst' 
    visibility bounds based on all available TAF trends (TEMPO, PROB).

    Parameters
    ----------
    df : pd.DataFrame
        TAF DataFrame containing 'main_vis', 'tempo', and 'prob' columns.

    Returns
    -------
    df : pd.DataFrame
        Updated DataFrame with 'worst_vis', 'best_vis', and 'main_scenario'.
    """
    comparison_cols = ['main_vis', 'tempo', 'prob30', 'prob40']
    df['worst_vis'] = df[comparison_cols].min(axis=1)
    df['best_vis'] = df[comparison_cols].max(axis=1)
    df['main_scenario'] = df['main_vis']
    return df

def assign_event_probabilities(df, v_thresh=1.0):
    """
    Maps TAF categorical trends to numerical event probabilities $P(\text{Vis} < v_{thresh})$.
    Priority follows: Main (100%) > PROB40 (40%) > PROB30 (30%) > TEMPO (10%).

    Parameters
    ----------
    df : pd.DataFrame
        TAF DataFrame with scenario columns.
    v_thresh : float, optional
        The visibility threshold (km) defining the "event" (e.g., fog), by default 1.0.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the 'p_event' column added.
    """
    df['p_event'] = 0.0
    
    # Use .maximum to ensure we keep the highest probability found for that minute
    # Priority: Main (100%) > PROB40 (40%) > PROB30 (30%) > TEMPO (10%)    
    mask_tempo = (df['tempo'] < v_thresh)
    df.loc[mask_tempo, 'p_event'] = np.maximum(df.loc[mask_tempo, 'p_event'], 0.1)
    
    mask_p30 = (df['prob30'] < v_thresh)
    df.loc[mask_p30, 'p_event'] = np.maximum(df.loc[mask_p30, 'p_event'], 0.3)
    
    mask_p40 = (df['prob40'] < v_thresh)
    df.loc[mask_p40, 'p_event'] = np.maximum(df.loc[mask_p40, 'p_event'], 0.4)
    
    mask_main = (df['main_scenario'] < v_thresh)
    df.loc[mask_main, 'p_event'] = 1.0 # Main always takes priority if it's below thresh
    
    df.loc[df['is_valid'] == False, 'p_event'] = np.nan
    return df

def get_metrics(forecast_bool, obs_bool):
    """
    Calculates standard binary verification metrics based on a $2 \times 2$ 
    contingency table.

    Metrics calculated:
    - $POD = \frac{a}{a+c}$
    - $FAR = \frac{b}{a+b}$
    - $Bias = \frac{a+b}{a+c}$
    - $CSI = \frac{a}{a+b+c}$

    Parameters
    ----------
    forecast_bool : pd.Series (bool)
        Boolean series indicating if the event was predicted.
    obs_bool : pd.Series (bool)
        Boolean series indicating if the event was observed.

    Returns
    -------
    metrics : dict
        Dictionary containing POD, FAR, Bias, CSI, Hits, and Misses.
    """
    # Ensure inputs are pandas Series
    f = pd.Series(forecast_bool)
    o = pd.Series(obs_bool)

    # Convert to boolean, treating NaN as False
    # This fixes the "bad operand type for unary ~" error
    valid = (~f.isna()) & (~o.isna())

    f_bin = f[valid].astype(bool)
    o_bin = o[valid].astype(bool)

    a = (f_bin & o_bin).sum()    # Hit
    b = (f_bin & ~o_bin).sum()   # False Alarm
    c = (~f_bin & o_bin).sum()   # Miss
    d = (~f_bin & ~o_bin).sum()  # Correct Negative
    
    pod = a / (a + c) if (a + c) > 0 else 0
    far = b / (a + b) if (a + b) > 0 else 0
    bias = (a + b) / (a + c) if (a + c) > 0 else 0
    csi = a / (a + b + c) if (a + b + c) > 0 else 0
    
    return {"POD": pod, "FAR": far, "Bias": bias, "CSI": csi, "Hits": a, "Misses": c}

def compute_all_metrics(truth, event_library):
    """
    Computes metrics for all entries in the event library.
    """
    all_metrics = {}
    for name, event_series in event_library.items():
        # Using your existing get_metrics function
        all_metrics[name] = get_metrics(event_series, truth)
    
    return pd.DataFrame(all_metrics).T

def plot_metrics_summary(metrics_df):
    """
    Splits metrics into two plots: Ratios (0-1) and Absolute Counts.
    """
    # 1. Plot Ratios (POD, FAR, CSI, Bias)
    ratios = metrics_df[['POD', 'FAR', 'CSI', 'Bias']]
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ratios.plot(kind='bar', ax=ax1, rot=0, edgecolor='black', alpha=0.8)
    ax1.set_ylim(0, 1.2) # Bias might go > 1, so 1.2 is a safe cap
    ax1.set_title('Performance Ratios (POD, FAR, CSI)')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    # 2. Plot Absolute Counts (Hits, Misses)
    counts = metrics_df[['Hits', 'Misses']]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    counts.plot(kind='bar', ax=ax2, rot=0, edgecolor='black', alpha=0.8)
    ax2.set_title('Absolute Frequency (Hits vs Misses)')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    return fig1, fig2

def get_evaluation_library(df, model_dict, obs_series, p_thresh = 0.0,fog_thresh=1.0):
    """
    Creates a standardized library of boolean event series for all models and the forecaster.
    
    Parameters
    ----------
    df : pd.DataFrame
        Contains the forecaster 'p_event' column.
    model_dict : dict
        { 'Name': pd.Series(vis_km) }
    obs_series : pd.Series
        Visibility observations in km.
    """
    # 1. Start with the observations (Truth)
    truth = (obs_series < fog_thresh)
    
    # 2. Build the Event Library
    event_library = {}
    
    # Add Forecaster (treating P > 0 as an event)
    event_library['Forecaster'] = (df['p_event'] > p_thresh)
    
    # Add all numerical models
    for name, vis_series in model_dict.items():
        event_library[name] = (vis_series < fog_thresh)
        
    return truth, event_library

def evaluate_models(model_dict, obs_series, forecaster_p_series, fog_thresh=1.0):
    """
    Computes metrics for N models and the Forecaster simultaneously.

    Parameters
    ----------
    model_dict : dict
        Dictionary { 'ModelName': pd.Series(vis_data_km) }
    obs_series : pd.Series
        Oden observations (visibility in km).
    forecaster_p_series : pd.Series
        Forecaster probability series (P > 0 implies fog event).
    fog_thresh : float
        Visibility threshold for fog definition (km).

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame of metrics, indexed by model name.
    """
    import pandas as pd
    
    # Initialize results container
    results = {}
    
    # 1. Compute Forecaster Metrics
    obs_event = (obs_series < fog_thresh)
    results['Forecaster'] = get_metrics(forecaster_p_series > 0, obs_event)
    
    # 2. Compute Model Metrics
    for name, vis_series in model_dict.items():
        is_fog = (vis_series < fog_thresh)
        results[name] = get_metrics(is_fog, obs_event)
    
    # Return as a clean DataFrame (Models as rows, Metrics as columns)
    return pd.DataFrame(results).T

def plot_vis_summary(df, vis_obs, vis_mod, fog_thresh,start_date=None, end_date=None):
    """
    Plots a log-scale time series comparison of TAF scenarios, 
    model output, and observations.
    """
    obs_series = vis_obs.reindex(df.index, method='nearest')
    mod_series = vis_mod.reindex(df.index, method='nearest')
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    
    if start_date and end_date:
        plot_df = df.loc[start_date:end_date]
        plot_obs = obs_series.loc[start_date:end_date]
        plot_mod = mod_series.loc[start_date:end_date]
    else:
        plot_df, plot_obs, plot_mod = df, obs_series, mod_series

    ax.fill_between(plot_df.index, plot_df['worst_vis'], plot_df['best_vis'], 
                    color='lightgray', alpha=0.5, label='TAF Uncertainty (TEMPO/PROB)')
    ax.plot(plot_df.index, plot_df['main_scenario'], color='black', linewidth=1.2, 
            label='TAF Main (Base/BECMG)', drawstyle='steps-post')
    ax.plot(plot_df.index, plot_mod, color='blue', linestyle='--', linewidth=1.5, 
            label='ECMWF Model (IFS Oper)', drawstyle='steps-post')
    ax.plot(plot_df.index, plot_obs, color='crimson', linewidth=2, label='Oden Observations')

    ax.set_yscale('log')
    ax.set_ylim(0.04, 15)
    y_ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(t) for t in y_ticks])
    ax.axhline(y=fog_thresh, color='red', linestyle=':', alpha=0.5, label='Fog Threshold (1km)')
    ax.set_ylabel('Visibility [km] (Log Scale)')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax.fill_between(plot_df.index, 0, 1, where=~plot_df['is_valid'], 
                    color='yellow', alpha=0.1, transform=ax.get_xaxis_transform(), label='No TAF')

    ax.legend(loc='lower right', frameon=True, fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

def calculate_stacked_probabilities(df):
    """
    Groups TAF scenarios and trends into discrete visibility bins to create 
    a "Forecaster Ensemble" probability distribution.
    """
    bins = [0, 0.15, 0.35, 0.6, 0.8, 1.5, 3.0, 5.0, 10.0]
    bin_names = ['<150m', '150-350m', '350-600m', '600-800m', '0.8-1.5km', '1.5-3km', '3-5km', '5-10km']
    prob_df = pd.DataFrame(0.0, index=df.index, columns=bin_names)
    
    for i, row in df.iterrows():
        if not row['is_valid']: continue
        main_val = row['main_scenario']
        main_bin_idx = np.digitize(main_val, bins) - 1
        main_exists = 0 <= main_bin_idx < len(bin_names)
        if main_exists: prob_df.at[i, bin_names[main_bin_idx]] = 1.0
            
        for col, weight in [('tempo', 0.1), ('prob30', 0.3), ('prob40', 0.4)]:
            val = row[col]
            if pd.notna(val):
                trend_bin_idx = np.digitize(val, bins) - 1
                if 0 <= trend_bin_idx < len(bin_names):
                    prob_df.at[i, bin_names[trend_bin_idx]] += weight
                    if main_exists: prob_df.at[i, bin_names[main_bin_idx]] -= weight
                    
    return prob_df.clip(lower=0, upper=1)

def plot_ens_meteogram(prob_df, model_dict, vis_obs, start_date, end_date, resample_freq='3H'):
    """
    Plots TAF probabilities with multiple model rows below the probability stack.
    model_dict : dict of {name: pd.Series(vis_km)}
    """
    start_date, end_date = str(start_date), str(end_date)
    p_sub = prob_df.loc[start_date:end_date].resample(resample_freq).mean()
    obs_3h = vis_obs.reindex(p_sub.index, method='nearest')
    
    bins = [0, 0.15, 0.35, 0.6, 0.8, 1.5, 3.0, 5.0, 10.0]
    colors = ['#191970', '#E31A1C', '#FF7F00', '#FFFF33', '#00FFFF', '#1F78B4', '#33A02C', '#B2DF8A']

    fig, ax = plt.subplots(figsize=(13, 3 + (len([model for model in model_dict if "Ens" not in model]) * 1.5)))
    fig.suptitle(f"TAF-Based Meteogram with (deterministic) model comparison\n({start_date} to {end_date})", fontweight='bold', fontsize=14)
    
    # 1. Plot Probability Stack
    bottom = np.zeros(len(p_sub))
    for i, col in enumerate(p_sub.columns):
        ax.bar(p_sub.index, p_sub[col] * 100, bottom=bottom, width=0.08, 
               color=colors[i], label=col, align='center', edgecolor='white', linewidth=0.1)
        bottom += p_sub[col].values * 100

    # 2. Plot Observation Row (Fixed at bottom)
    for t in p_sub.index:
        o_idx = np.digitize(obs_3h.loc[t], bins) - 1
        ax.plot(t, -5, marker='s', markersize=10, color=colors[o_idx] if 0 <= o_idx < len(colors) else '#E0E0E0')

    # 3. Plot Each Model Row
    i = 0
    for (name, vis_series) in (model_dict.items()):
        if "Ens" not in name:
            y_pos = -15 - (i * 10) # Each model gets a new row
            right_x = p_sub.index.max() + (p_sub.index.max() - p_sub.index.min()) *0.01
            mod_3h = vis_series.reindex(p_sub.index, method='nearest')
            
            for t in p_sub.index:
                m_idx = np.digitize(mod_3h.loc[t], bins) - 1
                ax.plot(t, y_pos, marker='s', markersize=10, 
                        color=colors[m_idx] if 0 <= m_idx < len(colors) else '#E0E0E0')
            
            ax.text(right_x, y_pos, f'{name}  ', ha='left', va='center', fontweight='bold', fontsize=9)
            i += 1

    ax.text(right_x, -5, 'Observations  ', ha='left', va='center', fontweight='bold', fontsize=9, color='crimson')
    ax.set_ylim(-15 - (len([model for model in model_dict if "Ens" not in model]) * 10), 105)
    ax.set_ylabel('Forecaster Probability [%]')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d\n%H:%M'))
    
    # Clean up aesthetics
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Visibility Bins", frameon=False)
    for day in pd.date_range(start_date, end_date, freq='D'):
        ax.axvline(day, c='k', alpha=0.3, lw=0.5)
        
    plt.tight_layout()
    plt.show()
    
def plot_performance_diagram(pods, fars, labels, colors=None):
    """
    Plots a Roebber (2009) Performance Diagram, showing POD vs Success Ratio 
    with CSI contours and Bias lines. 
    Ideal forecasts cluster toward the top-right corner.
    """
    x = np.linspace(0.001, 1, 100)
    y = np.linspace(0.001, 1, 100)
    SR_grid, POD_grid = np.meshgrid(x, y)
    CSI = 1 / (1/SR_grid + 1/POD_grid - 1) 

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    contour = ax.contourf(SR_grid, POD_grid, CSI, levels=np.arange(0, 1.1, 0.1), cmap='Greys', alpha=0.2)
    cbar = plt.colorbar(contour, ax=ax, pad=0.075)
    cbar.set_label('Critical Success Index (CSI)', fontsize=14)
    
    bias_values = [0.5, 0.8, 1, 1.3, 1.5, 2, 4]
    for b in bias_values:
        end_x, end_y = (1, b) if b <= 1 else (1/b, 1)
        ax.plot([0, end_x], [0, end_y], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.text(end_x, end_y, f' B={b}', fontsize=13, alpha=0.7)

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(pods)))

    for pod, far, label, color in zip(pods, fars, labels, colors):
        ax.scatter(1 - far, pod, s=120, label=label, color=color, edgecolor='black', zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=14)
    ax.set_ylabel('Probability of Detection (POD)', fontsize=14)
    # ax.set_title('Visibility Performance: TAF vs. Model Parametrizations', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='lower right', frameon=True, prop={'size': 13})
    plt.tight_layout()
    plt.show()

def plot_taf_window(df, fog_thresh, start_time, end_time):
    """
    Plots the Main, Best, and Worst TAF scenarios for a specific time window.
    Useful for detailed case studies of specific fog events.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    pdf = df.loc[start_time:end_time]
    
    ax.plot(pdf.index, pdf.best_vis, c="forestgreen", ls="--", alpha=0.7, label="Best Case")
    ax.plot(pdf.index, pdf.worst_vis, c="crimson", ls="--", alpha=0.7, label="Worst Case")
    ax.plot(pdf.index, pdf.main_vis, c="k", lw=1.8, label="Main Scenario (Base/BECMG)", drawstyle='steps-post')
    
    ax.axhline(fog_thresh, c='r', ls=":", alpha=0.8, label=f"Fog Threshold ({fog_thresh} km)")
    ax.set_ylabel("Visibility [km]", fontsize=12)
    ax.set_ylim(0, 10.5)
    ax.set_title(f"TAF Scenarios: {start_time} to {end_time}", fontweight='bold')
    
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    
    for day in pd.date_range(pdf.index.min(), pdf.index.max(), freq='D'):
        ax.axvline(day, c='k', alpha=0.3, lw=0.8, zorder=0)
        
    ax.legend(frameon=True, loc='upper right', fontsize='small')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    return fig, ax

def plot_taf_components(df):
    """
    Visualizes raw TAF components (Base, TEMPO, BECMG) as recorded.
    Useful for assessing how forecasters utilize specific TAF change-indicators.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df['base'], label='Base Visibility', color='royalblue', lw=1.5)
    ax.plot(df.index, df['tempo'], label='TEMPO', color='darkorange', marker='.', markersize=2, ls='')
    ax.plot(df.index, df['becmg'], label='BECMG', color='seagreen', lw=2)
    
    ax.set_ylim(0, 10.5)
    ax.set_ylabel('Visibility [km]')
    ax.set_title('Raw TAF Components (Evolution over Cruise)', fontweight='bold')

    days = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq='D')
    for i, day in enumerate(days[:-1]):
        if i % 2 == 0:
            ax.axvspan(day, day + pd.Timedelta(days=1), color='gray', alpha=0.1, zorder=0)

    ax.legend(loc='upper right', ncol=3)
    plt.tight_layout()
    return fig, ax

def apply_diurnal_mask(df, start_hour, end_hour, day_only=True):
    """
    Filters a DataFrame to only include specific hours of the day.
    
    Parameters:
    df : pd.DataFrame or pd.Series
    start_hour : int (0-23)
    end_hour : int (0-23)
    day_only : bool (If True, keeps hours between start/end. 
                     If False, keeps hours outside that range).
    """
    # Extract the hour from the datetime index
    hours = df.index.hour
    
    if day_only:
        # e.g., 07:00 to 15:00
        mask = (hours >= start_hour) & (hours <= end_hour)
    else:
        # e.g., everything EXCEPT 07:00 to 15:00
        mask = (hours < start_hour) | (hours > end_hour)
        
    return df.where(mask)

def plot_ensemble_spaghetti(ens_xr, obs_series, start_t, end_t, threshold=0.8):
    """
    Plots individual ensemble members as 'spaghetti' to show forecast spread.
    """
    import matplotlib.pyplot as plt
    
    # Slice data for the window
    window_ens = ens_xr.sel(time=slice(start_t, end_t))
    window_obs = obs_series.loc[start_t:end_t]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual members (Thin and Transparent)
    for i in window_ens.number.values:
        ax.plot(window_ens.time, window_ens.sel(number=i), 
                color='gray', alpha=0.5, lw=0.5, zorder=1)
    
    # Plot Observations
    ax.plot(window_obs.index, window_obs, 
            color='red', label='Observations', zorder=3)

    # Plot Ensemble Median
    ax.plot(window_ens.time, window_ens.median(dim='number'), 
            color='k',  label='Ensemble Median', zorder=2, lw=0.8)
    
    # Fog Threshold Line
    ax.axhline(threshold, color='black', ls='--', alpha=0.6, label='Fog Threshold')
    
    ax.set_ylim(0, 10.5) # Focus on the low-visibility range
    ax.set_ylabel('Visibility [km]')
    ax.set_title(f'Ensemble Spread vs. Observations ({start_t} to {end_t})')
    ax.legend()
    plt.tight_layout()
    return fig, ax

def compute_brier_score(f, o):
    # f = probabilities (0 to 1), o = binary observations (0 or 1)
    
    # Ensure no NaNs from your masks interfere
    valid_mask = f.notna() & o.notna()
    return ((f[valid_mask] - o[valid_mask])**2).mean()

def plot_reliability_diagram(prob_forecast, obs_binary, n_bins=10):
    # prob_forecast: 0.0 to 1.0 (Ens_Prob)
    # obs_binary: 0 or 1 (obs_event)
    
    # Remove NaNs
    valid = prob_forecast.notna() & obs_binary.notna()
    y_true = obs_binary[valid]
    y_prob = prob_forecast[valid]
    
    # Calculate calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", label="Ensemble")
    
    ax.set_xlabel("Forecasted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    plt.grid(True)
    return fig,ax

def plot_talagrand_histogram(ens_data, obs_data):
    """
    ens_data: xarray DataArray with dims (time, number)
    obs_data: pandas Series aligned to time
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Find the intersection of timestamps where both have valid data
    ens_times = ens_data.time.to_index()
    obs_times = obs_data.dropna().index
    common_time = ens_times.intersection(obs_times)

    if len(common_time) == 0:
        print("Error: No overlapping timestamps found between ensemble and observations.")
        return

    # 2. Subset both to the common timestamps
    ens_subset = ens_data.sel(time=common_time)
    obs_subset = obs_data.loc[common_time].values  # Convert to numpy for broadcasting

    # 3. Vectorized Rank Calculation
    # We compare the observation (Time, 1) to the ensemble (Time, Number)
    # The sum across the 'number' dimension gives the rank for each time step
    ranks = (ens_subset < obs_subset[:, np.newaxis]).sum(dim='number').values

    # 4. Plotting
    n_members = len(ens_data.number)
    plt.figure(figsize=(8, 5))
    
    # We want bins for each possible rank (0 to n_members)
    counts, bins, _ = plt.hist(ranks, bins=np.arange(n_members + 2) - 0.5, 
                               density=True, edgecolor='black', alpha=0.7, color='skyblue')
    
    # The "Ideal" line for a perfectly calibrated ensemble
    plt.axhline(1 / (n_members + 1), color='red', linestyle='--', lw=2, label='Perfectly Calibrated')
    
    plt.xlabel('Rank (Number of members < Observation)')
    plt.ylabel('Relative Frequency')
    plt.title(f'Talagrand Diagram (Rank Histogram)\nN = {n_members} members')
    plt.xticks(range(0, n_members + 1, max(1, n_members // 10)))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return ranks