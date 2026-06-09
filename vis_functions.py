# Maintainer: L.L. Donati - lorenzo.luca.donati@misu.su.se
# Scripts for "On the predictability of near-surface visibility over the Arctic Ocean"
# Authors: Luise Schulte, Lorenzo Luca Donati, Vania Lopez Garcia, Linus Magnusson, Ian M. Brooks

#%% Imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['figure.dpi'] = 300
import pandas as pd
import numpy as np
from metar_taf_parser.parser.parser import TAFParser
from sklearn.calibration import calibration_curve
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

#%% Functions
def TAF_parser(taf_string, debug=False):
    """
    Parses a raw TAF string into a structured TAF object.

    Parameters
    ----------
    taf_string : str
        The raw TAF text (e.g., 'TAF EBBR 121130Z...').
    debug : bool
        Whether to print debugging info or not.

    Returns
    -------
    taf : metar_taf_parser.model.taf.Taf
        A structured TAF object containing visibility and trend information.
    """
    clean_taf = taf_string.strip()
    if not clean_taf.startswith('TAF'):
        clean_taf = 'TAF ' + clean_taf
    
    taf = TAFParser().parse(clean_taf)

    if debug:
        # Trends (TEMPO, BECMG, etc.)
        base_vis = taf.visibility.distance  # in meters
        print("Base visibility:", base_vis)
        for trend in taf.trends:
            trend_type = trend.type.name  # TEMPO, BECMG, etc.
            trend_vis = trend.visibility.distance if trend.visibility else None
            start_hour = trend.validity.start_hour
            end_hour = trend.validity.end_hour
            print(f"{trend_type}: {trend_vis}, from {start_hour:02d}:00 to {end_hour:02d}:00")
    return taf

def df_TAF_gen(taf_table, time_vec, debug):
    """
    Generates a DataFrame of TAF visibility data by parsing TAF strings and extracting
    base visibility, BECMG trends (with linear interpolation), and TEMPO/PROB variations.

    Parameters
    ----------
    taf_table : pd.DataFrame
        DataFrame containing 'TAF Oden' column (raw TAF strings) and 'Date' column 
        (in format 'M/D/YYYY', e.g., '8/17/2025').
    time_vec : pd.DatetimeIndex
        The time index for the output DataFrame (hourly or sub-hourly timestamps).
    debug : bool
        Whether to print debugging info or not.

    Returns
    -------
    df : pd.DataFrame
        DataFrame indexed by time_vec with columns:
        - 'base': Base visibility from TAF (km)
        - 'tempo': TEMPO trend visibility (km)
        - 'becmg': BECMG trend visibility (km)
        - 'prob30': PROB30 trend visibility (km)
        - 'prob40': PROB40 trend visibility (km)
        - 'main_vis': Primary visibility scenario (base with BECMG interpolation applied)
        - 'is_valid': Boolean indicating if a TAF was active at that time

    Notes
    -----
    - BECMG trends are interpolated linearly from current to target visibility.
    - TEMPO and PROB trends are assigned as-is (no interpolation).
    - Visibility values are in kilometers (km).
    - Missing or invalid TAF strings are skipped with error logging.
    """
    columns = [
        'base','tempo','becmg',"prob30","prob40",
        "main_vis",
        "is_valid",
        "is_valid_first_half",
        "is_valid_second_half"
    ]
    df = pd.DataFrame(np.nan, index=time_vec, columns=columns)
    df['is_valid'] = False 
    df['is_valid_first_half'] = False
    df['is_valid_second_half'] = False

    for idx, row in taf_table.iterrows():
        try:
            raw_taf = str(row['TAF Oden'])
            if 'nan' in raw_taf.lower() or len(raw_taf) < 10: 
                continue 
            
            # safe date & time handling
            taf_date = pd.to_datetime(row['Date']).normalize() 
            taf = TAF_parser(raw_taf,debug)
            taf_start = taf_date + pd.Timedelta(hours=taf.validity.start_hour)
            taf_end = taf_date + pd.Timedelta(hours=taf.validity.end_hour)
            if taf_end <= taf_start:
                taf_end += pd.Timedelta(days=1)

            # Set validity mask
            df.loc[taf_start:taf_end, 'is_valid'] = True

            # Split TAF validity window
            taf_duration = taf_end - taf_start
            midpoint = taf_start + taf_duration / 2

            # First half includes midpoint
            df.loc[taf_start:midpoint, 'is_valid_first_half'] = True
            df.loc[midpoint:taf_end, 'is_valid_second_half'] = True

            def parse_vis_dist(dist_str):

                # Handle "9999" (10km+) or "CAVOK" strings specifically
                if dist_str == "> 10km":
                    num_part = 10000
                else:
                    # Remove non-numeric except decimal points
                    num_part = ''.join(c for c in dist_str if c.isdigit() or c == '.')
                    
                return float(num_part)

            base_viz = parse_vis_dist(taf.visibility.distance)/1000
            
            df.loc[taf_start:taf_end, 'base'] = base_viz
            df.loc[taf_start:taf_end, 'main_vis'] = base_viz

            # HAndle trends
            for trend in taf.trends:
                start_t = taf_date + pd.Timedelta(hours=trend.validity.start_hour) 
                end_t = taf_date + pd.Timedelta(hours=trend.validity.end_hour)
                if end_t <= start_t: end_t += pd.Timedelta(days=1)
                
                t_vis = parse_vis_dist(trend.visibility.distance)/1000 if trend.visibility else None

                if trend.type.name == 'BECMG' and t_vis is not None:
                    # 1. Get the current visibility right BEFORE this trend starts
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
                        # If the window is too small for the time_vec resolution, jupm to target
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
    
    return {"POD": pod, "FAR": far, "Bias": bias, "CSI": csi, "Hits": a, "Misses": c, "False alarms": b, "Correct negatives" : d}

def compute_all_metrics(truth, event_library):
    """
    Computes standard binary verification metrics for all forecasts in the event library.

    Parameters
    ----------
    truth : pd.Series (bool)
        Boolean series indicating observed events.
    event_library : dict
        Dictionary mapping forecast names to boolean event series.
        { 'ModelName': pd.Series(bool), ... }

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with models/forecasters as rows and metrics (POD, FAR, Bias, CSI, Hits, Misses) 
        as columns.
    """
    all_metrics = {}
    for name, event_series in event_library.items():
        all_metrics[name] = get_metrics(event_series, truth)
    
    return pd.DataFrame(all_metrics).T

def plot_metrics_summary(metrics_df):
    """
    Visualizes verification metrics as two subplots: ratios and absolute counts.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame output from compute_all_metrics with models as rows and 
        metrics (POD, FAR, CSI, Bias, Hits, Misses) as columns.

    Returns
    -------
    fig1 : matplotlib.figure.Figure
        Figure containing bar plot of POD, FAR, CSI, and Bias (0-1 scale).
    fig2 : matplotlib.figure.Figure
        Figure containing bar plot of absolute hit, miss,false alarms and correct negatives counts.
    """
    # 1. Plot Ratios (POD, FAR, CSI, Bias)
    ratios = metrics_df[['POD', 'FAR', 'CSI', 'Bias']]
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ratios.plot(kind='bar', ax=ax1, rot=0, edgecolor='black', alpha=0.8)
    ax1.set_ylim(0, 1.2) # Bias might go > 1, so 1.2 is a safe cap
    ax1.set_title('Performance Ratios (POD, FAR, CSI)')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
    
    # 2. Plot Absolute Counts (Hits, Misses)
    counts = metrics_df[['Hits', 'Misses','False alarms','Correct negatives']]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    counts.plot(kind='bar', ax=ax2, rot=0, edgecolor='black', alpha=0.8,width=0.8)
    ax2.set_title('Absolute Frequency (Hits vs Misses)')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
    
    # Add value labels on bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.0f')
    return fig1, fig2

def get_evaluation_library(model_dict, obs_series, fog_thresh, higher_than_fog_thresh):
    """
    Creates a standardized library of boolean event series for all models.

    Parameters
    ----------
    model_dict : dict
        Dictionary mapping model names to visibility series { 'Name': pd.Series(vis_km) }.
    obs_series : pd.Series
        Observed visibility time series (km).
    fog_thresh : float 
        Visibility threshold for defining fog event (km), by default 1.0.
        Event occurs when visibility < fog_thresh.
    higher_than_fog_thresh : bool
        If True, the "event" is defined as visibility > fog_thresh (opportunity). 
        If False, the "event" is defined as visibility <= fog_thresh (hazard), by default False.

    Returns
    -------
    truth : pd.Series (bool)
        Boolean series indicating observed fog events.
    event_library : dict
        Dictionary mapping forecast names to boolean event series.
        { 'Forecaster': pd.Series(bool), 'ModelName': pd.Series(bool), ... }
    """
    # 1. Start with the observations (Truth)
    if higher_than_fog_thresh:
        truth = (obs_series > fog_thresh)
    else:
        truth = (obs_series <= fog_thresh)
    
    # 2. Build the Event Library
    event_library = {}
    
    # Add all numerical models
    for name, vis_series in model_dict.items():
        if higher_than_fog_thresh:
            event_library[name] = (vis_series > fog_thresh)
        else:
            event_library[name] = (vis_series <= fog_thresh)
        
    return truth, event_library

def calculate_stacked_probabilities(df):
    """
    Groups TAF scenarios and trends into discrete visibility bins to create 
    a "Forecaster Ensemble" probability distribution.

    Parameters
    ----------
    df : pd.DataFrame
        TAF DataFrame containing 'main_scenario', 'tempo', 'prob30', 'prob40', 
        and 'is_valid' columns.

    Returns
    -------
    prob_df : pd.DataFrame
        DataFrame indexed by time with columns for each visibility bin.
        Each cell contains a probability (0.0 to 1.0) representing the likelihood 
        of visibility falling within that bin, based on TAF scenarios and trends.
        
    Notes
    -----
    - Main scenario receives a base probability of 1.0.
    - TEMPO trends add 0.1 probability weight.
    - PROB30 trends add 0.3 probability weight.
    - PROB40 trends add 0.4 probability weight.
    - Probabilities are clipped to [0, 1] range.
    - Invalid TAF periods are skipped (is_valid == False).
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

def compute_brier_score(f, o):
    """
    Computes the Brier Score, a measure of forecast probability accuracy.
    
    The Brier Score is the mean squared difference between forecasted probabilities 
    and observed binary outcomes. Lower values indicate better calibration.
    
    Parameters
    ----------
    f : pd.Series
        Forecasted probabilities (0.0 to 1.0).
    o : pd.Series
        Observed binary events (0 or 1).
    
    Returns
    -------
    brier_score : float
        Mean squared error between forecasts and observations.
    """
    # Ensure no NaNs from your masks interfere
    valid_mask = f.notna() & o.notna()

    return ((f[valid_mask] - o[valid_mask])**2).mean()

def plot_reliability_diagram(prob_forecast, obs_binary, n_bins=10):
    """
    Plots a reliability diagram showing forecast calibration.
    
    Compares forecasted probabilities against observed frequencies binned into 
    discrete probability intervals. Perfect calibration lies on the diagonal.

    Parameters
    ----------
    prob_forecast : pd.Series
        Forecasted probabilities (0.0 to 1.0).
    obs_binary : pd.Series
        Observed binary events (0 or 1).
    n_bins : int 
        Number of probability bins for calibration analysis, by default 10.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the reliability diagram.
    ax : matplotlib.axes.Axes
        Axes object for further customization.
    """
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
    Plots a Talagrand (rank) histogram to assess ensemble calibration.
    
    The Talagrand histogram shows how often observations fall within the range 
    of ensemble members. A perfectly calibrated ensemble produces a flat histogram.
    Bias toward low ranks indicates overconfidence; bias toward high ranks 
    indicates the ensemble is too dispersed.

    Parameters
    ----------
    ens_data : xarray.DataArray
        Ensemble visibility data with dimensions (time, number).
        Each 'number' represents an individual ensemble member.
    obs_data : pd.Series
        Observed visibility time series (km), indexed by datetime.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the rank histogram.
    ax : matplotlib.axes.Axes
        Axes object for further customization.
        
    Notes
    -----
    The rank is defined as the number of ensemble members with values less than 
    the observation. A uniform distribution across all ranks indicates good calibration.
    """

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
    # Compare the observation (time, 1) to the ensemble (time, number)
    # Sum across the 'number' dimension yields the rank for each time step
    ranks = (ens_subset < obs_subset[:, np.newaxis]).sum(dim='number').values

    # 4. Plotting
    n_members = len(ens_data.number)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Bins for each possible rank (0 to n_members)
    ax.hist(ranks, bins=np.arange(n_members + 2) - 0.5, 
            density=True, edgecolor='black', alpha=0.7, color='skyblue')
    
    # The "ideal" line for a  calibrated ensemble
    ax.axhline(1 / (n_members + 1), color='red', linestyle='--', lw=2, label='Uniform distribution')
    
    ax.set_xlabel('Rank (No. of members with visibility < observed)', fontsize=12)
    ax.set_ylabel('Relative Frequency', fontsize=12)
    # ax.set_title(f'Talagrand Diagram (Rank Histogram)\nN = {n_members} members', fontweight='bold')
    ax.set_xticks(range(0, n_members + 1, max(1, n_members // 10)))
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig, ax

def calculate_ets(a, b, c, d):
    """
    Calculates the Equitable Threat Score (ETS).
    
    Parameters:
    -----------
    a : int/float
        Hits
    b : int/float
        False Alarms
    c : int/float
        Misses
    d : int/float
        Correct Negatives
    
    Returns:
    -----------
    ets : float 
        ETS value (range -1/3 to 1. 1 is perfect, 0 is no skill).
    """
    
    n = a + b + c + d
    
    # 1. Calculate hits expected by chance (a_ref)
    # (Total Forecast Yes * Total Observed Yes) / Total Events
    a_ref = float((a + b) * (a + c)) / n
    
    # 2. Calculate ETS
    numerator = a - a_ref
    denominator = a + b + c - a_ref
    
    # Extreme case handling
    if denominator == 0:
        return np.nan
        
    ets = numerator / denominator
    
    return ets


def plot_multi_period_performance_matrix(results_high, results_low, period_names, model_style_map):
    """
    Generates a 4x2 matrix performance diagram matching an A4 page layout.
    
    Left Column (Col 0): Higher-than-fog threshold evaluation with detailed inset zooms.
    Right Column (Col 1): Low visibility threshold evaluation across full scale.
    
    Rows correspond to time windows: Row 0 is Entire Period; Rows 1-3 are sub-periods.
    """

    # Create markers for different data halves
    my_marker_1 = get_text_marker("A")
    my_marker_2 = get_text_marker("B")

    # 1. Setup Data Meshgrids for Background CSI
    x = np.linspace(0.001, 1, 100)
    y = np.linspace(0.001, 1, 100)
    SR_grid, POD_grid = np.meshgrid(x, y)
    CSI = 1 / (1/SR_grid + 1/POD_grid - 1)
    grid_data = (SR_grid, POD_grid, CSI)

    fig, axs = plt.subplots(4, 3, figsize=(17, 25), 
                            gridspec_kw={'width_ratios': [1, 1, 0.05]})
    contour_mappable = None
    plot_axs = axs[:, :2]   # 4x2 array for your performance plots
    cbar_axs = axs[:, 2]

    # Hardcoded Inset Axis Windows Config for Column 0 (Higher than Threshold)
    inset_configs_high = {
        0: {'bounds': [0.08, 0.18, 0.50, 0.56], 'xlim': [0.79, 1.0], 'ylim': [0.837, 1.01]},
        1: {'bounds': [0.08, 0.15, 0.70, 0.30], 'xlim': [0.38, 0.64], 'ylim': [0.84, 1.02]},
        2: {'bounds': [0.08, 0.15, 0.45, 0.62], 'xlim': [0.85, 1.01], 'ylim': [0.7, 1.02]},
        3: {'bounds': [0.08, 0.15, 0.35, 0.46], 'xlim': [0.85, 1.01], 'ylim': [0.93, 1.005]}
    }

    inset_configs_low = {
        0: {'bounds': None,                     'xlim': [0,0],         'ylim': [0,0]},
        1: {'bounds': [0.08, 0.15, 0.30, 0.70], 'xlim': [0.80,  1.02],  'ylim': [0.23, 0.75]},
        2: {'bounds': [0.55, 0.15, 0.37, 0.64], 'xlim': [-0.015, 0.25],   'ylim': [-0.015, 0.5]},
        3: {'bounds': None,                     'xlim': [0,0],         'ylim': [0,0]}
    }

    # FIX 1 & 2: Safe reordering of both data streams and period strings
    reordered_high = [results_high[-1]] + results_high[:-1]
    reordered_low  = [results_low[-1]] + results_low[:-1]
    reordered_periods = [period_names[-1]] + period_names[:-1]

    data_sources = [reordered_high, reordered_low]
    panel_labels = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]

    for i, p_name in enumerate(reordered_periods):
        for j in range(2):
            ax = plot_axs[i, j]
            is_high_thresh_col = (j == 0)
            
            # Extract metrics using the unified, safely reordered streams
            results = data_sources[j][i] 
            splits = results['splits']

            # Highlight Row 0 (Entire Cruise Summary) with prominent thick borders across both columns
            if i == 0:
                for spine in ax.spines.values():
                    spine.set_linewidth(1.8)       
                    spine.set_color('black')       
                    spine.set_zorder(10)           
            
            # FIX 3: Point to correct internal background-drawing helper function name
            contour_mappable = draw_perf_background(ax, grid_data, line_w=0.8, line_alpha=0.5, contour_alpha=0.2, show_text=True)

            # Generate and configure Inset Axis ONLY for the left column
            target_axs = [ax]
            cfg = inset_configs_high[i] if is_high_thresh_col else inset_configs_low[i]
            if cfg["bounds"] is not None:
                ax_ins = ax.inset_axes(cfg['bounds'])
                ax_ins.set_xlim(cfg['xlim'])
                ax_ins.set_ylim(cfg['ylim'])
                ax_ins.set_aspect('auto')
                if cfg['bounds'][2]>cfg['bounds'][2]*2:
                    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=3))
                elif cfg['bounds'][2] < cfg['bounds'][2]/2:
                    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=3))
                else: 
                    ax_ins.xaxis.set_major_locator(MaxNLocator(nbins=4))
                    ax_ins.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax_ins.tick_params(axis='both', which='major', labelsize=8)
                # ax_ins.axvspan(cfg['xlim'][0], cfg['xlim'][1], color='yellow', alpha=0.08, zorder=1)
                draw_hatching(ax_ins)
                draw_perf_background(ax_ins, grid_data, line_w=0.6, line_alpha=0.3, contour_alpha=0.15, show_text=False)
                target_axs.append(ax_ins)
                ax.indicate_inset_zoom(ax_ins, edgecolor="grey",alpha=1,lw=0.7)

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

                        # Success Ratio (1 - FAR) vs POD coordinates
                        pt_full = (1 - row_full['FAR'], row_full['POD'])
                        pt_1st  = (1 - row_1st['FAR'],  row_1st['POD'])
                        pt_2nd  = (1 - row_2nd['FAR'],  row_2nd['POD'])

                        # Marker configuration rules based on scenario types
                        mrkr = "*" if model_name == "Persist_10min" else "o"
                        sz = 180 if model_name == "Persist_10min" else 120

                        for t_ax in target_axs:
                            # First Half Window marker
                            t_ax.scatter(*pt_1st, s=sz*2.2, c=color, marker=my_marker_1, edgecolor=color, zorder=4, alpha=1)
                            # Second Half Window marker with transparency fade
                            t_ax.scatter(*pt_2nd, s=sz*2.2, c=color, marker=my_marker_2, edgecolor=color, zorder=4, alpha=1)
                            # Central Full Window reference point
                            t_ax.scatter(*pt_full, s=sz, c=color, marker=mrkr, edgecolor='black', zorder=5)
                            # Lead-time evolution connection track
                            t_ax.plot([pt_1st[0], pt_full[0], pt_2nd[0]], [pt_1st[1], pt_full[1], pt_2nd[1]], 
                                      color=color, linestyle='-', linewidth=1.2, alpha=0.4, zorder=3)

            # Distinct Subplot Titles indicating column metrics
            col_suffix = " (Windows of opportunity)" if is_high_thresh_col else " (Low visibility events)"
            ax.set_title(f"{p_name}{col_suffix}", pad=20, fontweight='bold' if i == 0 else 'normal')
            
            # Panel indexing indicators
            p_letter = panel_labels[i][j]
            ax.text(0.05, 0.95, f"{p_letter})", transform=ax.transAxes, fontsize=14, fontweight='bold', 
                    va='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=1))
            
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(True, linestyle=':', alpha=0.4)

    # 2. Clean Matrix Edge Axis Labels (Prevents visual clipping and inner crowding)
    for j in range(2):
        axs[3, j].set_xlabel('Success Ratio (1 - FAR)', fontsize=13)
    for i in range(4):
        axs[i, 0].set_ylabel('Probability of Detection (POD)', fontsize=13)

    # 3. Consolidated Global Legend on Top Left Panel
    legend_elements = []
    for l, c in model_style_map.items():
        mrkr = '*' if l == "Persist_10min" else 'o'
        legend_elements.append(Line2D([0], [0], color="none", markerfacecolor=c, label=l, marker=mrkr, markeredgecolor='black', markersize=10))
    
    legend_elements.extend([
        Line2D([0], [0], color='none', marker='o', markerfacecolor='white', markeredgecolor='black', label='Full Window Metric', markersize=10),
        Line2D([0], [0], color='none', marker=my_marker_1, markerfacecolor='white', markeredgecolor='black', label='First Half Window (A)', markersize=11, alpha=1),
        Line2D([0], [0], color='none', marker=my_marker_2, markerfacecolor='white', markeredgecolor='black', label='Second Half Window (B)', markersize=11, alpha=1),
    ])
    axs[0,0].legend(handles=legend_elements, frameon=True, loc='lower right', prop={'size': 8}, ncols=4)

    # 4. Vertical Colorbars aligned cleanly in the 3rd column
    for i in range(4):
        # Directly put the colorbar in the dedicated ax slot
        fig.colorbar(contour_mappable, cax=cbar_axs[i]).set_label('CSI', fontsize=10)
        
    # Set labels on the correct outer edge axes
    for j in range(2):
        plot_axs[3, j].set_xlabel('Success Ratio (1 - FAR)', fontsize=13)
    for i in range(4):
        plot_axs[i, 0].set_ylabel('Probability of Detection (POD)', fontsize=13)
    
    plt.tight_layout()
    return fig, plot_axs

# ================ #
# HELPER FUNCTIONS #
# ================ #

def draw_perf_background(ax, grid_data, line_w, line_alpha, contour_alpha, show_text=False):
    """Draws the CSI contours and dashed Bias lines on a given axis with an opaque background."""
    SR_grid, POD_grid, CSI = grid_data
    
    white_mask = plt.Rectangle((0, 0), 1, 1, transform=ax.transData, 
                               color='white', clip_on=True,zorder=2)
    ax.add_patch(white_mask)
    
    # 1. Contour Fill
    # Ensure zorder is higher than the mask (e.g., 6)
    mappable = ax.contourf(SR_grid, POD_grid, CSI, 
                           levels=np.arange(0, 1.1, 0.1), 
                           cmap='Greys', 
                           alpha=contour_alpha,zorder=3)
    
    # 2. Bias Lines
    # Ensure lines are on top of the contour (e.g., zorder=7)
    for b in [0.5, 0.75, 1, 1.25, 1.5, 2, 4]:
        end_x, end_y = (1, b) if b <= 1 else (1/b, 1)
        ax.plot([0, end_x], [0, end_y], color='gray', linestyle='--', 
                linewidth=line_w, alpha=line_alpha)
        
        if show_text:
            if b <= 1:
                # Labels on the right boundary (X=1)
                # Use x=1.02 to push it slightly outside the right spine
                ax.text(1.01, end_y, f' B={b}', 
                        transform=ax.transData,
                        fontsize=10, alpha=0.7, 
                        ha='left', va='center', 
                        clip_on=False, zorder=8)
            else:
                # Labels on the top boundary (Y=1)
                # Use y=1.02 to push it slightly above the top spine
                ax.text(end_x, 1.01, f' B={b}', 
                        transform=ax.transData,
                        fontsize=10, alpha=0.7, 
                        ha='center', va='bottom', 
                        clip_on=False, zorder=8)
                    
    return mappable

def draw_hatching(ax):
    """
    Creates a hatched buffer region beyond the [0, 1] domain.
    """
    # Create the hatched regions for the four sides beyond the [0, 1] box
    hatch_style = '///' 
    color = 'lightgray'
    
    # Left buffer
    ax.axvspan(-0.1, 0, facecolor='none', hatch=hatch_style, edgecolor=color, zorder=1)
    # Right buffer
    ax.axvspan(1, 1.1, facecolor='none', hatch=hatch_style, edgecolor=color, zorder=1)
    # Bottom buffer
    ax.axhspan(-0.1, 0, facecolor='none', hatch=hatch_style, edgecolor=color, zorder=1)
    # Top buffer
    ax.axhspan(1, 1.1, facecolor='none', hatch=hatch_style, edgecolor=color, zorder=1)

def get_text_marker(text, size=20):
    # Create a TextPath object
    fp = FontProperties(family='sans-serif',style="normal")
    return TextPath((0, 0), text, size=size, prop=fp)

# ============================== #
# LEGACY FUNCTIONS (not in main) #
# ============================== #

def assign_event_probabilities(df, fog_thresh, higher_than_fog_thresh):
    """
    Maps TAF categorical trends to numerical event probabilities $P(\text{Vis} >< fog_{thresh})$.
    Priority follows: Main (100%) > PROB40 (40%) > PROB30 (30%) > TEMPO (10%).
    Useful if a probability-based approach is used instead of strict categorical bins and best/worst/base
    visibility scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        TAF DataFrame with scenario columns.
    fog_thresh : float 
        The visibility threshold (km) defining the "event" (e.g., fog), by default 1.0.
    higher_than_fog_thresh : bool
        If True, the "event" is defined as visibility > fog_thresh (opportunity). 
        If False, the "event" is defined as visibility <= fog_thresh (hazard), by default False.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the 'p_event' column added.
    """

    # Initialize based on the Main Scenario
    if higher_than_fog_thresh:
        # Event is "Clear": Probability is 1.0 if main > threshold, else 0.0
        df['p_event'] = (df['main_scenario'] > fog_thresh).astype(float)
    else:
        # Event is "Fog": Probability is 1.0 if main <= threshold, else 0.0
        df['p_event'] = (df['main_scenario'] <= fog_thresh).astype(float)

    # Process Trends (TEMPO, PROB30, PROB40)
    for col, weight in [('tempo', 0.1), ('prob30', 0.3), ('prob40', 0.4)]:
        mask_trend_exists = df[col].notna()
        
        if higher_than_fog_thresh:
            mask_trend_is_clear = df[col] > fog_thresh
            mask_trend_is_fog = df[col] <= fog_thresh
            
            # If main was Fog (prob 0) but trend is Clear, ADD probability
            df.loc[mask_trend_exists & mask_trend_is_clear & (df['main_scenario'] <= fog_thresh), 'p_event'] += weight
            # If main was Clear (prob 1) but trend is Fog, SUBTRACT probability
            df.loc[mask_trend_exists & mask_trend_is_fog & (df['main_scenario'] > fog_thresh), 'p_event'] -= weight
            
        else:
            mask_trend_is_fog = df[col] <= fog_thresh
            mask_trend_is_clear = df[col] > fog_thresh
            
            # If main was Clear (prob 0) but trend is Fog, ADD probability
            df.loc[mask_trend_exists & mask_trend_is_fog & (df['main_scenario'] > fog_thresh), 'p_event'] += weight
            # If main was Fog (prob 1) but trend is Clear, SUBTRACT probability
            df.loc[mask_trend_exists & mask_trend_is_clear & (df['main_scenario'] <= fog_thresh), 'p_event'] -= weight

    # Ensure probabilities stay within [0, 1]
    df['p_event'] = df['p_event'].clip(0.0, 1.0)
    
    # Invalidate where no TAF exists
    df.loc[df['is_valid'] == False, 'p_event'] = np.nan
    return df

def plot_ens_meteogram(prob_df, model_dict, vis_obs, start_date, end_date, resample_freq='3H'):
    """
    Plots TAF probabilities as a stacked bar chart with model visibility rows below.

    Parameters
    ----------
    prob_df : pd.DataFrame
        DataFrame indexed by time with columns for each visibility bin.
        Each cell contains a probability (0.0 to 1.0) for that bin.
    model_dict : dict
        Dictionary mapping model names to visibility series { 'ModelName': pd.Series(vis_km) }.
    vis_obs : pd.Series
        Observed visibility time series (km).
    start_date : str or pd.Timestamp
        Start date for the plot window.
    end_date : str or pd.Timestamp
        End date for the plot window.
    resample_freq : str 
        Resampling frequency for aggregating probabilities, by default '3H'.

    Returns
    -------
    None
        Displays matplotlib figure with meteogram
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

def plot_taf_window(df, fog_thresh, start_time, end_time):
    """
    Plots the Main, Best, and Worst TAF scenarios for a specific time window.
    
    Useful for detailed case studies of specific fog events.

    Parameters
    ----------
    df : pd.DataFrame
        TAF DataFrame containing 'best_vis', 'worst_vis', and 'main_vis' columns.
    fog_thresh : float
        Visibility threshold for fog definition (km).
    start_time : str or pd.Timestamp
        Start time for the plot window.
    end_time : str or pd.Timestamp
        End time for the plot window.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the TAF scenarios plot.
    ax : matplotlib.axes.Axes
        Axes object for further customization.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
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

    Parameters
    ----------
    df : pd.DataFrame
        TAF DataFrame containing 'base', 'tempo', and 'becmg' columns.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the TAF components plot.
    ax : matplotlib.axes.Axes
        Axes object for further customization.
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

def plot_ensemble_spaghetti(ens_xr, obs_series, start_t, end_t, fog_thresh):
    """
    Plots individual ensemble members as 'spaghetti' lines to visualize forecast spread
    and uncertainty relative to observations.

    Parameters
    ----------
    ens_xr : xarray.DataArray
        Ensemble visibility data with dimensions (time, number).
        Each 'number' represents an individual ensemble member.
    obs_series : pd.Series
        Observed visibility time series (km), indexed by datetime.
    start_t : str or pd.Timestamp
        Start time for the plot window.
    end_t : str or pd.Timestamp
        End time for the plot window.
    fog_thresh : float 
        Visibility threshold for fog definition (km), by default 0.8.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the spaghetti plot.
    ax : matplotlib.axes.Axes
        Axes object for further customization.
    """
    
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
    ax.axhline(fog_thresh, color='black', ls='--', alpha=0.6, label='Fog Threshold')
    
    ax.set_ylim(0, 10.5) # Focus on the low-visibility range
    ax.set_ylabel('Visibility [km]')
    ax.set_title(f'Ensemble Spread vs. Observations ({start_t} to {end_t})')
    ax.legend()
    plt.tight_layout()
    return fig, ax

def plot_visibility_pdfs_cdfs(ds_obs, time_vec, periods, quant_vars, fog_thresh):
    """
    Generate probability density function (PDF) and cumulative distribution function (CDF) plots for visibility data.
    This function creates a 2x3 subplot grid displaying PDFs in the top row and empirical CDFs (ECDFs) 
    in the bottom row for different time periods. The plots highlight fog conditions based on a specified 
    visibility threshold.

    Parameters
    ----------
    ds_obs : xarray.Dataset
        Dataset containing observational visibility data with variables corresponding to quant_vars.
    time_vec : pandas.DatetimeIndex
        Time vector for reindexing the data to align observations with a common temporal grid.
    periods : list of tuple
        List of tuples containing ((start_time, end_time), period_name) pairs defining the time periods 
        to analyze and their descriptive names.
    quant_vars : list of str
        List of variable names representing different visibility measurement sources or types. 
        The first 4 variables are plotted in the PDFs and CDFs.
    fog_thresh : float
        Visibility threshold (in km) below which conditions are considered foggy. Used to highlight 
        the fog zone and set x-axis limits for the ECDF plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the reliability diagram.
    axs : matplotlib.axes.Axes
        Axes object for further customization.
    """
    raw_data = {}
    for v in quant_vars:
        raw_data[v] = ds_obs[v].to_series().reindex(time_vec, method='nearest', tolerance='5min') * 1e-3
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs1 = axs[0, :] 
    axs2 = axs[1, :] 
    
    for i, (bounds, period_name) in enumerate(periods):    
        p_start, p_end = bounds
        
        for var_name, ls in zip(quant_vars[:4], ["-", "--", ":", "-."]):
            # 1. Extract period subset
            subset = raw_data[var_name].loc[p_start:p_end].dropna()
            if subset.empty:
                continue

            # --- PDF Plotting ---
            sns.histplot(subset, stat="density", element="poly", label=var_name, 
                         bins=50, kde=False, fill=False, linestyle=ls, ax=axs1[i])
            
            # --- CDF Plotting ---
            # Sort full subset to get the true ECDF of the period
            x_sorted = np.sort(subset.values)
            # Probability P(X <= x)
            y_values = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            
            axs2[i].plot(x_sorted, y_values, label=var_name, linestyle=ls, 
                         marker='.', markersize=6, alpha=0.6, lw=0.5)

        axs1[i].set_title(f"Visibility PDF: {period_name}")
        axs1[i].set_xlim(0, 20)
        axs1[i].axvspan(0, fog_thresh, color='yellow', alpha=0.2, label='Fog Zone')
        axs1[i].axvline(fog_thresh, color='k', linestyle=':', alpha=0.5, label='Fog Threshold')
        axs2[i].set_title(f"Visibility ECDF (Fog Zoom): {period_name}")
        axs2[i].set_xlabel("Visibility (km)")
        axs2[i].set_ylabel("Cumulative Probability")
        
        axs2[i].set_xlim(0, fog_thresh * 1.5) 
        axs2[i].grid(True, which='both', alpha=0.3)
        axs2[i].axvspan(0, fog_thresh, color='yellow', alpha=0.1)
        axs2[i].axvline(fog_thresh, color='k', linestyle=':', alpha=0.5, label='Fog Threshold')

    axs1[0].legend()
    axs2[0].legend()
    plt.tight_layout()
    return fig,axs

def plot_vis_summary(df, vis_obs, vis_mod1, vis_mod2, fog_thresh, start_date=None, end_date=None):
    """
    Plots a log-scale time series comparison of TAF scenarios, 
    model output, and observations.

    Parameters
    ----------
    df : pd.DataFrame
        TAF DataFrame containing 'worst_vis', 'best_vis', 'main_scenario', and 'is_valid' columns.
    vis_obs : pd.Series
        Observed visibility time series (km).
    vis_mod1 : pd.Series
        First model visibility time series (km).
    vis_mod2 : pd.Series
        Second model visibility time series (km).
    fog_thresh : float
        Visibility threshold for fog definition (km).
    start_date : str or pd.Timestamp 
        Start date for the plot window. If None, uses full time range.
    end_date : str or pd.Timestamp 
        End date for the plot window. If None, uses full time range.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the log-scale visibility comparison plot.
    ax : matplotlib.axes.Axes
        Axes object for further customization.
    """
    obs_series = vis_obs.reindex(df.index, method='nearest')
    mod_series1 = vis_mod1.reindex(df.index, method='nearest')
    mod_series2 = vis_mod2.reindex(df.index, method='nearest')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if start_date and end_date:
        plot_df = df.loc[start_date:end_date]
        plot_obs = obs_series.loc[start_date:end_date]
        plot_mod1 = mod_series1.loc[start_date:end_date]
        plot_mod2 = mod_series2.loc[start_date:end_date]
    else:
        plot_df, plot_obs, plot_mod1, plot_mod2 = df, obs_series, mod_series1, mod_series2

    ax.fill_between(plot_df.index, plot_df['worst_vis'], plot_df['best_vis'], 
                    color='lightgray', alpha=0.5, label='TAF Uncertainty (TEMPO/PROB)')
    ax.plot(plot_df.index, plot_df['main_scenario'], color='black', linewidth=1.2, 
            label='TAF Main (Base/BECMG)', marker="o")
    ax.plot(plot_df.index, plot_mod1, color='blue', linestyle='--', linewidth=1.5, 
            label='IFS Oper. model')
    # ax.plot(plot_df.index, plot_mod2, color='green', linestyle='--', linewidth=1.5, 
    #         label='IFS lowLvlMean model')
    ax.plot(plot_df.index, plot_obs, color='crimson', linewidth=2, label='Oden Observations')

    ax.set_yscale('log')
    ax.set_ylim(0.04, 15)
    y_ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(t) for t in y_ticks])
    ax.axhline(y=fog_thresh, color='red', linestyle=':', alpha=0.5, label='Fog Threshold (0.8 km)')
    ax.set_ylabel('Visibility [km] (Log Scale)')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax.fill_between(plot_df.index, 0, 1, where=~plot_df['is_valid'], 
                    color='yellow', alpha=0.1, transform=ax.get_xaxis_transform(), label='No TAF')

    ax.legend(loc='lower right', frameon=True, fontsize='small', ncol=2)
    plt.tight_layout()
    return fig, ax


def plot_multi_period_performance(results_list, period_names, model_style_map, higher_than_fog_thresh):
    """
    Generates a 2x2 performance diagram comparing models and dual forecaster thresholds 
    across multiple time periods, using both 5-min and 15-min observations.

    Parameters
    ----------
    results_list : list of dict
        List containing dictionaries of metrics for each period.
    period_names : list of str
        Titles for each subplot (e.g., ['Period 1', 'Period 2', ...]).
    model_style_map : dict
        Mapping of model names to their styling colors.
    higher_than_fog_thresh : bool
        Flag indicating whether to evaluate higher-than-fog threshold.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the performance diagram.
    axs : matplotlib.axes.Axes
        Axes object for further customization.
    """

    # Create markers for different data halves
    my_marker_1 = get_text_marker("1")
    my_marker_2 = get_text_marker("2")

    # 1. Setup Data Meshgrids for Background CSI
    x = np.linspace(0.001, 1, 100)
    y = np.linspace(0.001, 1, 100)
    SR_grid, POD_grid = np.meshgrid(x, y)
    CSI = 1 / (1/SR_grid + 1/POD_grid - 1)
    grid_data = (SR_grid, POD_grid, CSI)

    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    axs = axs.flatten()
    contour_mappable = None

    # Hardcoded Inset Axis Windows Config
    inset_configs = {
        0: {'bounds': [0.08, 0.15, 0.50, 0.56], 'xlim': [0.75, 1.0], 'ylim': [0.8, 1.0]},
        1: {'bounds': [0.08, 0.15, 0.70, 0.27], 'xlim': [0.2,  0.9], 'ylim': [0.8, 1.0]},
        2: {'bounds': [0.08, 0.15, 0.45, 0.62], 'xlim': [0.85, 1.0], 'ylim': [0.7, 1.0]},
        3: {'bounds': [0.08, 0.15, 0.35, 0.46], 'xlim': [0.8,  1.0], 'ylim': [0.9, 1.0]}
    }

    # Reorder layout sequence: Put entire period first and then three separate sub-periods
    reordered_results = [results_list[-1]] + results_list[:-1]
    reordered_periods = [period_names[-1]] + period_names[:-1]

    for i, (period_data, p_name) in enumerate(zip(reordered_results, reordered_periods)):
        ax = axs[i]

        # Style first subplot panel with prominent borders
        if i == 0:
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)       
                spine.set_color('black')       
                spine.set_zorder(10)           
        
        # Draw Main Panel Background
        contour_mappable = draw_perf_background(ax, grid_data, line_w=0.8, line_alpha=0.5, contour_alpha=0.2, show_text=True)

        # Generate Inset Axis if requested
        target_axs = [ax]
        if higher_than_fog_thresh:
            cfg = inset_configs[i]
            ax_ins = ax.inset_axes(cfg['bounds'])
            ax_ins.set_xlim(cfg['xlim'])
            ax_ins.set_ylim(cfg['ylim'])
            ax_ins.set_aspect('auto')
            ax_ins.set_facecolor('white')
            ax_ins.tick_params(axis='both', which='major', labelsize=6)
            
            # Draw Inset Background
            draw_perf_background(ax_ins, grid_data, line_w=0.6, line_alpha=0.3, contour_alpha=0.15, show_text=False)
            target_axs.append(ax_ins)

        # --- UNIFORM TRAJECTORY PLOTTING BLOCK ---
        splits = period_data['splits']
        
        # Verify all necessary sub-period slices are verified in data structures
        if all(k in splits for k in ['Full', 'First_Half', 'Second_Half']):
            df_full = splits['Full']
            df_1st  = splits['First_Half']
            df_2nd  = splits['Second_Half']

            for model_name, color in model_style_map.items():
                # Verify identity existence inside current period split metrics
                if model_name in df_full.index and model_name in df_1st.index and model_name in df_2nd.index:
                    row_full = df_full.loc[model_name]
                    row_1st  = df_1st.loc[model_name]
                    row_2nd  = df_2nd.loc[model_name]

                    # Parse coordinates: Success Ratio (1 - FAR) vs Probability of Detection (POD)
                    pt_full = (1 - row_full['FAR'], row_full['POD'])
                    pt_1st  = (1 - row_1st['FAR'],  row_1st['POD'])
                    pt_2nd  = (1 - row_2nd['FAR'],  row_2nd['POD'])

                    # Set marker logic based on model type
                    mrkr = "*" if model_name == "Persist_10min" else "o"
                    sz = 180 if model_name == "Persist_10min" else 120

                    for t_ax in target_axs:
                        # First Half Point (Leftward Triangle)
                        t_ax.scatter(*pt_1st, s=sz, c=color, marker=my_marker_1, edgecolor='black', zorder=4, alpha=0.7)
                        
                        # Second Half Point (Rightward Triangle)
                        t_ax.scatter(*pt_2nd, s=sz, c=color, marker=my_marker_2, edgecolor='black', zorder=4, alpha=0.3)
                        
                        # Full Window Frame Point (Central Marker Circle/Star)
                        t_ax.scatter(*pt_full, s=sz, c=color, marker=mrkr, edgecolor='black', zorder=5)

                        # Draw the evolution trajectory connection line
                        t_ax.plot([pt_1st[0], pt_full[0], pt_2nd[0]], [pt_1st[1], pt_full[1], pt_2nd[1]], 
                                  color=color, linestyle='-', linewidth=1.2, alpha=0.4, zorder=3)

        # Labels, layout, and panel annotations
        ax.set_title(p_name, pad=20, fontweight='bold')
        ax.text(0.05, 0.95, f"{chr(97+i)})", transform=ax.transAxes, fontsize=14, fontweight='bold', 
                va='top', bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=1))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, linestyle=':', alpha=0.4)

    # 2. Global Grid Labels Configuration
    axs[2].set_xlabel('Success Ratio (1 - FAR)', fontsize=14)
    axs[3].set_xlabel('Success Ratio (1 - FAR)', fontsize=14)
    axs[0].set_ylabel('Probability of Detection (POD)', fontsize=14)
    axs[2].set_ylabel('Probability of Detection (POD)', fontsize=14)

    # 3. Global Legend Reconstruction
    legend_elements = []
    
    # Track models and scenario colors
    for l, c in model_style_map.items():
        mrkr = '*' if l == "Persist_10min" else 'o'
        legend_elements.append(Line2D([0], [0], color="none", markerfacecolor=c, label=l, marker=mrkr, markeredgecolor='black', markersize=10))
    
    # Trace/Time evolution guides
    legend_elements.extend([
        Line2D([0], [0], color='none', marker=my_marker_1, markerfacecolor='gray', markeredgecolor='black', label='First Half Window', markersize=12, alpha=0.7),
        Line2D([0], [0], color='none', marker='o', markerfacecolor='gray', markeredgecolor='black', label='Full Window Metric', markersize=10),
        Line2D([0], [0], color='none', marker=my_marker_2, markerfacecolor='gray', markeredgecolor='black', label='Second Half Window', markersize=12, alpha=0.3),
        Line2D([0], [0], color='black', linestyle='-', linewidth=1.2, alpha=0.4, label='Lead-Time Trajectory')
    ])
    
    axs[0].legend(handles=legend_elements, frameon=True, loc='lower right', prop={'size': 7}, ncols=3)

    # 4. Axes Colorbars Setup
    for idx in [1, 3]:
        divider = make_axes_locatable(axs[idx])
        cax = divider.append_axes("right", size="5%", pad=0.6)
        fig.colorbar(contour_mappable, cax=cax).set_label('CSI', fontsize=10)

    plt.tight_layout()
    return fig, axs