import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from tqdm import tqdm



# ---------------- PRE-PLR DILATION -------------------


def _compute_pre_plr_dilation(
    trace,
    stim_onset_idx,
    onset_idx,
    sample_interval_ms,
    baseline_value=0,
    search_start_ms=100
):

    """
    Compute pre-PLR dilation amplitude.

    Definition
    ----------
    Maximum pupil dilation occurring between:

        stimulus onset + search_start_ms
        and
        constriction onset
    """

    # -------------------------------------------------
    # Invalid onset
    # -------------------------------------------------

    if onset_idx is None:

        return np.nan, None, np.nan

    if np.isnan(onset_idx):

        return np.nan, None, np.nan

    # -------------------------------------------------
    # Search window
    # -------------------------------------------------

    start_offset_samples = int(

        np.round(
            search_start_ms
            / sample_interval_ms
        )
    )

    search_start_idx = (

        stim_onset_idx
        + start_offset_samples
    )

    # -------------------------------------------------
    # Window validation
    # -------------------------------------------------

    if search_start_idx >= onset_idx:

        return np.nan, None, np.nan

    segment = trace[
        search_start_idx:onset_idx
    ]

    # -------------------------------------------------
    # NaN handling
    # -------------------------------------------------

    if len(segment) == 0:

        return np.nan, None, np.nan

    if np.all(np.isnan(segment)):

        return np.nan, None, np.nan

    # -------------------------------------------------
    # Peak detection
    # -------------------------------------------------

    try:

        max_idx_rel = np.nanargmax(
            segment
        )

    except Exception:

        return np.nan, None, np.nan

    dilation_idx = (

        search_start_idx
        + max_idx_rel
    )

    dilation_value = trace[
        dilation_idx
    ]

    # -------------------------------------------------
    # Amplitude
    # -------------------------------------------------

    dilation_amplitude = (

        dilation_value
        - baseline_value
    )

    # -------------------------------------------------
    # Negative amplitude rejection
    # -------------------------------------------------

    if dilation_amplitude < 0:

        return np.nan, None, np.nan

    return (

        dilation_amplitude,

        dilation_idx,

        dilation_value
    )


# =====================================================
# -------- CONSTRICTION ONSET : WILCOXON --------------
# =====================================================

def _detect_constriction_onset_wilcoxon(
    trace,
    start_idx,
    sample_interval_ms,
    window_ms=100,
    step_ms=20,
    min_latency_ms=150,
    alpha=0.05
):

    """
    Detect constriction onset using
    sliding Wilcoxon windows.
    """

    # =====================================================
    # ---------------- CONVERT TO SAMPLES -----------------
    # =====================================================

    window_samples = int(

        np.round(
            window_ms
            / sample_interval_ms
        )
    )

    step_samples = int(

        np.round(
            step_ms
            / sample_interval_ms
        )
    )

    min_latency_samples = int(

        np.round(
            min_latency_ms
            / sample_interval_ms
        )
    )

    # =====================================================
    # ---------------- SEARCH START -----------------------
    # =====================================================

    search_start = (
        start_idx
        + min_latency_samples
    )

    # =====================================================
    # ---------------- MAIN SEARCH ------------------------
    # =====================================================

    for i in range(

        search_start,

        len(trace)
        - (2 * window_samples),

        step_samples
    ):

        # -------------------------------------------------
        # Consecutive windows
        # -------------------------------------------------

        w1 = trace[
            i :
            i + window_samples
        ]

        w2 = trace[
            i + window_samples :
            i + (2 * window_samples)
        ]

        # -------------------------------------------------
        # NaN check
        # -------------------------------------------------

        if (
            np.any(np.isnan(w1))
            or
            np.any(np.isnan(w2))
        ):
            continue

        # -------------------------------------------------
        # Wilcoxon
        # -------------------------------------------------

        try:

            stat, p = wilcoxon(
                w1,
                w2
            )

        except Exception:

            continue

        # -------------------------------------------------
        # Constriction criterion
        # -------------------------------------------------

        if (

            p < alpha

            and

            np.nanmean(w2)
            < np.nanmean(w1)

        ):

            onset_idx = i

            onset_time_ms = (

                onset_idx
                - start_idx

            ) * sample_interval_ms

            return (

                onset_time_ms,

                onset_idx
            )

    return np.nan, None


# =====================================================
# ---------- CONSTRICTION ONSET : SLOPE ---------------
# =====================================================

def _detect_constriction_onset_slope(
    trace,
    start_idx,
    sample_interval_ms,
    window_ms=100,
    step_ms=10,
    min_latency_ms=150
):

    """
    Detect constriction onset using
    sustained negative slopes.

    Method
    ------
    If slope remains negative for
    an entire window,
    constriction onset is detected.
    """

    # =====================================================
    # ---------------- CONVERT TO SAMPLES -----------------
    # =====================================================

    window_samples = int(

        np.round(
            window_ms
            / sample_interval_ms
        )
    )

    step_samples = int(

        np.round(
            step_ms
            / sample_interval_ms
        )
    )

    min_latency_samples = int(

        np.round(
            min_latency_ms
            / sample_interval_ms
        )
    )

    # =====================================================
    # ---------------- SEARCH START -----------------------
    # =====================================================

    search_start = (
        start_idx
        + min_latency_samples
    )

    # =====================================================
    # ---------------- MAIN SEARCH ------------------------
    # =====================================================

    for i in range(

        search_start,

        len(trace)
        - window_samples,

        step_samples
    ):

        segment = trace[
            i :
            i + window_samples
        ]

        # -------------------------------------------------
        # NaN check
        # -------------------------------------------------

        if np.any(np.isnan(segment)):

            continue

        # -------------------------------------------------
        # Local slopes
        # -------------------------------------------------

        slopes = np.diff(segment)

        # -------------------------------------------------
        # Sustained negative slope
        # -------------------------------------------------

        if np.all(slopes < 0):

            onset_idx = i

            onset_time_ms = (

                onset_idx
                - start_idx

            ) * sample_interval_ms

            return (

                onset_time_ms,

                onset_idx
            )

    return np.nan, None



# -------------- CONSTRICTION AMPLITUDE ---------------


def _compute_constriction_amplitude(
    trace,
    baseline_value,
    search_start_idx
):

    """
    Compute constriction amplitude.
    """

    segment = trace[search_start_idx:]

    if np.all(np.isnan(segment)):

        return np.nan, np.nan, None

    min_idx_rel = np.nanargmin(
        segment
    )

    min_idx = (

        search_start_idx
        + min_idx_rel
    )

    min_value = trace[min_idx]

    amplitude = (

        baseline_value
        - min_value
    )

    return (

        amplitude,

        min_value,

        min_idx
    )



# ---------------- CONSTRICTION RATE ------------------


def _compute_constriction_rate(
    trace,
    onset_idx,
    min_idx,
    sample_interval_ms
):

    """
    Compute constriction slope.
    """

    if onset_idx is None:

        return np.nan

    if onset_idx >= min_idx:

        return np.nan

    y = trace[
        onset_idx:min_idx
    ]

    if np.sum(~np.isnan(y)) < 3:

        return np.nan

    x = (

        np.arange(onset_idx, min_idx)

        * sample_interval_ms
    )

    slope = np.polyfit(

        x[~np.isnan(y)],

        y[~np.isnan(y)],

        1

    )[0]

    return slope



# ---------------- LATENCY ----------------------------


def _compute_latency(
    min_idx,
    stim_onset_idx,
    sample_interval_ms
):

    """
    Compute constriction latency.
    """

    if min_idx is None:

        return np.nan

    latency_ms = (

        min_idx
        - stim_onset_idx

    ) * sample_interval_ms

    return latency_ms



# ---------------- DILATION RATE ----------------------


def _compute_dilation_rate(
    trace,
    min_idx,
    sample_interval_ms,
    dilation_window_ms
):

    """
    Compute recovery dilation slope.
    """

    if min_idx is None:

        return np.nan

    window_samples = int(

        np.round(
            dilation_window_ms
            / sample_interval_ms
        )
    )

    end_idx = min(

        min_idx + window_samples,

        len(trace)
    )

    y = trace[min_idx:end_idx]

    if np.sum(~np.isnan(y)) < 3:

        return np.nan

    x = (

        np.arange(min_idx, end_idx)

        * sample_interval_ms
    )

    slope = np.polyfit(

        x[~np.isnan(y)],

        y[~np.isnan(y)],

        1

    )[0]

    return slope



# ---------------- MAIN FUNCTION ----------------------


def decompose_pupil(
    prepared_data,
    sample_interval_ms,
    baseline_window,
    stim_onset_idx,
    dilation_window_ms=300,
    onset_method="wilcoxon"
):

    """
    Main decomposition pipeline.
    """

    traces = prepared_data["traces"]

    participants = prepared_data["participants"]

    conditions = prepared_data["conditions"]

    baseline_start = baseline_window[0]

    baseline_end = baseline_window[1]

    feature_rows = []

    success_trials = []

    failed_trials = []

    total_trials = len(traces)

    print("\n========================================")
    print("FEATURE DECOMPOSITION")
    print("========================================\n")

    print(f"Total trials : {total_trials}")

  
    # ---------------- MAIN LOOP --------------------------


    for i in tqdm(

        range(total_trials),

        desc="Extracting features"
    ):

        trace = traces[i]

        participant = participants[i]

        condition = conditions[i]

        valid_trial = True

        failure_reason = None

        # -------------------------------------------------
        # Baseline
        # -------------------------------------------------

        baseline_value = np.nanmean(

            trace[
                baseline_start:baseline_end
            ]
        )

        # -------------------------------------------------
        # Constriction onset
        # -------------------------------------------------

        if onset_method == "wilcoxon":

            (
                onset_time_ms,
                onset_idx

            ) = _detect_constriction_onset_wilcoxon(

                trace,

                start_idx=stim_onset_idx,

                sample_interval_ms=sample_interval_ms
            )

        elif onset_method == "slope":

            (
                onset_time_ms,
                onset_idx

            ) = _detect_constriction_onset_slope(

                trace,

                start_idx=stim_onset_idx,

                sample_interval_ms=sample_interval_ms
            )

        else:

            raise ValueError(

                "onset_method must be "
                "'wilcoxon' or 'slope'"
            )

        if onset_idx is None:

            valid_trial = False

            failure_reason = (
                "No constriction onset"
            )

        # -------------------------------------------------
        # Pre-PLR dilation
        # -------------------------------------------------

        (
            pre_plr_dilation,
            pre_plr_idx,
            pre_plr_value

        ) = _compute_pre_plr_dilation(

            trace,

            stim_onset_idx,

            onset_idx,

            sample_interval_ms,

            baseline_value=baseline_value
        )

        # -------------------------------------------------
        # Constriction amplitude
        # -------------------------------------------------

        (
            constriction_amplitude,
            min_value,
            min_idx

        ) = _compute_constriction_amplitude(

            trace,

            baseline_value,

            stim_onset_idx
        )

        # -------------------------------------------------
        # Latency
        # -------------------------------------------------

        latency_ms = _compute_latency(

            min_idx,

            stim_onset_idx,

            sample_interval_ms
        )

        # -------------------------------------------------
        # Constriction rate
        # -------------------------------------------------

        constriction_rate = (

            _compute_constriction_rate(

                trace,

                onset_idx,

                min_idx,

                sample_interval_ms
            )
        )

        # -------------------------------------------------
        # Dilation rate
        # -------------------------------------------------

        dilation_rate = (

            _compute_dilation_rate(

                trace,

                min_idx,

                sample_interval_ms,

                dilation_window_ms
            )
        )

        # -------------------------------------------------
        # Feature validation
        # -------------------------------------------------

        failed_features = []

        if np.isnan(onset_time_ms):

            failed_features.append(
                "Constriction_Onset"
            )

        if np.isnan(pre_plr_dilation):

            failed_features.append(
                "Pre_PLR_Dilation"
            )

        if np.isnan(constriction_amplitude):

            failed_features.append(
                "Constriction_Amplitude"
            )

        if np.isnan(latency_ms):

            failed_features.append(
                "Constriction_Latency"
            )

        if np.isnan(constriction_rate):

            failed_features.append(
                "Constriction_Rate"
            )

        if np.isnan(dilation_rate):

            failed_features.append(
                "Dilation_Rate"
            )

        # -------------------------------------------------
        # Trial validity
        # -------------------------------------------------

        if len(failed_features) > 0:

            valid_trial = False

            failure_reason = (
                ", ".join(failed_features)
            )

        # -------------------------------------------------
        # Success tracking
        # -------------------------------------------------

        if valid_trial:

            success_trials.append(i)

        else:

            failed_trials.append(i)

        # -------------------------------------------------
        # Save row
        # -------------------------------------------------

        row = {

            "Row_Index":
                i,

            "Participant":
                participant,

            "Condition":
                condition,

            "Valid_Trial":
                valid_trial,

            "Failure_Reason":
                failure_reason,

            # -----------------------------------------
            # FEATURES
            # -----------------------------------------

            "Constriction_Onset_ms":
                onset_time_ms,

            "Pre_PLR_Dilation":
                pre_plr_dilation,

            "Constriction_Amplitude":
                constriction_amplitude,

            "Constriction_Latency_ms":
                latency_ms,

            "Constriction_Rate":
                constriction_rate,

            "Dilation_Rate":
                dilation_rate,

            # -----------------------------------------
            # FEATURE INDICES
            # -----------------------------------------

            "Constriction_Onset_Index":
                onset_idx,

            "Constriction_Minimum_Index":
                min_idx,

            "Pre_PLR_Dilation_Index":
                pre_plr_idx,

            # -----------------------------------------
            # FEATURE VALUES
            # -----------------------------------------

            "Pre_PLR_Dilation_Value":
                pre_plr_value,

            "Constriction_Minimum_Value":
                min_value
        }

        feature_rows.append(row)

    
    # ---------------- DATAFRAME --------------------------
   

    features_df = pd.DataFrame(
        feature_rows
    )

    # ---------------- SUMMARY ----------------------------
  

    print("\n========================================")
    print("DECOMPOSITION SUMMARY")
    print("========================================\n")

    print(f"Total trials      : {total_trials}")

    print(
        f"Successful trials : "
        f"{len(success_trials)}"
    )

    print(
        f"Failed trials     : "
        f"{len(failed_trials)}"
    )

    print("\nSuccessful row indices:\n")

    print(success_trials)

    print("\nFailed trials summary:\n")

    failed_df = features_df.loc[
        features_df["Valid_Trial"] == False,
        ["Row_Index", "Failure_Reason"]
    ]

    print(failed_df)


    # ---------------- SAVE OUTPUT ------------------------
  

    prepared_data["features"] = features_df

    prepared_data["success_trials"] = (
        success_trials
    )

    prepared_data["failed_trials"] = (
        failed_trials
    )

    prepared_data["parameters"] = {

        "sample_interval_ms":
            sample_interval_ms,

        "baseline_window":
            baseline_window,

        "stim_onset_idx":
            stim_onset_idx,

        "dilation_window_ms":
            dilation_window_ms,

        "onset_method":
            onset_method
    }

    print("\nFeature extraction completed.\n")

    return prepared_data