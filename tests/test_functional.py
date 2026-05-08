
import os
import numpy as np
import pandas as pd

from pupildecompose import run_decompose



np.random.seed(123)


N_PARTICIPANTS = 5

N_TRIALS = 8

CONDITIONS = [
    "Condition_A",
    "Condition_B"
]

SAMPLING_RATE = 100

SAMPLE_INTERVAL_MS = 1000 / SAMPLING_RATE

TOTAL_TIME_MS = 2000

N_SAMPLES = int(
    TOTAL_TIME_MS
    / SAMPLE_INTERVAL_MS
)

STIM_ONSET_MS = 500

STIM_ONSET_IDX = int(
    STIM_ONSET_MS
    / SAMPLE_INTERVAL_MS
)


# ---------------- SYNTHETIC PLR ----------------------


def generate_pupil_trace(condition):

    """
    Generate canonical synthetic PLR.
    """

    time = np.arange(N_SAMPLES)

    # -------------------------------------------------
    # Baseline
    # -------------------------------------------------

    trace = np.ones(N_SAMPLES)

    # -------------------------------------------------
    # Pre-PLR dilation
    # -------------------------------------------------

    if condition == "Condition_A":

        pre_amp = 0.04

    else:

        pre_amp = 0.07

    pre_center = STIM_ONSET_IDX + 15

    pre_width = 5

    pre_dilation = (

        pre_amp

        * np.exp(
            -((time - pre_center) ** 2)
            / (2 * pre_width**2)
        )
    )

    # -------------------------------------------------
    # Constriction
    # -------------------------------------------------

    if condition == "Condition_A":

        constriction_amp = -0.45

    else:

        constriction_amp = -0.60

    constriction_center = STIM_ONSET_IDX + 35

    constriction_width = 8

    constriction = (

        constriction_amp

        * np.exp(
            -((time - constriction_center) ** 2)
            / (2 * constriction_width**2)
        )
    )

    # -------------------------------------------------
    # Recovery dilation
    # -------------------------------------------------

    if condition == "Condition_A":

        recovery_amp = 0.18

    else:

        recovery_amp = 0.25

    recovery_center = STIM_ONSET_IDX + 90

    recovery_width = 25

    recovery = (

        recovery_amp

        * np.exp(
            -((time - recovery_center) ** 2)
            / (2 * recovery_width**2)
        )
    )

    # -------------------------------------------------
    # Combine
    # -------------------------------------------------

    trace = (
        trace
        + pre_dilation
        + constriction
        + recovery
    )

    # -------------------------------------------------
    # Noise
    # -------------------------------------------------

    noise = np.random.normal(
        0,
        0.008,
        N_SAMPLES
    )

    trace = trace + noise

    # -------------------------------------------------
    # Drift
    # -------------------------------------------------

    drift = np.linspace(
        0,
        np.random.uniform(-0.02, 0.02),
        N_SAMPLES
    )

    trace = trace + drift

    # -------------------------------------------------
    # Variable length
    # -------------------------------------------------

    random_cut = np.random.randint(
        0,
        10
    )

    trace = trace[:N_SAMPLES - random_cut]

    # -------------------------------------------------
    # Sparse NaNs
    # -------------------------------------------------

    n_nans = np.random.randint(0, 3)

    if n_nans > 0:

        nan_idx = np.random.choice(
            len(trace),
            size=n_nans,
            replace=False
        )

        trace[nan_idx] = np.nan

    return trace


# ---------------- DATAFRAME --------------------------


def create_test_dataframe():

    rows = []

    for participant in range(N_PARTICIPANTS):

        participant_name = f"P{participant+1}"

        for condition in CONDITIONS:

            for trial in range(N_TRIALS):

                trace = generate_pupil_trace(
                    condition
                )

                row = [
                    participant_name,
                    condition
                ]

                row.extend(trace)

                rows.append(row)

    max_len = max(len(r) for r in rows)

    for row in rows:

        missing = max_len - len(row)

        row.extend([np.nan] * missing)

    columns = [
        "Participant",
        "Condition"
    ]

    for i in range(max_len - 2):

        columns.append(f"T{i}")

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    return df


# ---------------- FULL PIPELINE TEST -----------------


def test_full_pipeline_trial_mode():

    df = create_test_dataframe()

    results = run_decompose(

        df,

        participant_col=0,

        condition_col=1,

        pupil_start_col=2,

        mode="trial",

        preprocess=True,

        preprocess_method="gaussian",

        sigma=1,

        sample_interval_ms=SAMPLE_INTERVAL_MS,

        baseline_window=(0, 40),

        stim_onset_idx=STIM_ONSET_IDX,

        dilation_window_ms=250,

        onset_method="wilcoxon",

        plot=False,

        save_csv=True,

        csv_filename="test_trial_output.csv"
    )

    # -------------------------------------------------
    # Core outputs
    # -------------------------------------------------

    assert results is not None

    assert "features" in results

    assert isinstance(
        results["features"],
        pd.DataFrame
    )

    # -------------------------------------------------
    # CSV output
    # -------------------------------------------------

    assert os.path.exists(
        "test_trial_output.csv"
    )

    # -------------------------------------------------
    # Feature columns
    # -------------------------------------------------

    expected_columns = [

        "Constriction_Onset_ms",

        "Pre_PLR_Dilation",

        "Constriction_Amplitude",

        "Constriction_Latency_ms",

        "Constriction_Rate",

        "Dilation_Rate"
    ]

    for col in expected_columns:

        assert col in results["features"].columns

    # -------------------------------------------------
    # Some successful trials
    # -------------------------------------------------

    success_n = len(
        results["success_trials"]
    )

    assert success_n > 0


# -------- PARTICIPANT-WISE PIPELINE TEST ------------


def test_full_pipeline_participant_mode():

    df = create_test_dataframe()

    results = run_decompose(

        df,

        participant_col=0,

        condition_col=1,

        pupil_start_col=2,

        mode="participant",

        preprocess=True,

        preprocess_method="gaussian",

        sigma=1,

        sample_interval_ms=SAMPLE_INTERVAL_MS,

        baseline_window=(0, 40),

        stim_onset_idx=STIM_ONSET_IDX,

        dilation_window_ms=250,

        onset_method="slope",

        plot=False,

        save_csv=True,

        csv_filename="test_participant_output.csv"
    )

    assert results is not None

    assert "features" in results

    assert os.path.exists(
        "test_participant_output.csv"
    )



# ---------------- CLEANUP TEST -----------------------


def test_cleanup_generated_files():

    files = [
        "test_trial_output.csv",
        "test_participant_output.csv"
    ]

    for file in files:

        if os.path.exists(file):

            os.remove(file)

    for file in files:

        assert not os.path.exists(file)
