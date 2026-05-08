import numpy as np
import pandas as pd

from pupildecompose import run_decompose

###########################################
# This example is with synthetic data #####
############################################


np.random.seed(42)

N_PARTICIPANTS = 10

N_TRIALS = 20

CONDITIONS = ["Condition_A", "Condition_B"]

SAMPLING_RATE = 100  # Hz

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



# ---------------- SYNTHETIC PUPIL --------------------


def generate_pupil_trace(
    condition
):

    """
    Generate strong synthetic PLR.

    Physiological sequence:
    baseline
    -> pre-PLR dilation
    -> sharp constriction
    -> slow recovery
    """

    time = np.arange(N_SAMPLES)

    # ---------------- BASELINE ---------------------------

    baseline = 1.0

    trace = np.ones(
        N_SAMPLES
    ) * baseline

    # ---------------- PRE-PLR DILATION -------------------

    if condition == "Condition_A":

        pre_amp = 0.05

    else:

        pre_amp = 0.08

    pre_center = (
        STIM_ONSET_IDX + 15
    )

    pre_width = 5

    pre_dilation = (

        pre_amp

        * np.exp(

            -(
                (
                    time
                    - pre_center
                ) ** 2
            )

            / (2 * pre_width**2)
        )
    )

    # ---------------- CONSTRICTION ----------------

    if condition == "Condition_A":

        constriction_amp = -0.45

    else:

        constriction_amp = -0.60

    constriction_center = (
        STIM_ONSET_IDX + 35
    )

    constriction_width = 8

    constriction = (

        constriction_amp

        * np.exp(

            -(
                (
                    time
                    - constriction_center
                ) ** 2
            )

            / (2 * constriction_width**2)
        )
    )

    # ---------------- RECOVERY DILATION -----------------

    if condition == "Condition_A":

        recovery_amp = 0.18

    else:

        recovery_amp = 0.25

    recovery_center = (
        STIM_ONSET_IDX + 90
    )

    recovery_width = 25

    recovery = (

        recovery_amp

        * np.exp(

            -(
                (
                    time
                    - recovery_center
                ) ** 2
            )

            / (2 * recovery_width**2)
        )
    )

    # ---------------- COMBINE ----------------------------

    trace = (

        trace

        + pre_dilation

        + constriction

        + recovery
    )


    noise = np.random.normal(

        0,
        0.008,
        N_SAMPLES
    )

    trace = trace + noise

    # ---------------- DRIFT ------------------------------

    drift = np.linspace(

        0,

        np.random.uniform(
            -0.02,
            0.02
        ),

        N_SAMPLES
    )

    trace = trace + drift

    # ---------------- VARIABLE LENGTH --------------------

    # keep enough recovery

    random_cut = np.random.randint(

        0,
        15
    )

    trace = trace[
        :N_SAMPLES - random_cut
    ]

    # ---------------- FEWER NaNs -------------------------

    n_nans = np.random.randint(
        0,
        5
    )

    if n_nans > 0:

        nan_indices = np.random.choice(

            len(trace),

            size=n_nans,

            replace=False
        )

        trace[nan_indices] = np.nan

    return trace


# ---------------- GENERATE DATAFRAME -----------------

rows = []

for participant in range(N_PARTICIPANTS):

    participant_name = (
        f"P{participant+1}"
    )

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


# ---------------- FIND MAX LENGTH --------------------

max_len = max(
    len(r)
    for r in rows
)

# pad rows equally
for row in rows:

    missing = max_len - len(row)

    row.extend(
        [np.nan] * missing
    )


# ---------------- COLUMN NAMES -----------------------

columns = [

    "Participant",
    "Condition"
]

for i in range(max_len - 2):

    columns.append(
        f"T{i+1}"
    )


# ---------------- DATAFRAME --------------------------

df = pd.DataFrame(

    rows,
    columns=columns
)

print("\nSynthetic dataframe created.\n")

print(df.head())

# ---------------- RUN PIPELINE -----------------------
results = run_decompose(

    df,

    # dataframe
    participant_col=0,
    condition_col=1,
    pupil_start_col=2,

    mode="participant",

    # preprocess
    preprocess=False,
    preprocess_method="gaussian",
    sigma=1,

    # features
    sample_interval_ms=SAMPLE_INTERVAL_MS,

    baseline_window=(0, 40),

    stim_onset_idx=STIM_ONSET_IDX,

    dilation_window_ms=250,

    # plotting
    plot=True
)


# ---------------- PRINT FEATURES ---------------------

print(
    results["features"].head()
)