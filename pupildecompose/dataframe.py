import numpy as np
import pandas as pd

from tqdm import tqdm



# ---------------- VALIDATION -------------------------


def _validate_dataframe(
    df,
    participant_col,
    condition_col,
    pupil_start_col,
    mode
):

    """
    Validate dataframe and user inputs.
    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "Input must be pandas DataFrame."
        )

    if mode not in ["trial", "participant"]:

        raise ValueError(
            "mode must be 'trial' or 'participant'."
        )

    if participant_col >= df.shape[1]:

        raise ValueError(
            "participant_col exceeds dataframe size."
        )

    if condition_col is not None:

        if condition_col >= df.shape[1]:

            raise ValueError(
                "condition_col exceeds dataframe size."
            )

    if pupil_start_col >= df.shape[1]:

        raise ValueError(
            "pupil_start_col exceeds dataframe size."
        )



# ---------------- SUMMARY PRINT ----------------------


def _print_summary(
    df,
    participant_col,
    condition_name,
    mode,
    condition_col
):

    """
    Print dataset summary.
    """

    participants = df.iloc[:, participant_col].unique()

    conditions = df[condition_name].unique()

    print("\n========================================")
    print("DATA SUMMARY")
    print("========================================\n")

    print(
        f"Number of participants : "
        f"{len(participants)}"
    )

    print(
        f"Conditions             : "
        f"{list(conditions)}"
    )

    print(
        f"Analysis mode          : "
        f"{mode}"
    )

    print(
        "\nTrials per "
        "participant/condition:\n"
    )

    if condition_col is not None:

        counts = (

            df.groupby(

                [
                    df.columns[participant_col],
                    condition_name
                ]
            )
            .size()
        )

    else:

        counts = (

            df.groupby(
                df.columns[participant_col]
            )
            .size()
        )

    print(counts)



# ---------------- PUPIL EXTRACTION -------------------


def _extract_pupil_matrix(
    df,
    pupil_start_col
):

    """
    Extract pupil matrix safely.
    """

    pupil_df = df.iloc[:, pupil_start_col:]

    pupil_df = pupil_df.apply(

        pd.to_numeric,
        errors="coerce"
    )

    return pupil_df



# ---------------- NAN QUALITY CHECK ------------------


def _check_nan_quality(
    pupil_df,
    nan_threshold=0.5
):

    """
    Check trials with excessive NaNs.
    """

    bad_nan_trials = []

    print("\nChecking NaN quality...\n")

    for i in tqdm(

        range(len(pupil_df)),
        desc="NaN inspection"
    ):

        row = pupil_df.iloc[i].values

        nan_ratio = np.mean(
            np.isnan(row)
        )

        if nan_ratio > nan_threshold:

            print(

                f"Warning: Trial {i} "
                f"contains "
                f"{nan_ratio*100:.2f}% NaNs. "
                f"Feature extraction may fail."
            )

            bad_nan_trials.append(i)

    return bad_nan_trials



# ---------------- TRACE PADDING ----------------------


def _pad_traces(
    pupil_df
):

    """
    Pad unequal traces using NaN.
    """

    trace_lengths = []

    print("\nCalculating trace lengths...\n")

    for i in tqdm(

        range(len(pupil_df)),
        desc="Trace length"
    ):

        row = pupil_df.iloc[i].values

        valid_length = np.sum(

            ~np.isnan(row)
        )

        trace_lengths.append(
            valid_length
        )

    max_length = np.max(
        trace_lengths
    )

    padded_traces = []

    print("\nPadding traces...\n")

    for i in tqdm(

        range(len(pupil_df)),
        desc="NaN padding"
    ):

        row = (

            pupil_df.iloc[i]
            .values
            .astype(float)
        )

        valid_values = row[
            ~np.isnan(row)
        ]

        padded = np.pad(

            valid_values,

            (
                0,
                max_length
                - len(valid_values)
            ),

            constant_values=np.nan
        )

        padded_traces.append(
            padded
        )

    padded_traces = np.vstack(
        padded_traces
    )

    return (
        padded_traces,
        max_length
    )



# -------- PARTICIPANT-WISE AVERAGING -----------------


def _average_participant_traces(
    df,
    padded_traces,
    participant_col,
    condition_name
):

    """
    Participant-wise averaging.
    """

    participants = (
        df.iloc[:, participant_col]
        .unique()
    )

    conditions = (
        df[condition_name]
        .unique()
    )

    averaged_traces = []

    averaged_participants = []

    averaged_conditions = []

    total_steps = (

        len(participants)
        * len(conditions)
    )

    print(
        "\nPreparing participant "
        "averages...\n"
    )

    with tqdm(

        total=total_steps,
        desc="Participant averaging"

    ) as pbar:

        for participant in participants:

            for condition in conditions:

                mask = (

                    (
                        df.iloc[
                            :,
                            participant_col
                        ]
                        == participant
                    )

                    &

                    (
                        df[condition_name]
                        == condition
                    )
                )

                traces = padded_traces[
                    mask
                ]

                if len(traces) == 0:

                    pbar.update(1)

                    continue

                mean_trace = np.nanmean(

                    traces,
                    axis=0
                )

                averaged_traces.append(
                    mean_trace
                )

                averaged_participants.append(
                    participant
                )

                averaged_conditions.append(
                    condition
                )

                pbar.update(1)

    averaged_traces = np.vstack(
        averaged_traces
    )

    return (

        averaged_traces,

        np.array(
            averaged_participants
        ),

        np.array(
            averaged_conditions
        )
    )



# ---------------- MAIN USER FUNCTION -----------------


def prepare_dataframe(

    df,

    participant_col=0,

    condition_col=1,

    pupil_start_col=2,

    mode="trial",

    nan_threshold=0.5
):

    """
    Prepare dataframe for decomposition.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    participant_col : int
        Participant column index.

    condition_col : int or None
        Condition column index.

    pupil_start_col : int
        First pupil-data column.

    mode : str
        "trial" or "participant"

    nan_threshold : float
        Threshold for excessive NaNs.
    """


    # ---------------- VALIDATION -------------------------


    _validate_dataframe(

        df,
        participant_col,
        condition_col,
        pupil_start_col,
        mode
    )

   
    # ---------------- COPY -------------------------------


    df = df.copy()


    # ---------------- CONDITION HANDLING -----------------


    if condition_col is None:

        df["Condition"] = (
            "Condition_1"
        )

        condition_name = (
            "Condition"
        )

    else:

        condition_name = (
            df.columns[
                condition_col
            ]
        )


    # ---------------- SUMMARY ----------------------------


    _print_summary(

        df,

        participant_col,

        condition_name,

        mode,

        condition_col
    )

    # ---------------- EXTRACT MATRIX ---------------------


    pupil_df = _extract_pupil_matrix(

        df,
        pupil_start_col
    )


    # ---------------- NAN CHECK --------------------------


    bad_nan_trials = _check_nan_quality(

        pupil_df,
        nan_threshold=nan_threshold
    )


    # ---------------- TRACE PADDING ----------------------


    (
        padded_traces,
        max_length

    ) = _pad_traces(
        pupil_df
    )

    print(
        f"\nMaximum trace length : "
        f"{max_length}"
    )

    print(
        "\nNaN padding completed."
    )


    # ---------------- OUTPUT -----------------------------


    prepared_data = {

        "participants":

            df.iloc[
                :,
                participant_col
            ].values,

        "conditions":

            df[
                condition_name
            ].values,

        "traces":

            padded_traces,

        "mode":

            mode,

        "max_length":

            max_length,

        "bad_nan_trials":

            bad_nan_trials
    }


    # ------------ PARTICIPANT MODE ----------------------
   

    if mode == "participant":

        (
            averaged_traces,

            averaged_participants,

            averaged_conditions

        ) = _average_participant_traces(

            df,

            padded_traces,

            participant_col,

            condition_name
        )

        prepared_data["traces"] = (

            averaged_traces
        )

        prepared_data["participants"] = (

            averaged_participants
        )

        prepared_data["conditions"] = (

            averaged_conditions
        )

        print(
            "\nParticipant-wise "
            "averaging completed."
        )


    # ---------------- FINISHED ---------------------------
 

    print(
        "\nData preparation complete.\n"
    )

    return prepared_data