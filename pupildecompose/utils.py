import numpy as np
import pandas as pd


# =====================================================
# ---------------- SAVE FEATURES CSV ------------------
# =====================================================

def output_csv(
    prepared_data,
    filename="pupil_features_output.csv"
):

    """
    Save extracted features to CSV.

    Parameters
    ----------
    prepared_data : dict
        Output from decompose_pupil()

    filename : str
        Output CSV filename.
    """


    # ---------------- INPUT VALIDATION -------------------


    if not isinstance(prepared_data, dict):

        raise TypeError(
            "prepared_data must be dictionary "
            "output from decompose_pupil()."
        )

    if "features" not in prepared_data:

        raise ValueError(
            "No features found in prepared_data."
        )

    features_df = prepared_data["features"]


    # ---------------- SAVE CSV ---------------------------
    

    features_df.to_csv(

        filename,
        index=False
    )

    print("\n========================================")
    print("CSV EXPORT")
    print("========================================\n")

    print(f"Output file saved:\n{filename}")

    print("\nCSV export completed.\n")



# ---------------- SUMMARY FUNCTION -------------------


def print_summary(
    prepared_data
):

    """
    Print full analysis summary.
    """


    # ---------------- INPUT VALIDATION -------------------


    if not isinstance(prepared_data, dict):

        raise TypeError(
            "prepared_data must be dictionary."
        )

    required_keys = [

        "participants",
        "conditions",
        "traces"
    ]

    for key in required_keys:

        if key not in prepared_data:

            raise ValueError(
                f"Missing key: {key}"
            )


    # ---------------- BASIC INFO -------------------------


    participants = prepared_data["participants"]

    conditions = prepared_data["conditions"]

    traces = prepared_data["traces"]

    unique_participants = np.unique(
        participants
    )

    unique_conditions = np.unique(
        conditions
    )

    total_trials = len(traces)

    # ---------------- FEATURE INFO -----------------------
    

    if "success_trials" in prepared_data:

        extracted_trials = len(

            prepared_data[
                "success_trials"
            ]
        )

    else:

        extracted_trials = "Not computed"

   
    # ---------------- PRINT SUMMARY ----------------------
   

    print("\n========================================")
    print("PUPILDECOMPOSE SUMMARY")
    print("========================================\n")

    print(
        f"Total participants : "
        f"{len(unique_participants)}"
    )

    print(
        f"Conditions          : "
        f"{list(unique_conditions)}"
    )

    print(
        f"Total trials        : "
        f"{total_trials}"
    )

    print(
        f"Extracted trials    : "
        f"{extracted_trials}"
    )

    # ---------------- PARTICIPANT INFO -------------------
   

    print("\nTrials per participant:\n")

    participant_counts = pd.Series(
        participants
    ).value_counts()

    print(participant_counts)

  
    # ---------------- CONDITION INFO ---------------------
  

    print("\nTrials per condition:\n")

    condition_counts = pd.Series(
        conditions
    ).value_counts()

    print(condition_counts)


    # ---------------- FAILED TRIALS ----------------------


    if "failed_trials" in prepared_data:

        print("\nFailed row indices:\n")

        print(
            prepared_data[
                "failed_trials"
            ]
        )

    print("\nSummary completed.\n")



# ---------------- FEATURE SUCCESS --------------------


def feature_success_rate(
    prepared_data
):

    """
    Print feature extraction success rate.
    """

    if "success_trials" not in prepared_data:

        raise ValueError(
            "No success_trials found."
        )

    total_trials = len(
        prepared_data["traces"]
    )

    successful = len(
        prepared_data["success_trials"]
    )

    failed = len(
        prepared_data["failed_trials"]
    )

    success_rate = (
        successful / total_trials
    ) * 100

    print("\n========================================")
    print("FEATURE SUCCESS RATE")
    print("========================================\n")

    print(f"Total trials      : {total_trials}")

    print(f"Successful trials : {successful}")

    print(f"Failed trials     : {failed}")

    print(
        f"Success rate      : "
        f"{success_rate:.2f}%"
    )

    print()



# ---------------- GET VALID TRIALS -------------------


def get_valid_trials(
    prepared_data
):

    """
    Return only valid feature rows.
    """

    if "features" not in prepared_data:

        raise ValueError(
            "No features found."
        )

    features_df = prepared_data["features"]

    valid_df = features_df.loc[
        features_df["Valid_Trial"] == True
    ]

    return valid_df



# ---------------- GET FAILED TRIALS ------------------


def get_failed_trials(
    prepared_data
):

    """
    Return failed feature rows.
    """

    if "features" not in prepared_data:

        raise ValueError(
            "No features found."
        )

    features_df = prepared_data["features"]

    failed_df = features_df.loc[
        features_df["Valid_Trial"] == False
    ]

    return failed_df