from .dataframe import prepare_dataframe

from .preprocess import preprocess_pupil

from .features import decompose_pupil

from .plotting import plot_pupil_features

from .utils import (
    print_summary,
    output_csv
)


# =====================================================
# ---------------- MAIN PIPELINE ----------------------
# =====================================================

def run_decompose(

    df,

    # ---------------------------------------------
    # dataframe.py
    # ---------------------------------------------

    participant_col=0,

    condition_col=1,

    pupil_start_col=2,

    mode="trial",

    # ---------------------------------------------
    # preprocess.py
    # ---------------------------------------------

    preprocess=None,

    preprocess_method="gaussian",

    sigma=2,

    # ---------------------------------------------
    # features.py
    # ---------------------------------------------

    sample_interval_ms=8,

    baseline_window=(0, 20),

    stim_onset_idx=90,

    dilation_window_ms=300,

    onset_method="wilcoxon",

    # ---------------------------------------------
    # plotting.py
    # ---------------------------------------------

    plot=True,

    # ---------------------------------------------
    # output
    # ---------------------------------------------

    return_results=True,

    save_csv=True,

    csv_filename="pupil_features.csv"
):

    """
    Complete pupil decomposition pipeline.

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
        Analysis mode.

        Options:
        - "trial"
        - "participant"

    preprocess : bool or None
        Whether preprocessing should be performed.

    preprocess_method : str
        Preprocessing method.

    sigma : float
        Gaussian smoothing sigma.

    sample_interval_ms : float
        Sampling interval in milliseconds.

    baseline_window : tuple
        Baseline window indices.

    stim_onset_idx : int
        Stimulus onset index.

    dilation_window_ms : float
        Dilation-rate fitting window.

    onset_method : str
        Constriction onset detection method.

        Options:
        - "wilcoxon"
        - "slope"

    plot : bool
        Whether to generate plots.

    return_results : bool
        Whether to return results dictionary.

    Returns
    -------
    prepared_data : dict
        Full decomposition output.
    """


    # ---------------- INPUT VALIDATION -------------------


    if onset_method not in [

        "wilcoxon",
        "slope"
    ]:

        raise ValueError(

            "onset_method must be "
            "'wilcoxon' or 'slope'"
        )

    
    # ---------------- START ------------------------------
   

    print("\n========================================")
    print("RUNNING PUPILDECOMPOSE")
    print("========================================\n")

    print(f"Mode                 : {mode}")

    print(f"Preprocessing        : {preprocess}")

    print(f"Onset method         : {onset_method}")


    # -------- DATAFRAME PREPARATION ----------------------
  

    prepared_data = prepare_dataframe(

        df=df,

        participant_col=participant_col,

        condition_col=condition_col,

        pupil_start_col=pupil_start_col,

        mode=mode
    )


    # ------------ PREPROCESSING --------------------------

    prepared_data = preprocess_pupil(

        prepared_data,

        preprocess=preprocess,

        method=preprocess_method,

        sigma=sigma
    )

   
    # ---------- FEATURE EXTRACTION -----------------------

    prepared_data = decompose_pupil(

        prepared_data,

        sample_interval_ms=sample_interval_ms,

        baseline_window=baseline_window,

        stim_onset_idx=stim_onset_idx,

        dilation_window_ms=dilation_window_ms,

        onset_method=onset_method
    )

  

    if plot:

        plot_pupil_features(

            prepared_data,

            sample_interval_ms=sample_interval_ms
        )

   

    print_summary(prepared_data)

    if save_csv:

        output_csv(

            prepared_data,

            filename=csv_filename
        )


    print("\n========================================")
    print("PIPELINE COMPLETED")
    print("========================================\n")

 

    if return_results:

        return prepared_data