
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------- GAUSSIAN SMOOTH --------------------

def _gaussian_smooth_trace(
    trace,
    sigma=2
):

    """
    Smooth a single pupil trace using
    Gaussian filtering.

    NaN values are preserved.
    """

    trace = np.asarray(trace, dtype=float)

    # Store NaN locations
   
    nan_mask = np.isnan(trace)

   # Temporary interpolation for smoothing
   
    valid_idx = np.where(~nan_mask)[0]

    if len(valid_idx) < 3:
        return trace

    interpolated = np.interp(
        np.arange(len(trace)),
        valid_idx,
        trace[valid_idx]
    )


    # Gaussian smoothing
   

    smoothed = gaussian_filter1d(
        interpolated,
        sigma=sigma
    )


    # Restore NaNs

    smoothed[nan_mask] = np.nan

    return smoothed



# ---------------- SMOOTH ALL TRACES ------------------

def _smooth_all_traces(
    traces,
    sigma=2
):

    """
    Smooth all traces individually.
    """

    smoothed_traces = []

    for trace in traces:

        smoothed = _gaussian_smooth_trace(
            trace,
            sigma=sigma
        )

        smoothed_traces.append(smoothed)

    smoothed_traces = np.vstack(
        smoothed_traces
    )

    return smoothed_traces



# ---------------- MAIN USER FUNCTION -----------------

def preprocess_pupil(
    prepared_data,
    preprocess=None,
    method="gaussian",
    sigma=2
):

    """
    Preprocess pupil traces.

    Parameters
    ----------
    prepared_data : dict
        Output dictionary from prepare_dataframe().

    preprocess : bool or None
        If None or False, no preprocessing performed.

    method : str
        Preprocessing method.

        Current options:
        - "gaussian"

    sigma : float
        Gaussian smoothing sigma.

    Returns
    -------
    prepared_data : dict
        Updated dictionary with preprocessed traces.
    """
    # ---------------- NO PREPROCESSING -------------------
   

    if preprocess is None or preprocess is False:

        print("\nNo preprocessing selected.\n")

        return prepared_data


    # ---------------- INPUT VALIDATION -------------------

    if not isinstance(prepared_data, dict):

        raise TypeError(
            "prepared_data must be dictionary output from prepare_dataframe()."
        )

    if "traces" not in prepared_data:

        raise ValueError(
            "prepared_data does not contain traces."
        )

    if method not in ["gaussian"]:

        raise ValueError(
            "Currently only 'gaussian' method is supported."
        )

    
    # ---------------- EXTRACT TRACES ---------------------


    traces = prepared_data["traces"]

    mode = prepared_data["mode"]

    print("\n========================================")
    print("PREPROCESSING")
    print("========================================\n")

    print(f"Mode                 : {mode}")
    print(f"Method               : {method}")
    print(f"Gaussian sigma       : {sigma}")

  
    # ---------------- TRIAL-WISE MODE -------------------


    if mode == "trial":

        print(
            "\nApplying smoothing trial-wise..."
        )

        smoothed_traces = _smooth_all_traces(
            traces,
            sigma=sigma
        )

   
    # ------------ PARTICIPANT-WISE MODE -----------------
  

    elif mode == "participant":

        print(
            "\nApplying smoothing to participant averages..."
        )

        smoothed_traces = _smooth_all_traces(
            traces,
            sigma=sigma
        )


    # ---------------- SAVE OUTPUT ------------------------


    prepared_data["traces"] = smoothed_traces

    prepared_data["preprocessing"] = {

        "performed": True,
        "method": method,
        "sigma": sigma
    }


    # ---------------- FINISHED ---------------------------
  

    print("\nPreprocessing completed.\n")

    return prepared_data



