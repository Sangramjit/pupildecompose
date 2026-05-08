import numpy as np
import pandas as pd

from pupildecompose import run_decompose



# ---------------- SYNTHETIC DATA ---------------------


def create_smoke_df():

    np.random.seed(42)

    rows = []

    for p in range(2):

        for c in ["A", "B"]:

            for t in range(3):

                time = np.arange(200)

                trace = (

                    1

                    - 0.3 * np.exp(
                        -((time - 80)**2)
                        / (2 * 10**2)
                    )

                    + np.random.normal(
                        0,
                        0.01,
                        200
                    )
                )

                row = [

                    f"P{p}",
                    c
                ]

                row.extend(trace)

                rows.append(row)

    columns = [

        "Participant",
        "Condition"
    ]

    columns.extend(

        [f"T{i}" for i in range(200)]
    )

    return pd.DataFrame(

        rows,
        columns=columns
    )



# ---------------- SMOKE TEST -------------------------


def test_pipeline_runs():

    df = create_smoke_df()

    results = run_decompose(

        df,

        participant_col=0,

        condition_col=1,

        pupil_start_col=2,

        preprocess=True,

        sigma=1,

        sample_interval_ms=10,

        baseline_window=(0, 20),

        stim_onset_idx=50,

        onset_method="slope",

        plot=False,

        save_csv=False
    )

    assert results is not None

    assert "features" in results