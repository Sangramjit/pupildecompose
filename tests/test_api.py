import numpy as np
import pandas as pd

from pupildecompose import run_decompose

from pupildecompose.dataframe import (
    prepare_dataframe
)

from pupildecompose.preprocess import (
    preprocess_pupil
)

from pupildecompose.features import (
    decompose_pupil
)


# ---------------- SYNTHETIC DATA ---------------------


def create_test_dataframe():

    np.random.seed(1)

    rows = []

    for p in range(3):

        for c in ["A", "B"]:

            for t in range(5):

                trace = (

                    np.ones(200)

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

    df = pd.DataFrame(

        rows,
        columns=columns
    )

    return df


# ---------------- DATAFRAME TEST ---------------------


def test_prepare_dataframe():

    df = create_test_dataframe()

    prepared = prepare_dataframe(

        df,

        participant_col=0,

        condition_col=1,

        pupil_start_col=2,

        mode="trial"
    )

    assert "traces" in prepared

    assert prepared["traces"].shape[0] == 30



# ---------------- PREPROCESS TEST --------------------


def test_preprocess():

    df = create_test_dataframe()

    prepared = prepare_dataframe(df)

    processed = preprocess_pupil(

        prepared,

        preprocess=True,

        sigma=1
    )

    assert "traces" in processed



# ---------------- FEATURE TEST -----------------------


def test_feature_extraction():

    df = create_test_dataframe()

    prepared = prepare_dataframe(df)

    processed = preprocess_pupil(

        prepared,

        preprocess=True
    )

    results = decompose_pupil(

        processed,

        sample_interval_ms=10,

        baseline_window=(0, 20),

        stim_onset_idx=50,

        onset_method="wilcoxon"
    )

    assert "features" in results

    assert isinstance(
        results["features"],
        pd.DataFrame
    )