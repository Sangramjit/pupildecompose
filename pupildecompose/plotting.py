import numpy as np
import matplotlib.pyplot as plt



# ---------------- SEM FUNCTION -----------------------


def _nansem(data, axis=0):

    valid_n = np.sum(

        ~np.isnan(data),
        axis=axis
    )

    sem = (

        np.nanstd(data, axis=axis)

        / np.sqrt(valid_n)
    )

    return sem



# ---------------- TIME VECTOR ------------------------


def _create_time_vector(
    n_samples,
    sample_interval_ms
):

    return (

        np.arange(n_samples)

        * sample_interval_ms
    )



# ---------------- STYLE ------------------------------


def _apply_style():

    plt.style.use("default")

    plt.rcParams.update({

        "font.size": 11,

        "axes.labelsize": 13,

        "axes.titlesize": 15,

        "xtick.labelsize": 10,

        "ytick.labelsize": 10,

        "legend.fontsize": 11,

        "axes.spines.top": False,

        "axes.spines.right": False,

        "lines.linewidth": 2.5
    })



# ---------------- FEATURE BARPLOT --------------------


def _plot_feature_barplot(

    ax,

    features_df,

    feature_name,

    condition_present
):

    if not condition_present:

        values = (

            features_df[
                feature_name
            ]
            .values
        )

        mean_val = np.nanmean(values)

        sem_val = _nansem(values)

        ax.bar(

            [0],

            [mean_val],

            yerr=[sem_val],

            capsize=5,

            alpha=0.85,

            width=0.6
        )

        ax.set_xticks([0])

        ax.set_xticklabels(["All"])

    else:

        conditions = (

            features_df[
                "Condition"
            ]
            .unique()
        )

        means = []

        sems = []

        for cond in conditions:

            vals = (

                features_df.loc[

                    features_df[
                        "Condition"
                    ] == cond,

                    feature_name
                ]
                .values
            )

            means.append(
                np.nanmean(vals)
            )

            sems.append(
                _nansem(vals)
            )

        ax.bar(

            np.arange(len(conditions)),

            means,

            yerr=sems,

            capsize=5,

            alpha=0.85,

            width=0.6
        )

        ax.set_xticks(
            np.arange(len(conditions))
        )

        ax.set_xticklabels(
            conditions
        )

    ax.set_title(
        feature_name.replace(
            "_",
            " "
        ),
        pad=10
    )

    ax.set_ylabel("Value")



# ---------------- MAIN FUNCTION ----------------------


def plot_pupil_features(

    prepared_data,

    sample_interval_ms
):

    """
    Plot pupil traces and features.
    """

    _apply_style()

    traces = prepared_data["traces"]

    features_df = prepared_data["features"]

    conditions = (

        features_df[
            "Condition"
        ]
        .unique()
    )

    condition_present = (
        len(conditions) > 1
    )

    n_samples = traces.shape[1]

    time_vector = _create_time_vector(

        n_samples,

        sample_interval_ms
    )


    # ---------------- TIMESCOURSE ------------------------


    fig1, ax = plt.subplots(

        figsize=(12, 6)
    )

    colors = [

        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red"
    ]

    # -----------------------------------------------------
    # SINGLE CONDITION
    # -----------------------------------------------------

    if not condition_present:

        mean_trace = np.nanmean(

            traces,
            axis=0
        )

        sem_trace = _nansem(

            traces,
            axis=0
        )

        ax.plot(

            time_vector,

            mean_trace,

            color=colors[0],

            label="Pupil"
        )

        ax.fill_between(

            time_vector,

            mean_trace - sem_trace,

            mean_trace + sem_trace,

            color=colors[0],

            alpha=0.25
        )

    # -----------------------------------------------------
    # MULTIPLE CONDITIONS
    # -----------------------------------------------------

    else:

        for idx, cond in enumerate(conditions):

            mask = (

                features_df[
                    "Condition"
                ] == cond
            )

            cond_traces = traces[mask]

            mean_trace = np.nanmean(

                cond_traces,
                axis=0
            )

            sem_trace = _nansem(

                cond_traces,
                axis=0
            )

            ax.plot(

                time_vector,

                mean_trace,

                label=cond,

                color=colors[idx]
            )

            ax.fill_between(

                time_vector,

                mean_trace - sem_trace,

                mean_trace + sem_trace,

                color=colors[idx],

                alpha=0.20
            )

        ax.legend(
            frameon=False
        )

    ax.set_title(
        "Pupil Timecourse"
    )

    ax.set_xlabel(
        "Time (ms)"
    )

    ax.set_ylabel(
        "Pupil Size"
    )

    plt.tight_layout()


   
    # ---------------- FEATURES ---------------------------


    feature_list = [

        "Constriction_Onset_ms",

        "Pre_PLR_Dilation",

        "Constriction_Amplitude",

        "Constriction_Latency_ms",

        "Constriction_Rate",

        "Dilation_Rate"
    ]

    fig2, axes = plt.subplots(

        2,
        3,

        figsize=(14, 8)
    )

    axes = axes.flatten()

    for ax, feature_name in zip(

        axes,
        feature_list
    ):

        _plot_feature_barplot(

            ax,

            features_df,

            feature_name,

            condition_present
        )

    plt.tight_layout()

    plt.show()