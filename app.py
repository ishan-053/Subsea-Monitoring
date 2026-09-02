import io
import zipfile
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from backend import PipelineBackend


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Subsea Pipeline Digital Twin",
    page_icon="◉",
    layout="wide"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #080E14;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.card {
    background-color: #101820;
    border: 1px solid #263743;
    border-radius: 10px;
    padding: 16px;
    min-height: 95px;
}

.label {
    color: #8D9CA7;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.value {
    color: #E7EDF1;
    font-size: 25px;
    font-weight: 600;
    margin-top: 8px;
}

.small {
    color: #8D9CA7;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "replay_index" not in st.session_state:
    st.session_state.replay_index = 0

if "replay_running" not in st.session_state:
    st.session_state.replay_running = False

if "backend" not in st.session_state:
    st.session_state.backend = PipelineBackend()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


# =========================================================
# HEADER
# =========================================================

left, right = st.columns([7, 3])

with left:

    st.title("SUBSEA PIPELINE DIGITAL TWIN")

    st.markdown(
        '<span class="small">'
        'Real-time leak detection · localization · response'
        '</span>',
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        """
        <div style="text-align:right; padding-top:15px;">
        <span style="color:#65B891;">●</span>
        SIMULATED REAL-TIME
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# FEATURE 1 — TELEMETRY DATASET
# =========================================================

st.subheader("Telemetry Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or XLSX telemetry",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    if (
        st.session_state.uploaded_name
        != uploaded_file.name
    ):

        # -------------------------------------------------
        # READ FILE
        # -------------------------------------------------

        try:

            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_excel(
                    uploaded_file
                )

        except Exception as e:

            st.error(
                f"Could not read dataset: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # CLEAN COLUMN NAMES
        # -------------------------------------------------

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )


        # -------------------------------------------------
        # TIME NORMALIZATION
        # -------------------------------------------------

        try:

            # Relative milliseconds
            if "Relative Time (ms)" in df.columns:

                df["Time (s)"] = (
                    pd.to_numeric(
                        df["Relative Time (ms)"],
                        errors="coerce"
                    )
                    / 1000.0
                )


            # Already seconds
            elif "Time (s)" in df.columns:

                df["Time (s)"] = pd.to_numeric(
                    df["Time (s)"],
                    errors="coerce"
                )


            # Timestamp in seconds
            elif "Timestamp (s)" in df.columns:

                df["Time (s)"] = pd.to_numeric(
                    df["Timestamp (s)"],
                    errors="coerce"
                )


            # Clock timestamp
            elif "Timestamp" in df.columns:

                timestamps = pd.to_datetime(
                    df["Timestamp"],
                    format="%H:%M:%S.%f",
                    errors="coerce"
                )

                df["Time (s)"] = (
                    timestamps
                    - timestamps.iloc[0]
                ).dt.total_seconds()


            else:

                st.error(
                    "No usable timestamp column found."
                )

                st.write(
                    "Columns detected:"
                )

                st.write(
                    df.columns.tolist()
                )

                st.stop()


        except Exception as e:

            st.error(
                f"Timestamp conversion failed: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # RESET APPLICATION
        # -------------------------------------------------

        st.session_state.df = df

        st.session_state.uploaded_name = (
            uploaded_file.name
        )

        st.session_state.replay_index = 0

        st.session_state.replay_running = False

        st.session_state.backend = (
            PipelineBackend()
        )

        st.session_state.history = []

        st.session_state.last_result = None


df = st.session_state.df


# =========================================================
# DATASET VALIDATION
# =========================================================

dataset_valid = False


if df is not None:

    required_columns = [
        "Time (s)",
        "Inlet Pressure (Bar)",
        "Outlet Pressure (Bar)"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        st.error(
            "Dataset validation failed."
        )

        for column in missing_columns:

            st.write(
                f"• {column}"
            )


    else:

        # -------------------------------------------------
        # NUMERIC CONVERSION
        # -------------------------------------------------

        for column in required_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        invalid_values = (
            df[required_columns]
            .isna()
            .sum()
            .sum()
        )


        if invalid_values > 0:

            st.error(
                f"Dataset contains "
                f"{int(invalid_values)} invalid values."
            )


        else:

            dataset_valid = True

            st.success(
                "✓ Dataset validated successfully"
            )


            info1, info2, info3, info4 = (
                st.columns(4)
            )


            with info1:

                st.metric(
                    "Samples",
                    len(df)
                )


            with info2:

                st.metric(
                    "Columns",
                    len(df.columns)
                )


            with info3:

                duration = (
                    df["Time (s)"].max()
                    -
                    df["Time (s)"].min()
                )

                st.metric(
                    "Duration",
                    f"{duration:.2f} s"
                )


            with info4:

                if len(df) > 1:

                    interval = (
                        df["Time (s)"]
                        .diff()
                        .dropna()
                        .median()
                    )

                    st.metric(
                        "Sampling",
                        f"{interval:.3f} s"
                    )


# =========================================================
# FEATURE 2 — REAL-TIME REPLAY
# =========================================================

st.divider()

st.subheader("Telemetry Replay")


c1, c2, c3, c4 = st.columns(
    [1.2, 1.2, 1.2, 2]
)


with c1:

    start_button = st.button(
        "▶ START",
        disabled=not dataset_valid,
        use_container_width=True,
        key="replay_start_button"
    )


with c2:

    pause_button = st.button(
        "⏸ PAUSE",
        use_container_width=True,
        key="replay_pause_button"
    )


with c3:

    reset_button = st.button(
        "↻ RESET",
        use_container_width=True,
        key="replay_reset_button"
    )


with c4:

    replay_speed = st.selectbox(
        "Replay Speed",
        [0.5, 1.0, 2.0, 5.0],
        index=1,
        format_func=lambda x: f"{x:g}×"
    )


# =========================================================
# BUTTON ACTIONS
# =========================================================

if start_button:

    st.session_state.replay_running = True


if pause_button:

    st.session_state.replay_running = False


if reset_button:

    st.session_state.replay_running = False

    st.session_state.replay_index = 0

    st.session_state.backend.reset()

    st.session_state.history = []

    st.session_state.last_result = None


# =========================================================
# PROCESS CURRENT SAMPLE
# =========================================================

result = st.session_state.last_result


if (
    dataset_valid
    and st.session_state.replay_index < len(df)
):

    index = st.session_state.replay_index

    row = df.iloc[index]


    current_time = float(
        row["Time (s)"]
    )

    inlet_pressure = float(
        row["Inlet Pressure (Bar)"]
    )

    outlet_pressure = float(
        row["Outlet Pressure (Bar)"]
    )


    # Process sample only once
    if len(st.session_state.history) <= index:

        result = (
            st.session_state.backend.process_sample(
                current_time,
                inlet_pressure,
                outlet_pressure
            )
        )

        st.session_state.history.append(
            result
        )

        st.session_state.last_result = result


# =========================================================
# CURRENT TELEMETRY
# =========================================================

st.subheader("Current Telemetry")


if result is not None:

    current_time = result["time"]

    inlet_pressure = result[
        "inlet_pressure"
    ]

    outlet_pressure = result[
        "outlet_pressure"
    ]

else:

    current_time = 0.0

    inlet_pressure = 0.0

    outlet_pressure = 0.0


live1, live2, live3 = st.columns(3)


with live1:

    st.metric(
        "Current Time",
        f"{current_time:.2f} s"
    )


with live2:

    st.metric(
        "Inlet Pressure",
        f"{inlet_pressure:.2f} bar"
    )


with live3:

    st.metric(
        "Outlet Pressure",
        f"{outlet_pressure:.2f} bar"
    )


# =========================================================
# PROGRESS
# =========================================================

if dataset_valid:

    current_sample = min(
        st.session_state.replay_index + 1,
        len(df)
    )

    progress = (
        current_sample / len(df)
    )

    st.progress(progress)

    st.caption(
        f"Sample {current_sample} / {len(df)}"
    )


# ============================================================
# DIGITAL TWIN — TIME MACHINE
# ============================================================

st.markdown("---")

st.markdown(
    "<h2 style='margin-bottom:5px;'>DIGITAL TWIN — TIME MACHINE</h2>",
    unsafe_allow_html=True
)

# Dataset duration
if st.session_state.df is not None:

    total_samples = len(st.session_state.df)

    if total_samples > 0:

        current_index = min(
            st.session_state.replay_index,
            total_samples - 1
        )

        current_time = float(
            st.session_state.df.iloc[current_index]["Time (s)"]
        )

        max_time = float(
            st.session_state.df.iloc[-1]["Time (s)"]
        )

        # ----------------------------------------------------
        # TIME DISPLAY
        # ----------------------------------------------------

        time_col1, time_col2 = st.columns([1, 3])

        with time_col1:

            st.metric(
                "SIMULATION TIME",
                f"{current_time:.2f} s"
            )

        with time_col2:

            st.progress(
                min(
                    current_index / max(total_samples - 1, 1),
                    1.0
                )
            )

        # ----------------------------------------------------
        # TIMELINE SLIDER
        # ----------------------------------------------------

        selected_index = st.slider(
            "TIME POSITION",
            min_value=0,
            max_value=total_samples - 1,
            value=current_index,
            step=1,
            format="%d"
        )

        # If user manually moves the slider
        if selected_index != st.session_state.replay_index:

            st.session_state.replay_index = selected_index
            st.session_state.backend.reset()
            st.session_state.history = []
            st.session_state.last_result = None

            # Re-process everything up to selected position
            for i in range(selected_index + 1):

                row = st.session_state.df.iloc[i]

                result = st.session_state.backend.process_sample(
                    row["Time (s)"],
                    row["Inlet Pressure (Bar)"],
                    row["Outlet Pressure (Bar)"]
                )

                st.session_state.history.append(result)

            if st.session_state.history:

                st.session_state.last_result = (
                    st.session_state.history[-1]
                )

            st.rerun()

        # ----------------------------------------------------
        # CONTROLS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            if st.button(
                "▶ START",
                use_container_width=True,
                key="time_machine_start_button"
            ):

                st.session_state.replay_running = True
                st.rerun()

        with c2:

            if st.button(
                "Ⅱ PAUSE",
                use_container_width=True,
                key="time_machine_pause_button"
            ):

                st.session_state.replay_running = False
                st.rerun()

        with c3:

            if st.button(
                "↻ RESET",
                use_container_width=True,
                key="time_machine_reset_button"
            ):

                st.session_state.replay_running = False
                st.session_state.replay_index = 0
                st.session_state.backend.reset()
                st.session_state.history = []
                st.session_state.last_result = None

                st.rerun()

        with c4:

            speed = st.selectbox(
                "SPEED",
                [0.5, 1.0, 2.0, 5.0],
                index=1,
                format_func=lambda x: f"{x}×"
            )

        # ----------------------------------------------------
        # TIME MACHINE STATUS
        # ----------------------------------------------------

        if st.session_state.replay_running:

            st.success(
                f"● SIMULATION RUNNING — {current_time:.2f} s"
            )

        else:

            st.info(
                f"● SIMULATION PAUSED — {current_time:.2f} s"
            )

        st.caption(
            f"Simulation range: 0.00 s → {max_time:.2f} s"
        )


# =========================================================
# LIVE TELEMETRY STATUS
# =========================================================

st.divider()

st.subheader("Live Telemetry Status")


if result is not None:

    t1, t2, t3, t4 = st.columns(4)


    with t1:

        st.metric(
            "Inlet Pressure",
            f'{result.get("inlet_pressure", 0):.2f} bar'
        )


    with t2:

        st.metric(
            "Outlet Pressure",
            f'{result.get("outlet_pressure", 0):.2f} bar'
        )


    with t3:

        st.metric(
            "Inlet ΔP",
            f'{result.get("inlet_dp", 0):.2f} bar'
        )


    with t4:

        st.metric(
            "Outlet ΔP",
            f'{result.get("outlet_dp", 0):.2f} bar'
        )


else:

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.metric("Inlet Pressure", "—")

    with t2:
        st.metric("Outlet Pressure", "—")

    with t3:
        st.metric("Inlet ΔP", "—")

    with t4:
        st.metric("Outlet ΔP", "—")

# =========================================================
# ANOMALY MONITORING
# =========================================================

st.divider()

st.subheader("Anomaly Monitoring")

if result is not None:

    a1, a2, a3, a4 = st.columns(4)

    # -----------------------------------------------------
    # INLET SCORE
    # -----------------------------------------------------

    with a1:

        st.metric(
            "Inlet Anomaly Score",
            f'{result["inlet_score"]:.2f}'
        )


    # -----------------------------------------------------
    # OUTLET SCORE
    # -----------------------------------------------------

    with a2:

        st.metric(
            "Outlet Anomaly Score",
            f'{result["outlet_score"]:.2f}'
        )


    # -----------------------------------------------------
    # INLET STATUS
    # -----------------------------------------------------

    with a3:

        if result.get("inlet_event") is not None:

            st.error(
                "⚠ INLET TRANSIENT"
            )

        else:

            st.success(
                "✓ INLET NORMAL"
            )


    # -----------------------------------------------------
    # OUTLET STATUS
    # -----------------------------------------------------

    with a4:

        if result.get("outlet_event") is not None:

            st.error(
                "⚠ OUTLET TRANSIENT"
            )

        else:

            st.success(
                "✓ OUTLET NORMAL"
            )

else:

    st.info(
        "Waiting for telemetry..."
    )

# =========================================================
# ANOMALY SCORE GRAPH
# =========================================================

st.subheader("Anomaly Score Trend")

if len(st.session_state.history) > 0:

    anomaly_df = pd.DataFrame(
        st.session_state.history
    )

    anomaly_fig = go.Figure()

    anomaly_fig.add_trace(
        go.Scatter(
            x=anomaly_df["time"],
            y=anomaly_df["inlet_score"],
            mode="lines",
            name="Inlet Score"
        )
    )

    anomaly_fig.add_trace(
        go.Scatter(
            x=anomaly_df["time"],
            y=anomaly_df["outlet_score"],
            mode="lines",
            name="Outlet Score"
        )
    )

    anomaly_fig.add_hline(
        y=5,
        line_dash="dash",
        annotation_text="Detection Threshold"
    )

    anomaly_fig.update_layout(
        height=300,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        paper_bgcolor="#080E14",
        plot_bgcolor="#080E14",
        font=dict(
            color="#DCE5EA"
        ),
        xaxis_title="Time (s)",
        yaxis_title="Anomaly Score",
        hovermode="x unified"
    )

    st.plotly_chart(
        anomaly_fig,
        use_container_width=True
    )

else:

    st.info(
        "Start telemetry replay to generate anomaly scores."
    )
# =========================================================
# ANOMALY MONITORING
# =========================================================

st.subheader("Anomaly Monitoring")


if result is not None:

    a1, a2, a3, a4 = st.columns(4)


    with a1:

        st.metric(
            "Inlet Anomaly Score",
            f'{result.get("inlet_score", 0):.2f}'
        )


    with a2:

        st.metric(
            "Outlet Anomaly Score",
            f'{result.get("outlet_score", 0):.2f}'
        )


    with a3:

        if result.get("inlet_event"):

            st.error(
                "⚠ INLET ANOMALY"
            )

        else:

            st.success(
                "✓ INLET NORMAL"
            )


    with a4:

        if result.get("outlet_event"):

            st.error(
                "⚠ OUTLET ANOMALY"
            )

        else:

            st.success(
                "✓ OUTLET NORMAL"
            )


else:

    st.info(
        "Waiting for telemetry replay..."
    )

# =========================================================
# LEAK LOCALIZATION
# =========================================================

st.divider()

st.subheader("Leak Localization")


if result is not None:

    inlet_event = result.get("inlet_event")
    outlet_event = result.get("outlet_event")

    # -----------------------------------------------------
    # WAITING FOR INLET
    # -----------------------------------------------------

    if inlet_event is None:

        st.info(
            "Waiting for inlet transient..."
        )


    # -----------------------------------------------------
    # INLET DETECTED — WAIT FOR OUTLET
    # -----------------------------------------------------

    elif outlet_event is None:

        tin = inlet_event["time"]

        st.warning(
            f"Inlet transient detected at "
            f"{tin:.2f} s — waiting for outlet confirmation."
        )

        l1, l2, l3 = st.columns(3)

        with l1:

            st.metric(
                "Inlet Detection",
                f"{tin:.2f} s"
            )

        with l2:

            st.metric(
                "Outlet Detection",
                "WAITING"
            )

        with l3:

            st.metric(
                "Leak Coordinate",
                "CALCULATING"
            )


    # -----------------------------------------------------
    # BOTH DETECTED — LOCALIZATION
    # -----------------------------------------------------

    else:

        tin = inlet_event["time"]

        tout = outlet_event["time"]

        delta_t = result.get("delta_t")

        leak_x = result.get("leak_coordinate")

        segment = result.get("segment")


        l1, l2, l3, l4 = st.columns(4)


        with l1:

            st.metric(
                "Inlet Transient",
                f"{tin:.2f} s" if tin is not None else "—"
            )


        with l2:

            st.metric(
                "Outlet Transient",
                f"{tout:.2f} s" if tout is not None else "—"
            )


        with l3:

            st.metric(
                "Propagation Δt",
                f"{delta_t:.2f} s" if delta_t is not None else "—"
            )


        with l4:

            st.metric(
                "Leak Coordinate",
                f"{leak_x:.0f} m" if leak_x is not None else "—"
            )


        # -------------------------------------------------
        # NPW CALCULATION
        # -------------------------------------------------

        if delta_t is not None and leak_x is not None and segment is not None:

            st.markdown(
                f"""
                ### NPW Localization

                **Pipeline length (L):** 10000 m

                **Pressure wave speed (C):** 1000 m/s

                **Propagation time (Δt):** {delta_t:.2f} s

                **Localization equation:**

                `X = (L − C × Δt) / 2`

                `X = (10000 − 1000 × {delta_t:.2f}) / 2`

                **Leak coordinate: {leak_x:.0f} m ({leak_x / 1000:.2f} km)**

                **Affected segment: {segment}**
                """
            )


            st.success(
                f"✓ LEAK LOCALIZED — "
                f"{leak_x:.0f} m — {segment}"
            )

        else:

            st.warning("Localization data incomplete")


else:

    st.info(
        "Start telemetry replay to perform localization."
    )
# =========================================================
# LIVE PRESSURE GRAPH
# =========================================================

st.divider()

st.subheader("Live Pressure Monitoring")


if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )


    fig_pressure = go.Figure()


    fig_pressure.add_trace(
        go.Scatter(
            x=history_df["time"],
            y=history_df["inlet_pressure"],
            mode="lines",
            name="Inlet Pressure"
        )
    )


    fig_pressure.add_trace(
        go.Scatter(
            x=history_df["time"],
            y=history_df["outlet_pressure"],
            mode="lines",
            name="Outlet Pressure"
        )
    )


    fig_pressure.update_layout(
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        paper_bgcolor="#080E14",
        plot_bgcolor="#080E14",
        font=dict(
            color="#DCE5EA"
        ),
        xaxis_title="Time (s)",
        yaxis_title="Pressure (bar)",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig_pressure,
        use_container_width=True
    )


# =========================================================
# LOCALIZATION → 3D MODEL CONNECTION
# =========================================================

if (
    result is not None
    and result.get("leak_coordinate") is not None
    and result.get("segment") is not None
    and result.get("state") in [
        "LOCALIZED",
        "ISOLATED"
    ]
):

    leak_coordinate = float(
        result.get("leak_coordinate", 0)
    )

    active_segment = result.get(
        "segment"
    )

else:

    leak_coordinate = None

    active_segment = None

# =========================================================
# NPW EQUATION
# =========================================================

if (
    result is not None
    and result.get("delta_t") is not None
    and result.get("leak_coordinate") is not None
):

    st.markdown(
        f"""
        **NPW Localization**

        `X = (L − C × Δt) / 2`

        `X = (10000 − 1000 × {result.get("delta_t", 0):.2f}) / 2`

        **Leak coordinate = `
        {result.get("leak_coordinate", 0):.0f} m`**
        """
    )


# =========================================================
# 3D DIGITAL TWIN
# PIPELINE VISUALIZATION ONLY
# =========================================================

st.divider()

st.subheader("3D Digital Twin")


# ---------------------------------------------------------
# PIPELINE CONFIGURATION
# ---------------------------------------------------------

PIPELINE_LENGTH = 10000
PIPE_RADIUS = 0.32
PIPE_SIDES = 40

segments = [
    (0, 2000, "SEGMENT 1"),
    (2000, 4000, "SEGMENT 2"),
    (4000, 6000, "SEGMENT 3"),
    (6000, 8000, "SEGMENT 4"),
    (8000, 10000, "SEGMENT 5")
]


# ---------------------------------------------------------
# GET CURRENT LOCALIZATION
# ---------------------------------------------------------

if result is not None:

    pipeline_state = result.get(
        "state",
        "NORMAL"
    )

    # Backend currently returns leak_coordinate.
    # Keep fallback for older result structure.
    leak_coordinate = result.get(
        "leak_coordinate",
        result.get("leak_coordinate")
    )

    active_segment = result.get(
        "segment"
    )

else:

    pipeline_state = "NORMAL"

    leak_coordinate = None

    active_segment = None


# Only show fault after localization
if pipeline_state not in [
    "LOCALIZED",
    "ISOLATED"
]:

    leak_coordinate = None
    active_segment = None


# ---------------------------------------------------------
# CREATE FIGURE
# ---------------------------------------------------------

fig_3d = go.Figure()


# =========================================================
# CYLINDER GENERATOR
# =========================================================

def create_pipe_segment(
    start,
    end,
    radius,
    color,
    name
):

    theta = np.linspace(
        0,
        2 * np.pi,
        PIPE_SIDES,
        endpoint=False
    )

    # Two circular rings
    x = np.concatenate([
        np.full(PIPE_SIDES, start),
        np.full(PIPE_SIDES, end)
    ])

    y = np.concatenate([
        radius * np.cos(theta),
        radius * np.cos(theta)
    ])

    z = np.concatenate([
        radius * np.sin(theta),
        radius * np.sin(theta)
    ])

    i = []
    j = []
    k = []

    # -----------------------------------------------------
    # Cylindrical surface
    # -----------------------------------------------------

    for n in range(PIPE_SIDES):

        nxt = (n + 1) % PIPE_SIDES

        # Front/back indices
        a = n
        b = nxt

        c = PIPE_SIDES + n
        d = PIPE_SIDES + nxt

        # Triangle 1
        i.append(a)
        j.append(b)
        k.append(c)

        # Triangle 2
        i.append(b)
        j.append(d)
        k.append(c)

    # -----------------------------------------------------
    # Front cap
    # -----------------------------------------------------

    front_center = len(x)

    x = np.append(x, start)
    y = np.append(y, 0)
    z = np.append(z, 0)

    for n in range(PIPE_SIDES):

        nxt = (n + 1) % PIPE_SIDES

        i.append(front_center)
        j.append(nxt)
        k.append(n)

    # -----------------------------------------------------
    # Back cap
    # -----------------------------------------------------

    back_center = len(x)

    x = np.append(x, end)
    y = np.append(y, 0)
    z = np.append(z, 0)

    for n in range(PIPE_SIDES):

        nxt = (n + 1) % PIPE_SIDES

        i.append(back_center)
        j.append(
            PIPE_SIDES + n
        )
        k.append(
            PIPE_SIDES + nxt
        )

    return go.Mesh3d(

        x=x,
        y=y,
        z=z,

        i=i,
        j=j,
        k=k,

        color=color,

        opacity=1.0,

        flatshading=False,

        lighting=dict(
            ambient=0.38,
            diffuse=0.82,
            specular=0.85,
            roughness=0.28,
            fresnel=0.12
        ),

        lightposition=dict(
            x=3000,
            y=-3000,
            z=5000
        ),

        name=name,

        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Range: {start:.0f} – {end:.0f} m"
            "<extra></extra>"
        ),

        showlegend=False
    )


# =========================================================
# PIPE SEGMENTS
# =========================================================

for start, end, name in segments:

    # Normal metallic pipe
    segment_color = "#526773"

    # Highlight ONLY affected segment
    if name == active_segment:

        segment_color = "#A94B48"

    fig_3d.add_trace(

        create_pipe_segment(
            start,
            end,
            PIPE_RADIUS,
            segment_color,
            name
        )

    )


# =========================================================
# SEGMENT JOINTS / FLANGES
# =========================================================

for position in [
    0,
    2000,
    4000,
    6000,
    8000,
    10000
]:

    theta = np.linspace(
        0,
        2 * np.pi,
        48
    )

    flange_radius = PIPE_RADIUS + 0.07

    y = flange_radius * np.cos(theta)
    z = flange_radius * np.sin(theta)

    fig_3d.add_trace(

        go.Scatter3d(

            x=np.full_like(
                theta,
                position
            ),

            y=y,
            z=z,

            mode="lines",

            line=dict(
                color="#71838D",
                width=5
            ),

            hoverinfo="skip",

            showlegend=False
        )

    )


# =========================================================
# FLOW DIRECTION ARROWS
# =========================================================

flow_positions = [
    700,
    2700,
    4700,
    6700,
    8700
]

for position in flow_positions:

    fig_3d.add_trace(

        go.Cone(

            x=[position],

            y=[0],

            z=[PIPE_RADIUS + 0.02],

            u=[500],

            v=[0],

            w=[0],

            sizemode="absolute",

            sizeref=0.45,

            anchor="tail",

            colorscale=[
                [0, "#A9BAC3"],
                [1, "#A9BAC3"]
            ],

            showscale=False,

            opacity=0.9,

            hovertemplate=(
                "<b>FLOW DIRECTION</b>"
                "<extra></extra>"
            )
        )

    )


# =========================================================
# PIPELINE COORDINATE AXIS
# =========================================================

fig_3d.add_trace(

    go.Scatter3d(

        x=[
            0,
            PIPELINE_LENGTH
        ],

        y=[
            -0.75,
            -0.75
        ],

        z=[
            -0.55,
            -0.55
        ],

        mode="lines",

        line=dict(
            color="#64757E",
            width=2
        ),

        hoverinfo="skip",

        showlegend=False
    )

)


# ---------------------------------------------------------
# Coordinate tick marks
# ---------------------------------------------------------

for position in [
    0,
    2000,
    4000,
    6000,
    8000,
    10000
]:

    fig_3d.add_trace(

        go.Scatter3d(

            x=[
                position,
                position
            ],

            y=[
                -0.75,
                -0.75
            ],

            z=[
                -0.55,
                -0.68
            ],

            mode="lines",

            line=dict(
                color="#64757E",
                width=2
            ),

            hoverinfo="skip",

            showlegend=False
        )

    )


# =========================================================
# SEGMENT LABELS
# =========================================================

for start, end, name in segments:

    center = (
        start + end
    ) / 2

    fig_3d.add_trace(

        go.Scatter3d(

            x=[center],

            y=[0],

            z=[0.72],

            mode="text",

            text=[
                f"{name}<br>"
                f"{start:.0f}–{end:.0f} m"
            ],

            textfont=dict(
                size=11,
                color="#B9C7CE"
            ),

            hoverinfo="skip",

            showlegend=False
        )

    )


# =========================================================
# COORDINATE LABELS
# =========================================================

for position in [
    0,
    2000,
    4000,
    6000,
    8000,
    10000
]:

    fig_3d.add_trace(

        go.Scatter3d(

            x=[position],

            y=[-0.75],

            z=[-0.78],

            mode="text",

            text=[
                f"{position} m"
            ],

            textfont=dict(
                size=10,
                color="#82939C"
            ),

            hoverinfo="skip",

            showlegend=False
        )

    )


# =========================================================
# INLET
# =========================================================

fig_3d.add_trace(

    go.Scatter3d(

        x=[0],

        y=[0],

        z=[0],

        mode="markers+text",

        marker=dict(
            size=8,
            color="#65B891",
            symbol="diamond"
        ),

        text=[
            "INLET"
        ],

        textposition="bottom center",

        textfont=dict(
            size=11,
            color="#AFC0C8"
        ),

        hovertemplate=(
            "<b>INLET</b><br>"
            "Coordinate: 0 m"
            "<extra></extra>"
        ),

        showlegend=False
    )

)


# =========================================================
# OUTLET
# =========================================================

fig_3d.add_trace(

    go.Scatter3d(

        x=[PIPELINE_LENGTH],

        y=[0],

        z=[0],

        mode="markers+text",

        marker=dict(
            size=8,
            color="#65B891",
            symbol="diamond"
        ),

        text=[
            "OUTLET"
        ],

        textposition="bottom center",

        textfont=dict(
            size=11,
            color="#AFC0C8"
        ),

        hovertemplate=(
            "<b>OUTLET</b><br>"
            "Coordinate: 10000 m"
            "<extra></extra>"
        ),

        showlegend=False
    )

)


# =========================================================
# LEAK LOCATION
# =========================================================

if leak_coordinate is not None:

    leak_coordinate = float(
        leak_coordinate
    )

    # -----------------------------------------------------
    # Vertical fault marker
    # -----------------------------------------------------

    fig_3d.add_trace(

        go.Scatter3d(

            x=[
                leak_coordinate,
                leak_coordinate
            ],

            y=[
                0,
                0
            ],

            z=[
                PIPE_RADIUS,
                PIPE_RADIUS + 0.65
            ],

            mode="lines",

            line=dict(
                color="#D6534D",
                width=5
            ),

            hoverinfo="skip",

            showlegend=False
        )

    )


    # -----------------------------------------------------
    # Leak point
    # -----------------------------------------------------

    fig_3d.add_trace(

        go.Scatter3d(

            x=[leak_coordinate],

            y=[0],

            z=[PIPE_RADIUS + 0.05],

            mode="markers+text",

            marker=dict(
                size=12,
                color="#D6534D",
                symbol="diamond",
                line=dict(
                    color="#E8EEF1",
                    width=2
                )
            ),

            text=[
                f"FAULT<br>"
                f"{leak_coordinate:.0f} m"
            ],

            textposition="top center",

            textfont=dict(
                size=12,
                color="#D6534D"
            ),

            hovertemplate=(
                "<b>FAULT LOCATION</b><br>"
                f"Coordinate: "
                f"{leak_coordinate:.0f} m"
                "<extra></extra>"
            ),

            showlegend=False
        )

    )


    # =====================================================
    # LEAK DIRECTION ARROW
    # =====================================================

    fig_3d.add_trace(

        go.Cone(

            x=[leak_coordinate],

            y=[0],

            z=[0.05],

            u=[0],

            v=[0],

            w=[-0.9],

            sizemode="absolute",

            sizeref=0.38,

            anchor="tail",

            colorscale=[
                [0, "#D6534D"],
                [1, "#D6534D"]
            ],

            showscale=False,

            opacity=1.0,

            hovertemplate=(
                "<b>LEAK DIRECTION</b>"
                "<extra></extra>"
            )
        )

    )


    # -----------------------------------------------------
    # Leak direction label
    # -----------------------------------------------------

    fig_3d.add_trace(

        go.Scatter3d(

            x=[leak_coordinate],

            y=[0],

            z=[-0.55],

            mode="text",

            text=[
                "LEAK DIRECTION"
            ],

            textfont=dict(
                size=10,
                color="#D6534D"
            ),

            hoverinfo="skip",

            showlegend=False
        )

    )


# =========================================================
# 3D SCENE
# =========================================================

fig_3d.update_layout(

    height=500,

    margin=dict(
        l=0,
        r=0,
        t=10,
        b=0
    ),

    paper_bgcolor="#080E14",

    scene=dict(

        bgcolor="#07131B",

        aspectmode="manual",

        aspectratio=dict(
            x=5.8,
            y=1,
            z=1
        ),

        camera=dict(
            eye=dict(
                x=1.55,
                y=1.35,
                z=0.75
            ),

            center=dict(
                x=0,
                y=0,
                z=0
            )
        ),

        xaxis=dict(
            title="Pipeline Distance (m)",

            range=[
                -500,
                PIPELINE_LENGTH + 500
            ],

            showgrid=False,

            zeroline=False,

            showbackground=False,

            color="#7F919A",

            tickmode="array",

            tickvals=[
                0,
                2000,
                4000,
                6000,
                8000,
                10000
            ],

            ticktext=[
                "0",
                "2000",
                "4000",
                "6000",
                "8000",
                "10000"
            ]
        ),

        yaxis=dict(
            visible=False
        ),

        zaxis=dict(
            visible=False
        )
    ),

    showlegend=False
)


st.plotly_chart(

    fig_3d,

    use_container_width=True,

    config={
        "displayModeBar": False
    }
)


# ---------------------------------------------------------
# IMPORTANT:
# Only show localized segment after localization.
# ---------------------------------------------------------

if (
    result is not None
    and result.get("state") in [
        "LOCALIZED",
        "ISOLATED"
    ]
):

    leak_coordinate = (
        result.get("leak_coordinate")
    )

    active_segment = (
        result.get("segment")
    )

else:

    leak_coordinate = None

    active_segment = None


fig_3d = go.Figure()


# =========================================================
# PIPE CREATION
# =========================================================

def create_pipe_segment(
    start,
    end,
    radius,
    color,
    name
):

    theta = np.linspace(
        0,
        2 * np.pi,
        PIPE_SIDES
    )


    x_values = np.concatenate([
        np.full(
            PIPE_SIDES,
            start
        ),
        np.full(
            PIPE_SIDES,
            end
        )
    ])


    y_values = np.concatenate([
        radius * np.cos(theta),
        radius * np.cos(theta)
    ])


    z_values = np.concatenate([
        radius * np.sin(theta),
        radius * np.sin(theta)
    ])


    i = []
    j = []
    k = []


    for n in range(
        PIPE_SIDES - 1
    ):

        i.append(n)
        j.append(n + 1)
        k.append(PIPE_SIDES + n)

        i.append(n + 1)
        j.append(
            PIPE_SIDES + n + 1
        )
        k.append(PIPE_SIDES + n)


    return go.Mesh3d(
        x=x_values,
        y=y_values,
        z=z_values,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=1.0,
        lighting=dict(
            ambient=0.45,
            diffuse=0.75,
            specular=0.65
        ),
        name=name,
        hovertemplate=(
            name +
            "<extra></extra>"
        )
    )


# =========================================================
# PIPE SEGMENTS
# =========================================================

for start, end, name in segments:

    if name == active_segment:

        segment_color = "#B44D48"

    else:

        segment_color = "#526B7A"


    fig_3d.add_trace(
        create_pipe_segment(
            start,
            end,
            PIPE_RADIUS,
            segment_color,
            name
        )
    )


# =========================================================
# LEAK MARKER
# =========================================================

if leak_coordinate is not None:

    fig_3d.add_trace(
        go.Scatter3d(
            x=[leak_coordinate],
            y=[0],
            z=[PIPE_RADIUS + 0.2],
            mode="markers+text",
            marker=dict(
                size=14,
                color="#D6534D"
            ),
            text=[
                f"LEAK<br>"
                f"{leak_coordinate / 1000:.2f} km"
            ],
            textposition="top center",
            showlegend=False
        )
    )


# =========================================================
# INLET
# =========================================================

fig_3d.add_trace(
    go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers+text",
        marker=dict(
            size=8,
            color="#9AAAB4"
        ),
        text=["INLET"],
        textposition="bottom center",
        showlegend=False
    )
)


# =========================================================
# OUTLET
# =========================================================

fig_3d.add_trace(
    go.Scatter3d(
        x=[PIPELINE_LENGTH],
        y=[0],
        z=[0],
        mode="markers+text",
        marker=dict(
            size=8,
            color="#9AAAB4"
        ),
        text=["OUTLET"],
        textposition="bottom center",
        showlegend=False
    )
)


# =========================================================
# 3D LAYOUT
# =========================================================

fig_3d.update_layout(

    height=500,

    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),

    paper_bgcolor="#080E14",

    scene=dict(

        bgcolor="#080E14",

        aspectmode="manual",

        aspectratio=dict(
            x=5.5,
            y=1,
            z=1
        ),

        camera=dict(
            eye=dict(
                x=1.7,
                y=1.25,
                z=0.75
            )
        ),

        xaxis=dict(
            visible=False
        ),

        yaxis=dict(
            visible=False
        ),

        zaxis=dict(
            visible=False
        )
    ),

    showlegend=False
)


st.plotly_chart(
    fig_3d,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# PIPELINE STATE
# =========================================================

st.divider()

st.subheader("Pipeline State")


current_state = (
    result.get("state", "NORMAL")
    if result is not None
    else "NORMAL"
)


s1, s2, s3, s4 = st.columns(4)


with s1:

    if current_state == "NORMAL":
        st.success("● NORMAL")
    else:
        st.caption("○ NORMAL")


with s2:

    if current_state == "ANOMALY":
        st.warning("● ANOMALY")
    else:
        st.caption("○ ANOMALY")


with s3:

    if current_state == "LOCALIZED":
        st.error("● LOCALIZED")
    else:
        st.caption("○ LOCALIZED")


with s4:

    if current_state == "ISOLATED":
        st.error("● ISOLATED")
    else:
        st.caption("○ ISOLATED")


# =========================================================
# AUTOMATED RESPONSE
# =========================================================

st.divider()

st.subheader("Automated Response")


r1, r2, r3 = st.columns(3)


with r1:

    if (
        result is not None
        and result.get("alarm")
    ):

        st.error(
            "🚨 ALARM ACTIVE"
        )

    else:

        st.success(
            "✓ ALARM OFF"
        )


with r2:

    isolation_active = (
        result is not None
        and (
            result.get("isolation_complete")
            or result.get("isolation_required")
            or result.get("state") == "ISOLATED"
        )
    )

    if isolation_active:

        st.error(
            "⛔ ISOLATION ACTIVE"
        )

    else:

        st.info(
            "ISOLATION STANDBY"
        )


with r3:

    if (
        result is not None
        and result.get("segment")
    ):

        st.metric(
            "Affected Segment",
            result.get("segment")
        )

    else:

        st.metric(
            "Affected Segment",
            "NONE"
        )


# =========================================================
# ADVANCE REPLAY
# =========================================================

if (
    st.session_state.replay_running
    and dataset_valid
):

    if (
        st.session_state.replay_index
        < len(df) - 1
    ):

        st.session_state.replay_index += 1

        time.sleep(
            0.10 / replay_speed
        )

        st.rerun()

    else:

        st.session_state.replay_running = False

        # ============================================================
# AUTOMATIC INCIDENT REPORT
# ============================================================

st.markdown("---")

st.markdown(
    "<h2 style='margin-bottom:5px;'>AUTOMATIC INCIDENT REPORT</h2>",
    unsafe_allow_html=True
)

backend = st.session_state.backend
df = st.session_state.df

if df is not None and st.session_state.history:

    history = st.session_state.history

    # --------------------------------------------------------
    # Extract event information
    # --------------------------------------------------------

    inlet_event = backend.inlet_event
    outlet_event = backend.outlet_event

    leak_coordinate = backend.leak_coordinate
    affected_segment = backend.segment

    # Maximum pressure changes
    max_inlet_dp = min(
        [x["inlet_dp"] for x in history]
    )

    max_outlet_dp = min(
        [x["outlet_dp"] for x in history]
    )

    # Maximum anomaly scores
    max_inlet_score = max(
        [x["inlet_score"] for x in history]
    )

    max_outlet_score = max(
        [x["outlet_score"] for x in history]
    )

    # Final state
    final_state = backend.state

    # --------------------------------------------------------
    # Report preview
    # --------------------------------------------------------

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "DATA POINTS",
            len(history)
        )

    with r2:
        st.metric(
            "MAX INLET ΔP",
            f"{max_inlet_dp:.2f} bar"
        )

    with r3:
        st.metric(
            "MAX OUTLET ΔP",
            f"{max_outlet_dp:.2f} bar"
        )

    with r4:
        st.metric(
            "FINAL STATE",
            final_state
        )

    st.markdown("### Incident Summary")

    if inlet_event:

        detection_time = inlet_event["time"]

        st.write(
            f"**Detection Time:** {detection_time:.2f} s"
        )

        st.write(
            f"**Inlet Anomaly Score:** "
            f"{inlet_event['score']:.2f}"
        )

    else:

        st.write("**Detection Time:** No confirmed event")

    if outlet_event:

        st.write(
            f"**Outlet Confirmation:** "
            f"{outlet_event['time']:.2f} s"
        )

        st.write(
            f"**Outlet Anomaly Score:** "
            f"{outlet_event['score']:.2f}"
        )

    else:

        st.write(
            "**Outlet Confirmation:** Not confirmed"
        )

    if leak_coordinate is not None:

        st.write(
            f"**Leak Coordinate:** "
            f"{leak_coordinate:.0f} m"
        )

        st.write(
            f"**Affected Segment:** "
            f"{affected_segment}"
        )

    else:

        st.write(
            "**Leak Coordinate:** Not localized"
        )

    st.markdown("### System Response")

    response_items = []

    if inlet_event:
        response_items.append("✓ Inlet transient detected")

    if outlet_event:
        response_items.append("✓ Outlet transient confirmed")

    if backend.localization:
        response_items.append("✓ Event localized")

    if backend.isolation_initiated:
        response_items.append("✓ Virtual isolation initiated")

    if backend.isolation_complete:
        response_items.append("✓ Virtual isolation completed")

    if not response_items:
        response_items.append("No incident response triggered")

    for item in response_items:
        st.write(item)

    # --------------------------------------------------------
    # Generate downloadable report
    # --------------------------------------------------------

    report = f"""
BRAIN BOLT — SUBSEA PIPELINE DIGITAL TWIN
AUTOMATIC INCIDENT REPORT
============================================

SIMULATION
--------------------------------------------
Dataset: {st.session_state.uploaded_name}
Data Points: {len(history)}
Simulation Duration: {float(df.iloc[-1]["Time (s)"]):.2f} s

DETECTION
--------------------------------------------
Inlet Transient:
{f"Detected at {inlet_event['time']:.2f} s" if inlet_event else "Not detected"}

Inlet Anomaly Score:
{max_inlet_score:.2f}

Maximum Inlet Pressure Change:
{max_inlet_dp:.2f} bar

Outlet Transient:
{f"Confirmed at {outlet_event['time']:.2f} s" if outlet_event else "Not confirmed"}

Outlet Anomaly Score:
{max_outlet_score:.2f}

Maximum Outlet Pressure Change:
{max_outlet_dp:.2f} bar

LOCALIZATION
--------------------------------------------
Leak Coordinate:
{f"{leak_coordinate:.0f} m" if leak_coordinate is not None else "Not localized"}

Affected Segment:
{affected_segment if affected_segment else "Not determined"}

Propagation Delta:
{f"{backend.delta_t:.2f} s" if backend.delta_t is not None else "N/A"}

SYSTEM RESPONSE
--------------------------------------------
Virtual Isolation Required:
{"YES" if backend.isolation_required else "NO"}

Isolation Initiated:
{"YES" if backend.isolation_initiated else "NO"}

Isolation Completed:
{"YES" if backend.isolation_complete else "NO"}

Isolated Segment:
{backend.isolated_segment if backend.isolated_segment else "N/A"}

FINAL SYSTEM STATE
--------------------------------------------
{final_state}

EVENT TIMELINE
--------------------------------------------
"""

    for event in backend.events:

        report += (
            f"{event['time']:.2f} s | "
            f"{event['type']} | "
            f"{event['message']}\n"
        )

    report += """
============================================
Generated by SeaAi
Subsea Pipeline Digital Twin
"""

    download_data = report.encode("utf-8")
    image_bytes = None

    if leak_coordinate is not None and affected_segment:

        try:

            image_buffer = io.BytesIO()
            pio.write_image(
                fig_3d,
                file=image_buffer,
                format="png",
                engine="kaleido"
            )
            image_bytes = image_buffer.getvalue()

        except Exception:
            image_bytes = None

    if image_bytes is not None:

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "Brain_Bolt_Incident_Report.txt",
                report
            )
            archive.writestr(
                "faulty_segment.png",
                image_bytes
            )

        download_data = zip_buffer.getvalue()

        st.download_button(
            label="⬇ DOWNLOAD INCIDENT REPORT",
            data=download_data,
            file_name="Brain_Bolt_Incident_Report.zip",
            mime="application/zip",
            use_container_width=True
        )

    else:

        st.download_button(
            label="⬇ DOWNLOAD INCIDENT REPORT",
            data=download_data,
            file_name="Brain_Bolt_Incident_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

else:

    st.info(
        "Run the Time Machine to generate an incident report."
    )