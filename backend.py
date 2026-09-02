import numpy as np


class PipelineBackend:

    def __init__(self):
        self.pipeline_length = 10000.0
        self.wave_speed = 1000.0
        self.num_segments = 5

        self.inlet_baseline = 60.0
        self.outlet_baseline = 55.1

        self.anomaly_threshold = 2.5
        self.confirmation_threshold = 2.5

        self.reset()

    def reset(self):
        self.previous_inlet = None
        self.previous_outlet = None

        self.inlet_dp_history = []
        self.outlet_dp_history = []

        self.inlet_event = None
        self.outlet_event = None
        self.outlet_candidate = None

        self.localization = None
        self.leak_coordinate = None
        self.segment = None
        self.delta_t = None

        self.state = "NORMAL"
        self.alarm = False

        # Virtual isolation
        self.isolation_required = False
        self.isolation_initiated = False
        self.isolation_complete = False
        self.isolated_segment = None

        self.events = []

        self.max_inlet_score = 0.0
        self.max_outlet_score = 0.0

    # ---------------------------------------------------------
    # MODIFIED Z-SCORE
    # ---------------------------------------------------------

    def modified_z_score(self, value, history):

        if len(history) < 3:
            return 0.0

        history_array = np.asarray(history, dtype=float)
        median = float(np.median(history_array))
        mad = float(np.median(np.abs(history_array - median)))

        if mad > 1e-9:
            return float(
                0.6745 * abs(value - median) / mad
            )

        baseline = float(np.std(history_array, ddof=0))

        if baseline > 1e-9:
            return float(
                abs(value - median) / baseline
            )

        if abs(value - median) > 1.0:
            return float(abs(value - median))

        return 0.0

    # ---------------------------------------------------------
    # HEALTH
    # ---------------------------------------------------------

    def health_status(self, pressure, baseline):

        ratio = pressure / baseline

        if ratio >= 0.95:
            return "GREEN"

        elif ratio >= 0.80:
            return "YELLOW"

        elif ratio >= 0.60:
            return "ORANGE"

        else:
            return "RED"

    # ---------------------------------------------------------
    # SEGMENT
    # ---------------------------------------------------------

    def get_segment(self, coordinate):

        segment_length = self.pipeline_length / self.num_segments

        segment_number = int(coordinate // segment_length) + 1

        segment_number = min(
            max(segment_number, 1),
            self.num_segments
        )

        return f"SEGMENT {segment_number}"

    # ---------------------------------------------------------
    # LOCALIZATION
    # ---------------------------------------------------------

    def localize(self):

        if self.inlet_event is None:
            return

        if self.outlet_event is None:
            return

        tin = self.inlet_event["time"]
        tout = self.outlet_event["time"]

        self.delta_t = tout - tin

        if self.delta_t <= 0:
            return

        X = (
            self.pipeline_length
            - self.wave_speed * self.delta_t
        ) / 2

        X = max(
            0,
            min(self.pipeline_length, X)
        )

        self.leak_coordinate = X
        self.segment = self.get_segment(X)

        self.localization = {
            "inlet_time": tin,
            "outlet_time": tout,
            "delta_t": self.delta_t,
            "coordinate": X,
            "segment": self.segment
        }

        self.state = "LOCALIZED"
        self.alarm = True

        self.events.append({
            "time": tout,
            "type": "LOCALIZATION",
            "message": f"Leak localized at {X:.0f} m"
        })

        self.events.append({
            "time": tout,
            "type": "SEGMENT",
            "message": f"Affected {self.segment}"
        })

    # ---------------------------------------------------------
    # VIRTUAL ISOLATION
    # ---------------------------------------------------------

    def initiate_isolation(self, time):

        if self.leak_coordinate is None:
            return

        if self.isolation_initiated:
            return

        self.isolation_required = True
        self.isolation_initiated = True
        self.isolated_segment = self.segment

        self.events.append({
            "time": time,
            "type": "ISOLATION_INITIATED",
            "message": (
                f"Virtual isolation initiated for "
                f"{self.isolated_segment}"
            )
        })

        # Simulated valve closure
        self.isolation_complete = True

        self.events.append({
            "time": time,
            "type": "ISOLATION_COMPLETE",
            "message": (
                f"Virtual isolation completed for "
                f"{self.isolated_segment}"
            )
        })

        self.state = "ISOLATED"

    # ---------------------------------------------------------
    # PROCESS SAMPLE
    # ---------------------------------------------------------

    def process_sample(
        self,
        time,
        inlet_pressure,
        outlet_pressure
    ):

        inlet_pressure = float(inlet_pressure)
        outlet_pressure = float(outlet_pressure)

        # First sample
        if self.previous_inlet is None:

            self.previous_inlet = inlet_pressure
            self.previous_outlet = outlet_pressure

            return self._result(
                time,
                inlet_pressure,
                outlet_pressure,
                0.0,
                0.0
            )

        # Pressure changes
        inlet_dp = (
            inlet_pressure -
            self.previous_inlet
        )

        outlet_dp = (
            outlet_pressure -
            self.previous_outlet
        )

        # Calculate anomaly scores
        inlet_score = self.modified_z_score(
            inlet_dp,
            self.inlet_dp_history
        )

        outlet_score = self.modified_z_score(
            outlet_dp,
            self.outlet_dp_history
        )

        self.max_inlet_score = max(
            self.max_inlet_score,
            inlet_score
        )

        self.max_outlet_score = max(
            self.max_outlet_score,
            outlet_score
        )

        # Add history AFTER calculating score
        self.inlet_dp_history.append(inlet_dp)
        self.outlet_dp_history.append(outlet_dp)

        # -----------------------------------------------------
        # INLET DETECTION
        # -----------------------------------------------------

        inlet_drop = abs(inlet_dp)
        outlet_drop = abs(outlet_dp)

        if (
            self.inlet_event is None
            and inlet_dp < 0
            and (
                inlet_score >= self.anomaly_threshold
                or inlet_drop >= 1.5
            )
        ):

            self.inlet_event = {
                "time": time,
                "score": inlet_score,
                "dp": inlet_dp
            }

            self.state = "ANOMALY"
            self.alarm = True

            self.events.append({
                "time": time,
                "type": "INLET_TRANSIENT",
                "message": (
                    f"Inlet transient detected "
                    f"(score {inlet_score:.2f})"
                )
            })

        # -----------------------------------------------------
        # OUTLET DETECTION
        # -----------------------------------------------------

        if (
            self.inlet_event is not None
            and self.outlet_event is None
            and time > self.inlet_event["time"]
            and outlet_dp < 0
            and (
                outlet_score >= self.confirmation_threshold
                or outlet_drop >= 1.0
            )
        ):

            # First outlet spike is treated as a candidate.
            # Confirm only when the following sample is also
            # a significant negative transient.
            if self.outlet_candidate is None:
                self.outlet_candidate = {
                    "time": time,
                    "score": outlet_score,
                    "dp": outlet_dp
                }

            else:
                self.outlet_event = {
                    "time": time,
                    "score": outlet_score,
                    "dp": outlet_dp
                }

                self.events.append({
                    "time": time,
                    "type": "OUTLET_TRANSIENT",
                    "message": (
                        f"Outlet transient confirmed "
                        f"(score {outlet_score:.2f})"
                    )
                })

                self.localize()
                self.initiate_isolation(time)

                # -----------------------------------------------------
                # HEALTH
                # -----------------------------------------------------

                inlet_health = self.health_status(
                    inlet_pressure,
                    self.inlet_baseline
                )

                outlet_health = self.health_status(
                    outlet_pressure,
                    self.outlet_baseline
                )

                # -----------------------------------------------------
                # RESULT
                # -----------------------------------------------------

                result = self._result(
                    time,
                    inlet_pressure,
                    outlet_pressure,
                    inlet_dp,
                    outlet_dp,
                    inlet_score,
                    outlet_score,
                    inlet_health,
                    outlet_health
                )

                self.previous_inlet = inlet_pressure
                self.previous_outlet = outlet_pressure

                return result

        # -----------------------------------------------------
        # HEALTH
        # -----------------------------------------------------

        inlet_health = self.health_status(
            inlet_pressure,
            self.inlet_baseline
        )

        outlet_health = self.health_status(
            outlet_pressure,
            self.outlet_baseline
        )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        result = self._result(
            time,
            inlet_pressure,
            outlet_pressure,
            inlet_dp,
            outlet_dp,
            inlet_score,
            outlet_score,
            inlet_health,
            outlet_health
        )

        self.previous_inlet = inlet_pressure
        self.previous_outlet = outlet_pressure

        return result

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    def _result(
        self,
        time,
        inlet_pressure,
        outlet_pressure,
        inlet_dp,
        outlet_dp,
        inlet_score=0.0,
        outlet_score=0.0,
        inlet_health=None,
        outlet_health=None
    ):

        return {

            "time": time,

            "inlet_pressure": inlet_pressure,
            "outlet_pressure": outlet_pressure,

            "inlet_dp": inlet_dp,
            "outlet_dp": outlet_dp,

            "inlet_score": inlet_score,
            "outlet_score": outlet_score,

            "inlet_health": inlet_health,
            "outlet_health": outlet_health,

            "state": self.state,

            "alarm": self.alarm,

            # Localization
            "inlet_event": self.inlet_event,
            "outlet_event": self.outlet_event,

            "delta_t": self.delta_t,
            "leak_coordinate": self.leak_coordinate,
            "segment": self.segment,

            # Isolation
            "isolation_required": self.isolation_required,
            "isolation_initiated": self.isolation_initiated,
            "isolation_complete": self.isolation_complete,
            "isolated_segment": self.isolated_segment,

            "events": self.events.copy(),

            "max_inlet_score": self.max_inlet_score,
            "max_outlet_score": self.max_outlet_score
        }