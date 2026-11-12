"""Custom Mosaico ontology for weld-process telemetry.

Mosaico ships robotics/automotive types (IMU, GPS, RobotJoint, Pose, Image...), but
nothing for an arc. These ``Serializable`` subclasses follow the SDK's custom-ontology
pattern (cf. ``mosaicolabs/examples/bridges/ros/custom_ontology/isaac.py``): every field
is a typed Arrow column, the class auto-registers in the SDK's ontology registry, and
it gets a ``.Q`` query proxy so the server can filter on physical values, e.g.::

    QueryOntologyCatalog(WeldArc.Q.current_avg_a.gt(240.0))

The ``*_avg_*`` fields are 0.5 s moving averages, the value a weld data monitor
displays and the one a WPS compliance check uses; raw 100 Hz samples include
short-circuit spikes (FCAW) that are physically real but not WPS excursions.
"""
from __future__ import annotations

from typing import Optional

from mosaicolabs import MosaicoField, MosaicoType, Serializable


def _opt(description: str):
    return MosaicoField(default=None, nullable=True, description=description)


class WeldArc(Serializable):
    """Arc electrical + consumables state @100 Hz (topic /weld/arc).

    Single-wire fields describe the (lead) electrode. Tandem processes (AIMEN 2-wire SAW, BAM
    tandem GMAW) add the trail electrode in the optional ``*2*`` fields. ``gas_lpm`` is NaN for
    SAW (flux-shielded, no gas).
    """

    current_a: MosaicoType.float64 = MosaicoField(description="Welding current, raw sample [A]")
    voltage_v: MosaicoType.float64 = MosaicoField(description="Arc voltage, raw sample [V]")
    wfs_mpm: MosaicoType.float64 = MosaicoField(description="Wire feed speed [m/min]")
    gas_lpm: MosaicoType.float64 = MosaicoField(description="Shielding gas flow [L/min]")
    current_avg_a: MosaicoType.float64 = MosaicoField(description="Current, 0.5 s moving average [A]")
    voltage_avg_v: MosaicoType.float64 = MosaicoField(description="Voltage, 0.5 s moving average [V]")
    wfs_avg_mpm: MosaicoType.float64 = MosaicoField(description="Wire feed, 0.5 s moving average [m/min]")
    gas_avg_lpm: MosaicoType.float64 = MosaicoField(description="Gas flow, 0.5 s moving average [L/min]")
    current2_a: Optional[MosaicoType.float64] = _opt("Trail electrode current, raw sample [A] (tandem only)")
    voltage2_v: Optional[MosaicoType.float64] = _opt("Trail electrode voltage, raw sample [V] (tandem only)")
    wfs2_mpm: Optional[MosaicoType.float64] = _opt("Trail electrode wire feed speed [m/min] (tandem only)")
    current2_avg_a: Optional[MosaicoType.float64] = _opt("Trail current, 0.5 s moving average [A]")
    current_rms_a: Optional[MosaicoType.float64] = _opt("Lead current, 10 ms RMS [A] (AC / pulsed processes)")
    current2_rms_a: Optional[MosaicoType.float64] = _opt("Trail current, 10 ms RMS [A] (AC trail electrode, AIMEN)")
    voltage2_rms_v: Optional[MosaicoType.float64] = _opt("Trail voltage, 10 ms RMS [V]")
    voltage2_avg_v: Optional[MosaicoType.float64] = _opt("Trail voltage, 0.5 s moving average [V]")
    wfs2_avg_mpm: Optional[MosaicoType.float64] = _opt("Trail wire feed, 0.5 s moving average [m/min]")


class WeldTravel(Serializable):
    """Travel / positioner / heat input @100 Hz (topic /weld/travel)."""

    travel_speed_mmps: MosaicoType.float64 = MosaicoField(description="Torch travel speed over the work [mm/s]")
    heat_input_kj_mm: MosaicoType.float64 = MosaicoField(description="Arc heat input eta*V*I/(1000*v) [kJ/mm]")
    positioner_deg: Optional[MosaicoType.float64] = _opt("Positioner rotation since arc start [deg] (simulator)")
    position_mm: Optional[MosaicoType.float64] = _opt("Torch position along the joint [mm] (real data)")


class WeldThermal(Serializable):
    """Interpass temperature (topic /weld/thermal).

    Simulator: IR pyrometer @10 Hz. AIMEN: max(pyrometer_L, pyrometer_R) as ``interpass_c`` plus the raw
    left/right/back pyrometers; BAM: thermocouple 1 as ``interpass_c`` plus thermocouple 2.
    """

    interpass_c: MosaicoType.float64 = MosaicoField(description="Interpass temperature, raw [degC]")
    interpass_avg_c: MosaicoType.float64 = MosaicoField(description="Interpass temperature, 1 s median [degC]")
    temp_left_c: Optional[MosaicoType.float64] = _opt("Left pyrometer [degC] (AIMEN)")
    temp_right_c: Optional[MosaicoType.float64] = _opt("Right pyrometer [degC] (AIMEN)")
    temp_back_c: Optional[MosaicoType.float64] = _opt("Back pyrometer, reads the pool region [degC] (AIMEN)")
    thermo2_c: Optional[MosaicoType.float64] = _opt("Second thermocouple [degC] (BAM)")


# parameter name -> (topic, ontology class, server-side field used by find_anomalies)
PARAM_INDEX = {
    "current_a": ("/weld/arc", WeldArc, "current_avg_a"),
    "voltage_v": ("/weld/arc", WeldArc, "voltage_avg_v"),
    "wfs_mpm": ("/weld/arc", WeldArc, "wfs_avg_mpm"),
    "gas_lpm": ("/weld/arc", WeldArc, "gas_avg_lpm"),
    "current2_a": ("/weld/arc", WeldArc, "current2_avg_a"),
    "voltage2_v": ("/weld/arc", WeldArc, "voltage2_avg_v"),
    "wfs2_mpm": ("/weld/arc", WeldArc, "wfs2_avg_mpm"),
    "current2_rms_a": ("/weld/arc", WeldArc, "current2_rms_a"),
    "travel_speed_mmps": ("/weld/travel", WeldTravel, "travel_speed_mmps"),
    "heat_input_kj_mm": ("/weld/travel", WeldTravel, "heat_input_kj_mm"),
    "positioner_deg": ("/weld/travel", WeldTravel, "positioner_deg"),
    "position_mm": ("/weld/travel", WeldTravel, "position_mm"),
    "interpass_c": ("/weld/thermal", WeldThermal, "interpass_avg_c"),
}
