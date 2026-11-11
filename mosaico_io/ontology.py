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

from mosaicolabs import MosaicoField, MosaicoType, Serializable


class WeldArc(Serializable):
    """Arc electrical + consumables state @100 Hz (topic /weld/arc)."""

    current_a: MosaicoType.float64 = MosaicoField(description="Welding current, raw sample [A]")
    voltage_v: MosaicoType.float64 = MosaicoField(description="Arc voltage, raw sample [V]")
    wfs_mpm: MosaicoType.float64 = MosaicoField(description="Wire feed speed [m/min]")
    gas_lpm: MosaicoType.float64 = MosaicoField(description="Shielding gas flow [L/min]")
    current_avg_a: MosaicoType.float64 = MosaicoField(description="Current, 0.5 s moving average [A]")
    voltage_avg_v: MosaicoType.float64 = MosaicoField(description="Voltage, 0.5 s moving average [V]")
    wfs_avg_mpm: MosaicoType.float64 = MosaicoField(description="Wire feed, 0.5 s moving average [m/min]")
    gas_avg_lpm: MosaicoType.float64 = MosaicoField(description="Gas flow, 0.5 s moving average [L/min]")


class WeldTravel(Serializable):
    """Travel / positioner / heat input @100 Hz (topic /weld/travel)."""

    travel_speed_mmps: MosaicoType.float64 = MosaicoField(description="Torch travel speed over the work [mm/s]")
    positioner_deg: MosaicoType.float64 = MosaicoField(description="Positioner rotation since arc start [deg]")
    heat_input_kj_mm: MosaicoType.float64 = MosaicoField(description="Arc heat input eta*V*I/(1000*v) [kJ/mm]")


class WeldThermal(Serializable):
    """IR pyrometer interpass temperature @10 Hz (topic /weld/thermal)."""

    interpass_c: MosaicoType.float64 = MosaicoField(description="Interpass temperature, raw [degC]")
    interpass_avg_c: MosaicoType.float64 = MosaicoField(description="Interpass temperature, 1 s median [degC]")


# parameter name -> (topic, ontology class, server-side field used by find_anomalies)
PARAM_INDEX = {
    "current_a": ("/weld/arc", WeldArc, "current_avg_a"),
    "voltage_v": ("/weld/arc", WeldArc, "voltage_avg_v"),
    "wfs_mpm": ("/weld/arc", WeldArc, "wfs_avg_mpm"),
    "gas_lpm": ("/weld/arc", WeldArc, "gas_avg_lpm"),
    "travel_speed_mmps": ("/weld/travel", WeldTravel, "travel_speed_mmps"),
    "heat_input_kj_mm": ("/weld/travel", WeldTravel, "heat_input_kj_mm"),
    "positioner_deg": ("/weld/travel", WeldTravel, "positioner_deg"),
    "interpass_c": ("/weld/thermal", WeldThermal, "interpass_avg_c"),
}
