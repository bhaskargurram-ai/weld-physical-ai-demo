"""Shared data contract for the WeldWorld demo.

Every component (simulator, Mosaico ingest, curation, world models, copilot)
imports from here so names, units and rates stay consistent.

Scenario: a robotic pipe-spool / pressure-vessel weld cell in a fabrication
shop (ASME Sec. IX / B31.3 style work: carbon-steel pipe, GTAW root,
FCAW/SMAW fill & cap, pipe on a positioner, torch on a 6-axis robot).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------- rates (Hz)
ARC_HZ = 100          # arc current / voltage / wire feed / gas flow
ROBOT_HZ = 50         # joint states + torch TCP pose
THERMAL_HZ = 10       # IR pyrometer interpass temperature
CAMERA_FPS = 24       # torch-mounted weld camera
CLIP_SECONDS = 5      # curation clip length handed to Cosmos

# ------------------------------------------------------- Mosaico topic names
TOPIC_ARC = "/weld/arc"                 # current_a, voltage_v, wfs_mpm, gas_lpm
TOPIC_ROBOT_JOINTS = "/robot/joint_states"
TOPIC_TCP_POSE = "/robot/tcp_pose"      # torch tip pose
TOPIC_TRAVEL = "/weld/travel"           # travel_speed_mmps, positioner_deg
TOPIC_THERMAL = "/weld/thermal"         # interpass_c
TOPIC_CAMERA = "/camera/torch/image"    # compressed JPEG frames
TOPIC_EVENTS = "/weld/events"           # arc_start, arc_stop, defect onset/offset


class Process(str, Enum):
    GTAW = "GTAW"   # TIG - root pass
    GMAW = "GMAW"   # MIG
    FCAW = "FCAW"   # flux-core - fill / cap
    SMAW = "SMAW"   # stick
    SAW = "SAW"     # submerged arc - vessel long seams


class Position(str, Enum):
    P1G = "1G"  # pipe rolled, flat
    P2G = "2G"
    P5G = "5G"  # pipe fixed horizontal
    P6G = "6G"  # pipe fixed 45 deg


class Defect(str, Enum):
    NONE = "none"
    POROSITY = "porosity"                 # shielding gas loss / contamination
    LACK_OF_FUSION = "lack_of_fusion"     # low heat input / fast travel
    UNDERCUT = "undercut"                 # high voltage / fast travel
    BURN_THROUGH = "burn_through"         # excessive heat on root
    ARC_INSTABILITY = "arc_instability"   # stick-out / wire feed issues
    CRACK = "crack"                       # high interpass temp / hydrogen


# Signature each defect leaves in the sensor streams; the simulator uses it to
# inject anomalies and the copilot uses it to explain root cause.
DEFECT_SIGNATURES: dict[Defect, dict] = {
    Defect.POROSITY: {"gas_lpm": "drops to <8 L/min", "voltage_v": "spiky"},
    Defect.LACK_OF_FUSION: {"current_a": "-20..-30%", "travel_speed_mmps": "+40%"},
    Defect.UNDERCUT: {"voltage_v": "+15..25%", "travel_speed_mmps": "+20%"},
    Defect.BURN_THROUGH: {"current_a": "+30%", "travel_speed_mmps": "-40%"},
    Defect.ARC_INSTABILITY: {"voltage_v": "high-freq noise", "wfs_mpm": "oscillating"},
    Defect.CRACK: {"interpass_c": "> WPS max interpass"},
}


@dataclass
class WPS:
    """Welding Procedure Specification limits (what a CWI checks against)."""
    wps_id: str
    process: Process
    current_a: tuple[float, float]
    voltage_v: tuple[float, float]
    travel_speed_mmps: tuple[float, float]
    wfs_mpm: tuple[float, float]
    gas_lpm: tuple[float, float]
    max_interpass_c: float
    heat_input_kj_mm: tuple[float, float]


WPS_LIBRARY: dict[str, WPS] = {
    "WPS-GTAW-ROOT-01": WPS("WPS-GTAW-ROOT-01", Process.GTAW, (90, 130), (10, 14),
                            (1.0, 2.0), (0.0, 1.5), (12, 18), 250, (0.6, 1.4)),
    "WPS-FCAW-FILL-02": WPS("WPS-FCAW-FILL-02", Process.FCAW, (180, 240), (23, 27),
                            (4.0, 7.0), (7.0, 10.0), (18, 25), 250, (0.7, 1.5)),
    "WPS-SAW-SEAM-03": WPS("WPS-SAW-SEAM-03", Process.SAW, (450, 650), (28, 34),
                           (8.0, 12.0), (1.5, 3.0), (0.0, 0.1), 260, (1.2, 2.6)),
}


def heat_input_kj_mm(current_a: float, voltage_v: float, travel_mmps: float,
                     efficiency: float = 0.8) -> float:
    """ASME/AWS arc heat input: (eta * V * I) / (1000 * travel speed[mm/s])."""
    if travel_mmps <= 0:
        return float("inf")
    return efficiency * voltage_v * current_a / (1000.0 * travel_mmps)


@dataclass
class DefectEvent:
    defect: Defect
    t_start: float   # seconds from arc start
    t_end: float
    severity: float  # 0..1


@dataclass
class WeldPass:
    """One welding pass = one Mosaico sequence."""
    sequence_name: str          # e.g. "SP-1043_J07_P2_FCAW"
    spool_id: str               # pipe spool / vessel ID
    joint_id: str
    pass_no: int
    pass_type: str              # root | hot | fill | cap
    wps_id: str
    process: Process
    position: Position
    material: str               # e.g. "A106 Gr.B"
    pipe_od_mm: float
    wall_mm: float
    cell_id: str
    duration_s: float
    defects: list[DefectEvent] = field(default_factory=list)
    rt_result: Optional[str] = None   # "accept" | "reject" (radiography)

    def to_metadata(self) -> dict:
        d = asdict(self)
        d["process"] = self.process.value
        d["position"] = self.position.value
        d["defects"] = [
            {**asdict(e), "defect": e.defect.value} for e in self.defects
        ]
        return d
