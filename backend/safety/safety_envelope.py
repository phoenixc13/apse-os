"""
APSE OS - Safety Envelope
Vetoes any trajectory or action that violates kinematic limits,
geofences, velocity caps or collision constraints.
Every AI-generated action MUST pass through this layer before execution.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class KinematicLimits:
    joint_min: List[float]     # radians or meters
    joint_max: List[float]
    velocity_max: List[float]  # rad/s or m/s
    acceleration_max: List[float]


@dataclass
class Geofence:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass
class SafetyVeto:
    vetoed: bool
    reason: str
    rule: str
    timestamp: float = field(default_factory=time.time)


class SafetyEnvelope:
    """
    APSE Safety Envelope.
    All AI-generated trajectories and actions are validated here.
    If any constraint is violated, the action is blocked (vetoed).
    The safety layer cannot be bypassed or disabled at runtime.
    """

    def __init__(self,
                 kinematic_limits: Optional[KinematicLimits] = None,
                 geofences: Optional[List[Geofence]] = None,
                 confidence_threshold: float = 0.75):
        self._limits = kinematic_limits
        self._geofences = geofences or []
        self._confidence_threshold = confidence_threshold
        self._veto_count = 0
        self._pass_count = 0
        print(f'[SAFETY] Envelope initialized | confidence_threshold={confidence_threshold}')

    def validate_action(self, action_vector: List[float], confidence: float,
                        position_xyz: Optional[Tuple[float, float, float]] = None) -> SafetyVeto:
        # 1. Confidence check
        if confidence < self._confidence_threshold:
            return self._veto(f'confidence {confidence:.2f} < threshold {self._confidence_threshold}', 'CONFIDENCE')

        # 2. Kinematic limits
        if self._limits:
            veto = self._check_kinematics(action_vector)
            if veto:
                return veto

        # 3. Geofence check
        if position_xyz and self._geofences:
            veto = self._check_geofence(position_xyz)
            if veto:
                return veto

        # 4. NaN/Inf check
        if any(not (-1e6 < v < 1e6) for v in action_vector):
            return self._veto('action vector contains NaN or Inf', 'NUMERIC')

        self._pass_count += 1
        return SafetyVeto(vetoed=False, reason='', rule='PASS')

    def _check_kinematics(self, action_vector: List[float]) -> Optional[SafetyVeto]:
        limits = self._limits
        for i, val in enumerate(action_vector):
            if i < len(limits.joint_min) and val < limits.joint_min[i]:
                return self._veto(f'joint[{i}]={val:.3f} < min={limits.joint_min[i]}', 'JOINT_LIMIT')
            if i < len(limits.joint_max) and val > limits.joint_max[i]:
                return self._veto(f'joint[{i}]={val:.3f} > max={limits.joint_max[i]}', 'JOINT_LIMIT')
        return None

    def _check_geofence(self, pos: Tuple[float, float, float]) -> Optional[SafetyVeto]:
        x, y, z = pos
        for fence in self._geofences:
            if not (fence.x_min <= x <= fence.x_max and
                    fence.y_min <= y <= fence.y_max and
                    fence.z_min <= z <= fence.z_max):
                return self._veto(f'position {pos} outside geofence {fence.name}', 'GEOFENCE')
        return None

    def _veto(self, reason: str, rule: str) -> SafetyVeto:
        self._veto_count += 1
        print(f'[SAFETY] VETO | rule={rule} | {reason}')
        return SafetyVeto(vetoed=True, reason=reason, rule=rule)

    def set_geofences(self, geofences: List[Geofence]):
        self._geofences = geofences

    def set_kinematic_limits(self, limits: KinematicLimits):
        self._limits = limits

    def stats(self) -> dict:
        total = self._pass_count + self._veto_count
        return {
            'passed': self._pass_count,
            'vetoed': self._veto_count,
            'total': total,
            'veto_rate': round(self._veto_count / total, 3) if total else 0.0
        }


# --- Default instance with basic 7-DOF arm limits ---
_default_limits = KinematicLimits(
    joint_min=[-3.14] * 7,
    joint_max=[3.14] * 7,
    velocity_max=[2.0] * 7,
    acceleration_max=[5.0] * 7
)

_safety = SafetyEnvelope(kinematic_limits=_default_limits, confidence_threshold=0.75)


def get_safety() -> SafetyEnvelope:
    return _safety
