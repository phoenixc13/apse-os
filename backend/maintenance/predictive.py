"""
APSE OS - Predictive Maintenance Monitor
Ingests hardware telemetry and computes Remaining Useful Life (RUL).
Components below RUL threshold trigger alerts and maintenance requests.
"""
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TelemetrySample:
    component_id: str
    temperature_c: float
    load_percent: float
    vibration_g: float
    ecc_errors: int = 0
    jitter_us: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComponentHealth:
    component_id: str
    rul_normalized: float   # 0.0 = end of life, 1.0 = brand new
    risk_level: str         # 'ok', 'warn', 'critical'
    last_updated: float = field(default_factory=time.time)
    anomalies: List[str] = field(default_factory=list)


class PredictiveMaintenance:
    """
    APSE Predictive Maintenance Layer.
    Monitors telemetry and predicts component degradation.
    RUL < 0.30 = warning, RUL < 0.10 = critical.
    In production: replace RUL model with trained ONNX degradation model.
    """

    RUL_WARN = 0.30
    RUL_CRITICAL = 0.10

    TEMP_LIMITS = {'cpu': 85.0, 'gpu': 90.0, 'npu': 80.0, 'motor': 70.0, 'default': 75.0}
    VIBRATION_LIMIT = 2.5  # g

    def __init__(self):
        self._health: Dict[str, ComponentHealth] = {}
        self._samples: Dict[str, List[TelemetrySample]] = {}
        self._alert_callbacks = []
        print('[MAINTENANCE] Predictive maintenance monitor initialized')

    def ingest(self, sample: TelemetrySample):
        cid = sample.component_id
        if cid not in self._samples:
            self._samples[cid] = []
        self._samples[cid].append(sample)
        # Keep last 100 samples
        if len(self._samples[cid]) > 100:
            self._samples[cid].pop(0)
        self._update_health(cid)

    def _update_health(self, cid: str):
        samples = self._samples.get(cid, [])
        if not samples:
            return
        last = samples[-1]
        anomalies = []

        # Temperature check
        comp_type = cid.split('.')[0] if '.' in cid else 'default'
        temp_limit = self.TEMP_LIMITS.get(comp_type, self.TEMP_LIMITS['default'])
        if last.temperature_c > temp_limit:
            anomalies.append(f'HIGH_TEMP:{last.temperature_c:.1f}C>{temp_limit}C')

        # Vibration check
        if last.vibration_g > self.VIBRATION_LIMIT:
            anomalies.append(f'HIGH_VIBRATION:{last.vibration_g:.2f}g')

        # ECC errors
        if last.ecc_errors > 0:
            anomalies.append(f'ECC_ERRORS:{last.ecc_errors}')

        # Compute RUL (simplified degradation model)
        rul = self._compute_rul(samples)

        risk = 'ok'
        if rul < self.RUL_WARN:
            risk = 'warn'
        if rul < self.RUL_CRITICAL:
            risk = 'critical'

        health = ComponentHealth(component_id=cid, rul_normalized=rul,
                                 risk_level=risk, anomalies=anomalies)
        self._health[cid] = health

        if risk in ('warn', 'critical'):
            print(f'[MAINTENANCE] {risk.upper()} {cid} RUL={rul:.2f} anomalies={anomalies}')
            for cb in self._alert_callbacks:
                try:
                    cb(health)
                except Exception:
                    pass

    def _compute_rul(self, samples: List[TelemetrySample]) -> float:
        """
        Simplified RUL model based on recent telemetry trend.
        In production: replace with trained ONNX RUL regression model.
        """
        if len(samples) < 2:
            return 1.0
        # Average load and temperature trend
        avg_load = sum(s.load_percent for s in samples[-10:]) / min(len(samples), 10)
        avg_temp = sum(s.temperature_c for s in samples[-10:]) / min(len(samples), 10)
        max_temp = self.TEMP_LIMITS.get('default', 75.0)
        rul = max(0.0, 1.0 - (avg_load / 100.0) * 0.4 - (avg_temp / max_temp) * 0.6)
        return round(rul, 3)

    def on_alert(self, callback):
        self._alert_callbacks.append(callback)

    def get_health(self, cid: str) -> Optional[ComponentHealth]:
        return self._health.get(cid)

    def all_health(self) -> List[dict]:
        return [
            {'id': h.component_id, 'rul': h.rul_normalized,
             'risk': h.risk_level, 'anomalies': h.anomalies}
            for h in self._health.values()
        ]

    def simulate_telemetry(self, component_id: str):
        """Generates a synthetic telemetry sample for testing."""
        sample = TelemetrySample(
            component_id=component_id,
            temperature_c=random.uniform(45.0, 82.0),
            load_percent=random.uniform(20.0, 95.0),
            vibration_g=random.uniform(0.1, 2.8),
            ecc_errors=random.randint(0, 3),
            jitter_us=random.uniform(0.5, 15.0)
        )
        self.ingest(sample)
        return sample


# --- Singleton ---
_maintenance = PredictiveMaintenance()


def get_maintenance() -> PredictiveMaintenance:
    return _maintenance
