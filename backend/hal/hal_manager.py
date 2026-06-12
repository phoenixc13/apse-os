"""
APSE OS - Hardware Abstraction Layer (HAL) Manager
Abstracts sensors, actuators, GPUs, NPUs, CAN buses, power and telemetry.
Provides unified device registry and capability detection.
"""
from __future__ import annotations
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DeviceType(Enum):
    SENSOR = 'sensor'
    ACTUATOR = 'actuator'
    GPU = 'gpu'
    NPU = 'npu'
    TPU = 'tpu'
    FPGA = 'fpga'
    CPU = 'cpu'
    CAMERA = 'camera'
    LIDAR = 'lidar'
    IMU = 'imu'
    CAN_BUS = 'can_bus'
    ETHERNET = 'ethernet'
    POWER = 'power'
    RF = 'rf'


class DeviceStatus(Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'
    DEGRADED = 'degraded'
    UNKNOWN = 'unknown'


@dataclass
class DeviceDescriptor:
    id: str
    name: str
    type: DeviceType
    vendor: str = 'unknown'
    firmware: str = '0.0.0'
    capabilities: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    status: DeviceStatus = DeviceStatus.UNKNOWN
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'vendor': self.vendor,
            'firmware': self.firmware,
            'capabilities': self.capabilities,
            'status': self.status.value,
            'last_seen': self.last_seen
        }


class HALManager:
    """
    APSE Hardware Abstraction Layer.
    Registers, monitors and provides access to all hardware devices.
    Supports hot-plug detection and capability queries.
    """

    def __init__(self):
        self._devices: Dict[str, DeviceDescriptor] = {}
        self._profile = self._detect_profile()
        print(f'[HAL] Manager initialized | Profile: {self._profile}')
        self._auto_discover()

    def _detect_profile(self) -> str:
        machine = platform.machine().lower()
        if 'aarch64' in machine or 'arm' in machine:
            return 'embedded'
        return 'industrial'

    def _auto_discover(self):
        """Auto-discover basic CPU info as a device."""
        cpu = DeviceDescriptor(
            id='cpu.0',
            name=f'CPU ({platform.processor() or platform.machine()})',
            type=DeviceType.CPU,
            vendor=platform.system(),
            capabilities=['inference', 'scheduling', 'ipc'],
            status=DeviceStatus.ONLINE
        )
        self.register_device(cpu)

    def register_device(self, device: DeviceDescriptor):
        self._devices[device.id] = device
        print(f'[HAL] Device registered: {device.id} ({device.type.value}) status={device.status.value}')

    def unregister_device(self, device_id: str):
        if device_id in self._devices:
            del self._devices[device_id]
            print(f'[HAL] Device unregistered: {device_id}')

    def get_device(self, device_id: str) -> Optional[DeviceDescriptor]:
        return self._devices.get(device_id)

    def list_devices(self, device_type: Optional[DeviceType] = None) -> List[DeviceDescriptor]:
        if device_type:
            return [d for d in self._devices.values() if d.type == device_type]
        return list(self._devices.values())

    def get_accelerators(self) -> List[DeviceDescriptor]:
        accel_types = {DeviceType.GPU, DeviceType.NPU, DeviceType.TPU, DeviceType.FPGA}
        return [d for d in self._devices.values() if d.type in accel_types and d.status == DeviceStatus.ONLINE]

    def set_status(self, device_id: str, status: DeviceStatus):
        if device_id in self._devices:
            self._devices[device_id].status = status
            self._devices[device_id].last_seen = time.time()

    def get_profile(self) -> str:
        return self._profile

    def summary(self) -> dict:
        return {
            'profile': self._profile,
            'total_devices': len(self._devices),
            'online': sum(1 for d in self._devices.values() if d.status == DeviceStatus.ONLINE),
            'devices': [d.to_dict() for d in self._devices.values()]
        }


# --- Singleton ---
_hal = HALManager()


def get_hal() -> HALManager:
    return _hal
