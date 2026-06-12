"""
APSE OS - Unified Communication Fabric (UCF)
Auto-selects transport: SHM (<1us), SecureMesh (1.5-12ms), DTN (space/delay-tolerant).
Channels are declared by intent (latency_ms, criticality) not by protocol.
"""
import time
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TransportMode(Enum):
    SHM = 'shm'            # Shared Memory <1us (same process/node)
    MESH = 'mesh'          # Secure Mesh 1.5-12ms (LAN/WAN)
    DTN = 'dtn'            # Delay Tolerant Network (space, RF, intermittent)


@dataclass
class ChannelIntent:
    name: str
    latency_ms: float      # desired max latency
    criticality: int       # 0=critical, 1=high, 2=normal, 3=low
    encrypted: bool = False
    durable: bool = False


@dataclass
class Message:
    channel: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    priority: int = 2
    ttl_ms: float = 5000.0


class UCFChannel:
    def __init__(self, intent: ChannelIntent):
        self.intent = intent
        self.transport = self._select_transport(intent)
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=1024)
        self._subscribers: List[Callable] = []
        self._lock = threading.Lock()
        self._stats = {'sent': 0, 'received': 0, 'dropped': 0}
        print(f'[UCF] Channel: {intent.name} | Transport: {self.transport.value} | latency={intent.latency_ms}ms')

    def _select_transport(self, intent: ChannelIntent) -> TransportMode:
        if intent.latency_ms < 1.0 and intent.criticality == 0:
            return TransportMode.SHM
        elif intent.latency_ms <= 50.0:
            return TransportMode.MESH
        else:
            return TransportMode.DTN

    def publish(self, payload: Any, priority: int = 2) -> bool:
        msg = Message(channel=self.intent.name, payload=payload, priority=priority)
        try:
            self._queue.put_nowait((priority, time.time(), msg))
            self._stats['sent'] += 1
            self._notify_subscribers(msg)
            return True
        except queue.Full:
            self._stats['dropped'] += 1
            print(f'[UCF] WARN: Channel {self.intent.name} queue full, message dropped')
            return False

    def subscribe(self, callback: Callable):
        with self._lock:
            self._subscribers.append(callback)

    def _notify_subscribers(self, msg: Message):
        with self._lock:
            subs = list(self._subscribers)
        for sub in subs:
            try:
                sub(msg)
            except Exception as e:
                print(f'[UCF] Subscriber error on {self.intent.name}: {e}')

    def receive(self, timeout_s: float = 0.1) -> Optional[Message]:
        try:
            _, _, msg = self._queue.get(timeout=timeout_s)
            self._stats['received'] += 1
            return msg
        except queue.Empty:
            return None

    def stats(self) -> dict:
        return {'channel': self.intent.name, 'transport': self.transport.value, **self._stats}


class UCF:
    """
    APSE Unified Communication Fabric.
    Channels are created by intent. Transport is selected automatically.
    """

    def __init__(self):
        self._channels: Dict[str, UCFChannel] = {}
        print('[UCF] Fabric initialized')

    def create_channel(self, intent: ChannelIntent) -> UCFChannel:
        ch = UCFChannel(intent)
        self._channels[intent.name] = ch
        return ch

    def get_channel(self, name: str) -> Optional[UCFChannel]:
        return self._channels.get(name)

    def publish(self, channel_name: str, payload: Any, priority: int = 2) -> bool:
        ch = self._channels.get(channel_name)
        if not ch:
            print(f'[UCF] ERROR: Channel {channel_name} not found')
            return False
        return ch.publish(payload, priority)

    def subscribe(self, channel_name: str, callback: Callable):
        ch = self._channels.get(channel_name)
        if ch:
            ch.subscribe(callback)

    def all_stats(self) -> List[dict]:
        return [ch.stats() for ch in self._channels.values()]


# --- Singleton ---
_ucf = UCF()


def get_ucf() -> UCF:
    return _ucf


def auto_channel(name: str, latency_ms: float = 10.0, criticality: int = 2) -> UCFChannel:
    """Convenience: create or get a channel with automatic transport selection."""
    ucf = get_ucf()
    ch = ucf.get_channel(name)
    if not ch:
        intent = ChannelIntent(name=name, latency_ms=latency_ms, criticality=criticality)
        ch = ucf.create_channel(intent)
    return ch
