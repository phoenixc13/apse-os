"""
APSE OS - Fault Containment Sandbox (FCD)
Each driver/sensor runs in an isolated FaultDomain.
Failures are contained without crashing the full system.
"""
import threading
import time
from enum import Enum
from typing import Callable, Optional


class DomainState(Enum):
    RUNNING = 'running'
    FAULTED = 'faulted'
    RESTARTING = 'restarting'
    STOPPED = 'stopped'


class FaultDomain:
    """
    Isolated execution domain for a single module (sensor, driver, plugin).
    On failure, the domain restarts automatically (hot-restart) without
    affecting other domains or the APSE kernel.
    """

    MAX_RESTARTS = 5
    RESTART_DELAY_S = 0.5

    def __init__(self, name: str, entrypoint: Callable, watchdog_timeout_s: float = 2.0):
        self.name = name
        self.entrypoint = entrypoint
        self.watchdog_timeout_s = watchdog_timeout_s
        self.state = DomainState.STOPPED
        self._restart_count = 0
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat = time.time()
        self._lock = threading.Lock()
        print(f'[FCD] Domain created: {self.name}')

    def start(self):
        with self._lock:
            if self.state == DomainState.RUNNING:
                return
            self.state = DomainState.RUNNING
        self._thread = threading.Thread(target=self._run, name=f'fcd-{self.name}', daemon=True)
        self._thread.start()
        print(f'[FCD] Domain started: {self.name}')

    def _run(self):
        while self.state == DomainState.RUNNING:
            try:
                self.entrypoint(self._heartbeat)
            except Exception as e:
                self._handle_fault(e)
                return

    def _heartbeat(self):
        self._last_heartbeat = time.time()

    def _handle_fault(self, error: Exception):
        with self._lock:
            self.state = DomainState.FAULTED
            self._restart_count += 1
        print(f'[FCD] FAULT in {self.name}: {error} (restart {self._restart_count}/{self.MAX_RESTARTS})')
        if self._restart_count <= self.MAX_RESTARTS:
            time.sleep(self.RESTART_DELAY_S)
            self.state = DomainState.RESTARTING
            print(f'[FCD] Hot-restart: {self.name}')
            self.start()
        else:
            self.state = DomainState.STOPPED
            print(f'[FCD] CRITICAL: {self.name} exceeded max restarts. Domain permanently stopped.')

    def stop(self):
        with self._lock:
            self.state = DomainState.STOPPED
        print(f'[FCD] Domain stopped: {self.name}')

    def is_healthy(self) -> bool:
        if self.state != DomainState.RUNNING:
            return False
        return (time.time() - self._last_heartbeat) < self.watchdog_timeout_s

    def status(self) -> dict:
        return {
            'name': self.name,
            'state': self.state.value,
            'restarts': self._restart_count,
            'healthy': self.is_healthy(),
            'last_heartbeat_s': round(time.time() - self._last_heartbeat, 3)
        }


class SandboxManager:
    """Manages all FaultDomains in the APSE OS."""

    def __init__(self):
        self._domains: dict[str, FaultDomain] = {}
        print('[SANDBOX] SandboxManager initialized')

    def create_domain(self, name: str, entrypoint: Callable,
                      watchdog_timeout_s: float = 2.0) -> FaultDomain:
        domain = FaultDomain(name, entrypoint, watchdog_timeout_s)
        self._domains[name] = domain
        return domain

    def start_all(self):
        for domain in self._domains.values():
            domain.start()

    def stop_all(self):
        for domain in self._domains.values():
            domain.stop()

    def get_status(self) -> list:
        return [d.status() for d in self._domains.values()]

    def get_domain(self, name: str) -> Optional[FaultDomain]:
        return self._domains.get(name)


# --- Singleton ---
_sandbox_manager = SandboxManager()


def get_sandbox() -> SandboxManager:
    return _sandbox_manager
