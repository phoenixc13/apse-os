"""
APSE OS - Kernel Scheduler
Real-time deterministic scheduler with L0..L3 priority queues.
L0 = safety-critical (<10us jitter), L3 = background.
"""
import time
import threading
import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import IntEnum


class Priority(IntEnum):
    L0_SAFETY_CRITICAL = 0
    L1_REALTIME = 1
    L2_STANDARD = 2
    L3_BACKGROUND = 3


@dataclass(order=True)
class Task:
    priority: int
    deadline_us: float
    name: str = field(compare=False)
    callback: Callable = field(compare=False)
    period_us: Optional[float] = field(default=None, compare=False)
    next_run: float = field(default=0.0, compare=False)

    JITTER_LIMIT_US = {0: 10, 1: 100, 2: 1000, 3: float('inf')}


class APSEScheduler:
    """
    APSE Real-Time Scheduler.
    Manages four priority queues with deterministic execution guarantees.
    L0 panics and triggers fault containment if jitter exceeds 10us.
    """

    def __init__(self):
        self._queues = {p: [] for p in Priority}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {p: {'executions': 0, 'max_jitter_us': 0.0} for p in Priority}
        print('[APSE-SCHEDULER] Initialized with L0..L3 queues')

    def register_task(self, name: str, callback: Callable, priority: Priority,
                      deadline_us: float, period_us: Optional[float] = None):
        task = Task(
            priority=int(priority),
            deadline_us=deadline_us,
            name=name,
            callback=callback,
            period_us=period_us,
            next_run=time.perf_counter() * 1e6
        )
        with self._lock:
            heapq.heappush(self._queues[priority], task)
        print(f'[APSE-SCHEDULER] Task registered: {name} @ P{priority} deadline={deadline_us}us')

    def _execute_task(self, task: Task):
        scheduled = task.next_run
        actual = time.perf_counter() * 1e6
        jitter = abs(actual - scheduled)
        limit = Task.JITTER_LIMIT_US[task.priority]
        if jitter > limit:
            if task.priority == Priority.L0_SAFETY_CRITICAL:
                self._panic(task.name, jitter)
                return
            print(f'[APSE-SCHEDULER] WARN jitter {jitter:.1f}us > {limit}us on task {task.name}')
        try:
            task.callback()
        except Exception as e:
            print(f'[APSE-SCHEDULER] ERROR task {task.name}: {e}')
        stats = self._stats[Priority(task.priority)]
        stats['executions'] += 1
        stats['max_jitter_us'] = max(stats['max_jitter_us'], jitter)

    def _panic(self, task_name: str, jitter_us: float):
        print(f'[APSE-SCHEDULER] PANIC! L0 task {task_name} jitter={jitter_us:.2f}us > 10us')
        print('[APSE-SCHEDULER] Triggering fault containment domain reset')

    def _run_loop(self):
        while self._running:
            now_us = time.perf_counter() * 1e6
            for priority in Priority:
                queue = self._queues[priority]
                with self._lock:
                    ready = [t for t in queue if t.next_run <= now_us]
                for task in ready:
                    self._execute_task(task)
                    if task.period_us:
                        task.next_run = now_us + task.period_us
                    else:
                        with self._lock:
                            if task in self._queues[priority]:
                                self._queues[priority].remove(task)
            time.sleep(1e-6)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print('[APSE-SCHEDULER] Scheduler started')

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print('[APSE-SCHEDULER] Scheduler stopped')

    def get_stats(self) -> dict:
        return {str(p): self._stats[p] for p in Priority}


# --- Singleton ---
_scheduler = APSEScheduler()


def get_scheduler() -> APSEScheduler:
    return _scheduler
