"""
APSE OS - ROS/ROS 2 Compatibility Bridge
Allows existing ROS nodes to run inside APSE without full rewrite.
The shim intercepts DDS/QoS calls and republishes them on UCF.
Target overhead: <3% vs native ROS 2.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ROSMessage:
    """Minimal ROS-compatible message container."""
    topic: str
    msg_type: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    qos_reliability: str = 'reliable'
    qos_durability: str = 'volatile'


class APSECompatibilityContainer:
    """
    Container for a ROS node running inside APSE.
    Intercepts pub/sub calls and translates them to UCF channels.
    The node does NOT need to know it's running under APSE.
    """

    def __init__(self, node_name: str):
        self.node_name = node_name
        self._publishers: Dict[str, dict] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_count = 0
        self._overhead_samples: List[float] = []
        print(f'[ROS-BRIDGE] Container created for node: {node_name}')

    def create_publisher(self, topic: str, msg_type: str,
                         qos_reliability: str = 'reliable') -> 'ROSPublisherShim':
        pub = ROSPublisherShim(topic=topic, msg_type=msg_type,
                               container=self, qos_reliability=qos_reliability)
        self._publishers[topic] = {'pub': pub, 'msg_type': msg_type, 'count': 0}
        print(f'[ROS-BRIDGE] Publisher: {topic} ({msg_type})')
        return pub

    def create_subscription(self, topic: str, msg_type: str,
                            callback: Callable) -> 'ROSSubscriptionShim':
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        sub = ROSSubscriptionShim(topic=topic, msg_type=msg_type, callback=callback)
        print(f'[ROS-BRIDGE] Subscription: {topic} ({msg_type})')
        return sub

    def _publish_to_ucf(self, msg: ROSMessage):
        t0 = time.perf_counter()
        # Translate DDS QoS to UCF intent
        latency_ms = 1.5 if msg.qos_reliability == 'reliable' else 5.0
        # In production: route to UCF.publish(msg.topic, msg.data)
        self._message_count += 1
        overhead_us = (time.perf_counter() - t0) * 1e6
        self._overhead_samples.append(overhead_us)
        if len(self._overhead_samples) > 100:
            self._overhead_samples.pop(0)
        # Notify subscribers
        for cb in self._subscribers.get(msg.topic, []):
            try:
                cb(msg)
            except Exception as e:
                print(f'[ROS-BRIDGE] Subscriber error on {msg.topic}: {e}')

    def overhead_stats(self) -> dict:
        if not self._overhead_samples:
            return {'avg_us': 0.0, 'max_us': 0.0, 'messages': 0}
        avg = sum(self._overhead_samples) / len(self._overhead_samples)
        return {
            'avg_overhead_us': round(avg, 2),
            'max_overhead_us': round(max(self._overhead_samples), 2),
            'messages_bridged': self._message_count
        }


class ROSPublisherShim:
    """Mimics rclpy Publisher API while routing to UCF."""

    def __init__(self, topic: str, msg_type: str,
                 container: APSECompatibilityContainer, qos_reliability: str = 'reliable'):
        self.topic = topic
        self.msg_type = msg_type
        self._container = container
        self._qos = qos_reliability

    def publish(self, data: Any):
        msg = ROSMessage(
            topic=self.topic,
            msg_type=self.msg_type,
            data=data,
            qos_reliability=self._qos
        )
        self._container._publish_to_ucf(msg)


class ROSSubscriptionShim:
    """Mimics rclpy Subscription."""

    def __init__(self, topic: str, msg_type: str, callback: Callable):
        self.topic = topic
        self.msg_type = msg_type
        self.callback = callback


class ROSBridgeManager:
    """Manages all APSE-ROS compatibility containers."""

    def __init__(self):
        self._containers: Dict[str, APSECompatibilityContainer] = {}
        print('[ROS-BRIDGE] Manager ready | overhead target: <3%')

    def wrap_node(self, node_name: str) -> APSECompatibilityContainer:
        container = APSECompatibilityContainer(node_name)
        self._containers[node_name] = container
        return container

    def get_container(self, node_name: str) -> Optional[APSECompatibilityContainer]:
        return self._containers.get(node_name)

    def all_stats(self) -> List[dict]:
        return [
            {'node': name, **c.overhead_stats()}
            for name, c in self._containers.items()
        ]


# --- Singleton ---
_bridge = ROSBridgeManager()


def get_bridge() -> ROSBridgeManager:
    return _bridge


def wrap_ros_node(node_name: str) -> APSECompatibilityContainer:
    """Convenience: wrap a ROS node for APSE compatibility."""
    return get_bridge().wrap_node(node_name)
