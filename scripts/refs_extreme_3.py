PUBSUB = {
    "pubsub_types.py": '''
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class Message:
    topic: str
    payload: Any
    id: int

@dataclass
class SubscriberInfo:
    topic_pattern: str
    handler: Callable
    id: int
''',
    "pubsub_broker.py": '''
from pubsub_types import Message, SubscriberInfo

def _matches(pattern, topic):
    pp = pattern.split(".")
    tp = topic.split(".")
    i = j = 0
    while i < len(pp) and j < len(tp):
        seg = pp[i]
        if seg == "#":
            return True if i == len(pp) - 1 else False
        if seg == "*":
            i += 1
            j += 1
            continue
        if seg != tp[j]:
            return False
        i += 1
        j += 1
    return i == len(pp) and j == len(tp)


class Broker:
    def __init__(self):
        self._next_msg_id = 1
        self._next_sub_id = 1
        self._subs = {}  # sub_id -> SubscriberInfo
        self._published_topics = set()

    def publish(self, topic, payload):
        mid = self._next_msg_id
        self._next_msg_id += 1
        self._published_topics.add(topic)
        msg = Message(topic=topic, payload=payload, id=mid)
        for sid in sorted(self._subs):
            sub = self._subs[sid]
            if _matches(sub.topic_pattern, topic):
                sub.handler(msg)
        return mid

    def subscribe(self, topic_pattern, handler):
        sid = self._next_sub_id
        self._next_sub_id += 1
        self._subs[sid] = SubscriberInfo(topic_pattern, handler, sid)
        return sid

    def unsubscribe(self, sub_id):
        if sub_id in self._subs:
            del self._subs[sub_id]
            return True
        return False

    def topics(self):
        return sorted(self._published_topics)
''',
}

VALIDATOR = {
    "validator_types.py": '''
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class ValidationError:
    field: str
    message: str
    code: str

@dataclass
class ValidationResult:
    valid: bool
    errors: list = field(default_factory=list)

class Validator(ABC):
    @abstractmethod
    def validate(self, value, field_name=""): ...
''',
    "validators.py": '''
import re as _re
from validator_types import Validator, ValidationResult, ValidationError


class Required(Validator):
    def validate(self, value, field_name=""):
        if value is None or (isinstance(value, str) and value == ""):
            return ValidationResult(False, [ValidationError(field_name, "value is required", "required")])
        return ValidationResult(True, [])


class MinLength(Validator):
    def __init__(self, n):
        self.n = n

    def validate(self, value, field_name=""):
        try:
            if len(value) < self.n:
                return ValidationResult(False, [ValidationError(field_name, f"min length {self.n}", "min_length")])
            return ValidationResult(True, [])
        except TypeError:
            return ValidationResult(False, [ValidationError(field_name, "no length", "min_length")])


class MaxLength(Validator):
    def __init__(self, n):
        self.n = n

    def validate(self, value, field_name=""):
        try:
            if len(value) > self.n:
                return ValidationResult(False, [ValidationError(field_name, f"max length {self.n}", "max_length")])
            return ValidationResult(True, [])
        except TypeError:
            return ValidationResult(False, [ValidationError(field_name, "no length", "max_length")])


class Range(Validator):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value, field_name=""):
        if value < self.min_val or value > self.max_val:
            return ValidationResult(False, [ValidationError(field_name, f"out of range", "range")])
        return ValidationResult(True, [])


class Regex(Validator):
    def __init__(self, pattern):
        self.pattern = _re.compile(pattern)

    def validate(self, value, field_name=""):
        if not isinstance(value, str) or not self.pattern.match(value):
            return ValidationResult(False, [ValidationError(field_name, "regex mismatch", "regex")])
        return ValidationResult(True, [])


class All(Validator):
    def __init__(self, *validators):
        self.validators = validators

    def validate(self, value, field_name=""):
        errors = []
        for v in self.validators:
            r = v.validate(value, field_name)
            if not r.valid:
                errors.extend(r.errors)
        return ValidationResult(len(errors) == 0, errors)


class Any(Validator):
    def __init__(self, *validators):
        self.validators = validators

    def validate(self, value, field_name=""):
        all_errors = []
        for v in self.validators:
            r = v.validate(value, field_name)
            if r.valid:
                return ValidationResult(True, [])
            all_errors.extend(r.errors)
        return ValidationResult(False, all_errors)


class Schema(Validator):
    def __init__(self, fields):
        self.fields = fields

    def validate(self, value, field_name=""):
        errors = []
        if not isinstance(value, dict):
            return ValidationResult(False, [ValidationError(field_name, "expected dict", "schema")])
        for fname, validator in self.fields.items():
            r = validator.validate(value.get(fname), fname)
            if not r.valid:
                errors.extend(r.errors)
        return ValidationResult(len(errors) == 0, errors)
''',
}

DEP_RES = {
    "dep_types.py": '''
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Node:
    name: str
    dependencies: list = field(default_factory=list)
    data: Any = None

class CycleError(Exception):
    def __init__(self, cycle):
        super().__init__(f"cycle detected: {cycle}")
        self.cycle = cycle

@dataclass
class ResolutionPlan:
    order: list
    parallel_groups: list
''',
    "dep_resolver.py": '''
from dep_types import Node, CycleError, ResolutionPlan

class Graph:
    def __init__(self):
        self._nodes = {}  # name -> Node

    def add_node(self, node):
        self._nodes[node.name] = node

    def remove_node(self, name):
        if name in self._nodes:
            del self._nodes[name]
        for n in self._nodes.values():
            n.dependencies = [d for d in n.dependencies if d != name]

    def add_edge(self, from_name, to_name):
        node = self._nodes.get(from_name)
        if node is None:
            return
        if to_name not in node.dependencies:
            node.dependencies.append(to_name)

    def _collect_subgraph(self, target):
        if target is None:
            return set(self._nodes.keys())
        result = set()
        stack = [target]
        while stack:
            n = stack.pop()
            if n in result:
                continue
            result.add(n)
            node = self._nodes.get(n)
            if node:
                for d in node.dependencies:
                    if d not in result:
                        stack.append(d)
        return result

    def resolve(self, target=None):
        nodes = self._collect_subgraph(target)
        # detect cycles via DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        path = []

        def dfs(n):
            color[n] = GRAY
            path.append(n)
            node = self._nodes.get(n)
            if node:
                for d in node.dependencies:
                    if d not in color:
                        continue
                    if color[d] == GRAY:
                        idx = path.index(d)
                        raise CycleError(path[idx:] + [d])
                    if color[d] == WHITE:
                        dfs(d)
            color[n] = BLACK
            path.pop()

        for n in nodes:
            if color[n] == WHITE:
                dfs(n)

        # Compute parallel groups via Kahn-like layering
        in_subgraph = nodes
        deps = {n: set(d for d in self._nodes[n].dependencies if d in in_subgraph) for n in in_subgraph}
        groups = []
        order = []
        remaining = set(in_subgraph)
        while remaining:
            ready = sorted(n for n in remaining if not deps[n])
            if not ready:
                raise CycleError(list(remaining))
            groups.append(ready)
            order.extend(ready)
            for n in ready:
                remaining.remove(n)
                for m in remaining:
                    deps[m].discard(n)
        return ResolutionPlan(order=order, parallel_groups=groups)
''',
}

CIRCUIT = {
    "breaker_types.py": '''
from enum import Enum
from dataclasses import dataclass

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class BreakerStats:
    successes: int
    failures: int
    state: CircuitState
    opened_at: float | None

class BreakerOpenError(Exception):
    pass
''',
    "circuit_breaker.py": '''
import time
from breaker_types import CircuitState, BreakerStats, BreakerOpenError

class CircuitBreaker:
    def __init__(self, failure_threshold, recovery_timeout, half_open_max_calls=1):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._successes = 0
        self._failures = 0
        self._opened_at = None
        self._half_open_count = 0

    def _now(self):
        return time.monotonic()

    def _maybe_transition_to_half_open(self):
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if self._now() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_count = 0

    def call(self, fn, *args, **kwargs):
        self._maybe_transition_to_half_open()
        if self._state == CircuitState.OPEN:
            raise BreakerOpenError("circuit is open")
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_count += 1
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._failures += 1
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = self._now()
            elif self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._now()
            raise
        self._successes += 1
        if self._state == CircuitState.CLOSED:
            self._consecutive_failures = 0
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
        return result

    def reset(self):
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._successes = 0
        self._failures = 0
        self._opened_at = None
        self._half_open_count = 0

    def stats(self):
        return BreakerStats(self._successes, self._failures, self._state, self._opened_at)
''',
}

RETRY = {
    "retry_types.py": '''
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

class BackoffStrategy(ABC):
    @abstractmethod
    def delay(self, attempt): ...

@dataclass
class Attempt:
    number: int
    exception: Exception | None
    delay_before: float
    result: Any = None

class RetryExhausted(Exception):
    def __init__(self, attempts):
        super().__init__(f"retry exhausted after {len(attempts)} attempts")
        self.attempts = attempts
''',
    "retry.py": '''
import time
from retry_types import BackoffStrategy, Attempt, RetryExhausted


class ConstantBackoff(BackoffStrategy):
    def __init__(self, delay_seconds):
        self.delay_seconds = delay_seconds

    def delay(self, attempt):
        return self.delay_seconds


class LinearBackoff(BackoffStrategy):
    def __init__(self, initial, increment):
        self.initial = initial
        self.increment = increment

    def delay(self, attempt):
        return self.initial + self.increment * (attempt - 1)


class ExponentialBackoff(BackoffStrategy):
    def __init__(self, initial, multiplier=2.0, max_delay=float("inf")):
        self.initial = initial
        self.multiplier = multiplier
        self.max_delay = max_delay

    def delay(self, attempt):
        d = self.initial * (self.multiplier ** (attempt - 1))
        return min(d, self.max_delay)


class RetryPolicy:
    def __init__(self, max_attempts, backoff, retry_on=(Exception,), sleep_fn=None):
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.retry_on = retry_on
        self.sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
        self._attempts = []

    def execute(self, fn, *args, **kwargs):
        self._attempts = []
        for n in range(1, self.max_attempts + 1):
            delay_before = 0.0
            if n > 1:
                delay_before = self.backoff.delay(n - 1)
                self.sleep_fn(delay_before)
            try:
                result = fn(*args, **kwargs)
                self._attempts.append(Attempt(n, None, delay_before, result))
                return result
            except Exception as e:
                if not isinstance(e, self.retry_on):
                    raise
                self._attempts.append(Attempt(n, e, delay_before, None))
        raise RetryExhausted(list(self._attempts))

    @property
    def attempts(self):
        return list(self._attempts)
''',
}

REFS = {
    "pubsub_topics": PUBSUB,
    "validator_chain": VALIDATOR,
    "dependency_resolver": DEP_RES,
    "circuit_breaker": CIRCUIT,
    "retry_policy": RETRY,
}
