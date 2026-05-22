CACHE_WITH_TTL = {
    "cache_types.py": '''
from dataclasses import dataclass
from typing import Any

@dataclass
class CacheEntry:
    value: Any
    expires_at: float
    created_at: float

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
''',
    "ttl_cache.py": '''
import time
from collections import OrderedDict
from cache_types import CacheEntry, CacheStats

class TTLCache:
    def __init__(self, capacity, default_ttl=None):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._store = OrderedDict()
        self._stats = CacheStats()

    def _now(self):
        return time.monotonic()

    def get(self, key):
        if key not in self._store:
            self._stats.misses += 1
            return None
        entry = self._store[key]
        if entry.expires_at != float("inf") and self._now() >= entry.expires_at:
            del self._store[key]
            self._stats.expirations += 1
            self._stats.misses += 1
            return None
        self._store.move_to_end(key)
        self._stats.hits += 1
        return entry.value

    def set(self, key, value, ttl=None):
        eff_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = float("inf") if eff_ttl is None else self._now() + eff_ttl
        if key in self._store:
            self._store[key] = CacheEntry(value, expires_at, self._now())
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.capacity:
            self._store.popitem(last=False)
            self._stats.evictions += 1
        self._store[key] = CacheEntry(value, expires_at, self._now())

    def delete(self, key):
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self):
        self._store.clear()
        self._stats = CacheStats()

    def stats(self):
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            expirations=self._stats.expirations,
        )
''',
}

RATE_LIMITER = {
    "limiter_types.py": '''
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class LimitResult:
    allowed: bool
    remaining: int
    retry_after: float

class RateLimiter(ABC):
    @abstractmethod
    def allow(self, key): ...
    @abstractmethod
    def reset(self, key): ...
''',
    "limiters.py": '''
import time
from collections import deque
from limiter_types import RateLimiter, LimitResult

class TokenBucketLimiter(RateLimiter):
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._state = {}

    def _now(self):
        return time.monotonic()

    def _bucket(self, key):
        if key not in self._state:
            self._state[key] = (self.capacity, self._now())
        return self._state[key]

    def allow(self, key):
        tokens, last = self._bucket(key)
        now = self._now()
        elapsed = now - last
        tokens = min(self.capacity, tokens + elapsed * self.refill_rate)
        if tokens >= 1.0:
            tokens -= 1.0
            self._state[key] = (tokens, now)
            return LimitResult(True, int(tokens), 0.0)
        retry = (1.0 - tokens) / self.refill_rate
        self._state[key] = (tokens, now)
        return LimitResult(False, 0, retry)

    def reset(self, key):
        self._state.pop(key, None)


class SlidingWindowLimiter(RateLimiter):
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window = window_seconds
        self._state = {}

    def _now(self):
        return time.monotonic()

    def allow(self, key):
        now = self._now()
        if key not in self._state:
            self._state[key] = deque()
        q = self._state[key]
        cutoff = now - self.window
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) < self.max_requests:
            q.append(now)
            return LimitResult(True, self.max_requests - len(q), 0.0)
        retry = q[0] + self.window - now
        return LimitResult(False, 0, max(retry, 0.0))

    def reset(self, key):
        self._state.pop(key, None)
''',
}

EVENT_BUS = {
    "event_types.py": '''
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Callable
import time

@dataclass
class Event:
    name: str
    payload: Any
    timestamp: float = field(default_factory=time.monotonic)

@dataclass
class Subscription:
    handler: Callable
    priority: int
    filter_fn: Callable | None

class EventBusBase(ABC):
    @abstractmethod
    def subscribe(self, event_name, handler, priority=0, filter_fn=None): ...
    @abstractmethod
    def unsubscribe(self, subscription_id): ...
    @abstractmethod
    def publish(self, event): ...
''',
    "event_bus.py": '''
from event_types import Event, Subscription, EventBusBase

class EventBus(EventBusBase):
    def __init__(self):
        self._subs = {}  # sid -> (event_name, sub, order)
        self._next_id = 1
        self._next_order = 0

    def subscribe(self, event_name, handler, priority=0, filter_fn=None):
        sid = self._next_id
        self._next_id += 1
        sub = Subscription(handler, priority, filter_fn)
        self._subs[sid] = (event_name, sub, self._next_order)
        self._next_order += 1
        return sid

    def unsubscribe(self, subscription_id):
        if subscription_id in self._subs:
            del self._subs[subscription_id]
            return True
        return False

    def publish(self, event):
        matching = []
        for sid, (name, sub, order) in self._subs.items():
            if name == event.name or name == "*":
                matching.append((sub, order))
        matching.sort(key=lambda x: (-x[0].priority, x[1]))
        for sub, _ in matching:
            if sub.filter_fn is not None and not sub.filter_fn(event):
                continue
            sub.handler(event)

    def subscriber_count(self, event_name):
        return sum(1 for (n, _, _) in self._subs.values() if n == event_name)
''',
}

PARSER_COMB = {
    "parser_core.py": '''
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class ParseResult:
    success: bool
    value: Any
    remaining: str
    error: str = ""

class Parser:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, text):
        return self.fn(text)

    def then(self, other):
        def f(text):
            r1 = self.fn(text)
            if not r1.success:
                return r1
            r2 = other.fn(r1.remaining)
            if not r2.success:
                return ParseResult(False, None, text, r2.error)
            return ParseResult(True, (r1.value, r2.value), r2.remaining)
        return Parser(f)

    def or_else(self, other):
        def f(text):
            r1 = self.fn(text)
            if r1.success:
                return r1
            return other.fn(text)
        return Parser(f)

    def map(self, fn):
        def f(text):
            r = self.fn(text)
            if not r.success:
                return r
            return ParseResult(True, fn(r.value), r.remaining)
        return Parser(f)

    def many(self):
        def f(text):
            results = []
            cur = text
            while True:
                r = self.fn(cur)
                if not r.success:
                    break
                results.append(r.value)
                if r.remaining == cur:
                    break
                cur = r.remaining
            return ParseResult(True, results, cur)
        return Parser(f)
''',
    "combinators.py": '''
from parser_core import Parser, ParseResult

def char(c):
    def f(text):
        if text and text[0] == c:
            return ParseResult(True, c, text[1:])
        return ParseResult(False, None, text, f"expected '{c}'")
    return Parser(f)

def digit():
    def f(text):
        if text and text[0].isdigit():
            return ParseResult(True, text[0], text[1:])
        return ParseResult(False, None, text, "expected digit")
    return Parser(f)

def literal(s):
    def f(text):
        if text.startswith(s):
            return ParseResult(True, s, text[len(s):])
        return ParseResult(False, None, text, f"expected '{s}'")
    return Parser(f)

def whitespace():
    def f(text):
        i = 0
        while i < len(text) and text[i] in " \\t":
            i += 1
        return ParseResult(True, text[:i], text[i:])
    return Parser(f)

def integer():
    def f(text):
        i = 0
        if i < len(text) and text[i] == "-":
            i += 1
        start = i
        while i < len(text) and text[i].isdigit():
            i += 1
        if i == start or (i == 1 and text[0] == "-"):
            return ParseResult(False, None, text, "expected integer")
        try:
            return ParseResult(True, int(text[:i]), text[i:])
        except ValueError:
            return ParseResult(False, None, text, "invalid integer")
    return Parser(f)
''',
}

TASK_QUEUE = {
    "queue_types.py": '''
from dataclasses import dataclass, field
from typing import Any
import time

@dataclass
class Task:
    id: int
    payload: Any
    priority: int
    created_at: float = field(default_factory=time.monotonic)

@dataclass
class TaskResult:
    task_id: int
    status: str
    value: Any
    error: str = ""
''',
    "task_queue.py": '''
import heapq
import itertools
from queue_types import Task, TaskResult

class TaskQueue:
    def __init__(self):
        self._heap = []
        self._counter = itertools.count()
        self._next_id = 1
        self._workers = {}
        self._default = None
        self._cancelled = set()

    def submit(self, payload, priority=5):
        tid = self._next_id
        self._next_id += 1
        seq = next(self._counter)
        heapq.heappush(self._heap, (priority, seq, Task(tid, payload, priority)))
        return tid

    def register_worker(self, name, handler):
        self._workers[name] = handler

    def set_default_worker(self, name):
        self._default = name

    def _pop_next(self):
        while self._heap:
            _, _, task = heapq.heappop(self._heap)
            if task.id in self._cancelled:
                self._cancelled.discard(task.id)
                continue
            return task
        return None

    def process_one(self):
        task = self._pop_next()
        if task is None:
            return None
        handler = self._workers.get(self._default)
        try:
            value = handler(task.payload)
            return TaskResult(task.id, "completed", value)
        except Exception as e:
            return TaskResult(task.id, "failed", None, str(e))

    def process_all(self):
        results = []
        while True:
            r = self.process_one()
            if r is None:
                break
            results.append(r)
        return results

    def pending_count(self):
        return sum(1 for _, _, t in self._heap if t.id not in self._cancelled)

    def cancel(self, task_id):
        for _, _, t in self._heap:
            if t.id == task_id and t.id not in self._cancelled:
                self._cancelled.add(task_id)
                return True
        return False
''',
}

REFS = {
    "cache_with_ttl": CACHE_WITH_TTL,
    "rate_limiter": RATE_LIMITER,
    "event_bus_priority": EVENT_BUS,
    "parser_combinator": PARSER_COMB,
    "task_queue": TASK_QUEUE,
}
