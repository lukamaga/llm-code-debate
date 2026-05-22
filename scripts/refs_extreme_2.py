BLOOM = {
    "hash_utils.py": '''
import hashlib
from dataclasses import dataclass

@dataclass
class HashFn:
    seed: int
    def __call__(self, item):
        h = hashlib.md5(self.seed.to_bytes(8, "little", signed=False) + item).digest()
        return int.from_bytes(h, "little") & 0x7FFFFFFF

def make_hashes(k, seeds):
    return [HashFn(seed=s) for s in seeds]
''',
    "bloom.py": '''
from hash_utils import HashFn, make_hashes

class BloomFilter:
    def __init__(self, size, num_hashes, seeds=None):
        self.size = size
        self.num_hashes = num_hashes
        if seeds is None:
            seeds = [(i + 1) * 2654435761 & 0xFFFFFFFF for i in range(num_hashes)]
        if len(seeds) != num_hashes:
            seeds = (seeds * num_hashes)[:num_hashes]
        self.hashes = make_hashes(num_hashes, seeds)
        self.bits = bytearray((size + 7) // 8)
        self._count = 0

    def _set_bit(self, i):
        self.bits[i // 8] |= (1 << (i % 8))

    def _get_bit(self, i):
        return bool(self.bits[i // 8] & (1 << (i % 8)))

    def add(self, item):
        b = item.encode("utf-8")
        for h in self.hashes:
            self._set_bit(h(b) % self.size)
        self._count += 1

    def contains(self, item):
        b = item.encode("utf-8")
        for h in self.hashes:
            if not self._get_bit(h(b) % self.size):
                return False
        return True

    def __len__(self):
        return self._count

    def approximate_fill_ratio(self):
        ones = sum(bin(b).count("1") for b in self.bits)
        return ones / self.size


class CountingBloomFilter(BloomFilter):
    def __init__(self, size, num_hashes, seeds=None):
        super().__init__(size, num_hashes, seeds)
        self.counters = [0] * size

    def _set_bit(self, i):
        # bookkeeping handled separately via counters; bits mirror non-zero counters
        if self.counters[i] == 0:
            self.bits[i // 8] |= (1 << (i % 8))
        self.counters[i] += 1

    def _clear_bit(self, i):
        if self.counters[i] > 0:
            self.counters[i] -= 1
        if self.counters[i] == 0:
            self.bits[i // 8] &= ~(1 << (i % 8))

    def add(self, item):
        b = item.encode("utf-8")
        for h in self.hashes:
            self._set_bit(h(b) % self.size)
        self._count += 1

    def remove(self, item):
        b = item.encode("utf-8")
        positions = [h(b) % self.size for h in self.hashes]
        if any(self.counters[p] == 0 for p in positions):
            return False
        for p in positions:
            self._clear_bit(p)
        if self._count > 0:
            self._count -= 1
        return True
''',
}

SIMPLE_ORM = {
    "orm_fields.py": '''
class Field:
    def __init__(self, default=None, required=False, validator=None):
        self.default = default
        self.required = required
        self.validator = validator

    def validate(self, value):
        if self.validator is not None and not self.validator(value):
            raise ValueError("custom validator failed")


class IntField(Field):
    def validate(self, value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("expected int")
        super().validate(value)


class StringField(Field):
    def validate(self, value):
        if not isinstance(value, str):
            raise ValueError("expected str")
        super().validate(value)


class BoolField(Field):
    def validate(self, value):
        if not isinstance(value, bool):
            raise ValueError("expected bool")
        super().validate(value)


class _ModelMeta(type):
    def __new__(mcs, name, bases, attrs):
        fields = {k: v for k, v in attrs.items() if isinstance(v, Field)}
        attrs["_fields"] = fields
        return super().__new__(mcs, name, bases, attrs)


class Model(metaclass=_ModelMeta):
    def __init__(self, **kwargs):
        for fname, field in self._fields.items():
            if fname in kwargs:
                value = kwargs[fname]
                field.validate(value)
                setattr(self, fname, value)
            elif field.required and field.default is None:
                raise ValueError(f"field {fname} is required")
            else:
                setattr(self, fname, field.default)
''',
    "orm_store.py": '''
from orm_fields import Model

class Store:
    def __init__(self):
        self._data = {}        # cls -> {id: instance}
        self._next_id = {}     # cls -> int
        self._registered = set()

    def register(self, model_cls):
        self._registered.add(model_cls)
        self._data.setdefault(model_cls, {})
        self._next_id.setdefault(model_cls, 1)

    def insert(self, instance):
        cls = type(instance)
        if cls not in self._registered:
            self.register(cls)
        new_id = self._next_id[cls]
        self._next_id[cls] += 1
        instance.id = new_id
        self._data[cls][new_id] = instance
        return new_id

    def get(self, model_cls, id):
        return self._data.get(model_cls, {}).get(id)

    def all(self, model_cls):
        return list(self._data.get(model_cls, {}).values())

    def filter(self, model_cls, **kwargs):
        result = []
        for inst in self._data.get(model_cls, {}).values():
            if all(getattr(inst, k, None) == v for k, v in kwargs.items()):
                result.append(inst)
        return result

    def delete(self, model_cls, id):
        if id in self._data.get(model_cls, {}):
            del self._data[model_cls][id]
            return True
        return False

    def count(self, model_cls):
        return len(self._data.get(model_cls, {}))
''',
}

MIGRATION = {
    "migration_types.py": '''
from dataclasses import dataclass
from typing import Callable

@dataclass
class Migration:
    version: int
    name: str
    up: Callable
    down: Callable

@dataclass
class MigrationRecord:
    version: int
    name: str
    applied_at: float
''',
    "migration_runner.py": '''
import time
from migration_types import Migration, MigrationRecord

class Runner:
    def __init__(self, state):
        self.state = state
        self._migrations = {}
        self._applied = []  # list of MigrationRecord in order

    def register(self, migration):
        self._migrations[migration.version] = migration

    def current_version(self):
        if not self._applied:
            return 0
        return max(r.version for r in self._applied)

    def migrate_to(self, target_version):
        cur = self.current_version()
        if target_version == cur:
            return
        if target_version > cur:
            steps = sorted(v for v in self._migrations if cur < v <= target_version)
            for v in steps:
                m = self._migrations[v]
                m.up(self.state)
                self._applied.append(MigrationRecord(v, m.name, time.monotonic()))
        else:
            applied_versions = [r.version for r in self._applied]
            to_revert = [v for v in reversed(applied_versions) if v > target_version]
            for v in to_revert:
                m = self._migrations[v]
                m.down(self.state)
                # remove from applied
                for i in range(len(self._applied) - 1, -1, -1):
                    if self._applied[i].version == v:
                        del self._applied[i]
                        break

    def applied(self):
        return list(self._applied)

    def pending(self, target_version):
        cur = self.current_version()
        if target_version == cur:
            return []
        if target_version > cur:
            return [self._migrations[v] for v in sorted(self._migrations) if cur < v <= target_version]
        applied_versions = [r.version for r in self._applied]
        to_revert = [v for v in reversed(applied_versions) if v > target_version]
        return [self._migrations[v] for v in to_revert]
''',
}

DI = {
    "di_types.py": '''
from enum import Enum
from dataclasses import dataclass
from typing import Callable

class Lifetime(Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"

@dataclass
class Registration:
    key: str
    factory: Callable
    lifetime: Lifetime

class ResolutionError(Exception):
    pass
''',
    "di_container.py": '''
from di_types import Lifetime, Registration, ResolutionError

class Container:
    def __init__(self):
        self._registrations = {}
        self._singletons = {}
        self._resolving = set()

    def register(self, key, factory, lifetime=Lifetime.TRANSIENT):
        self._registrations[key] = Registration(key, factory, lifetime)
        # invalidate any cached singleton
        self._singletons.pop(key, None)

    def register_value(self, key, value):
        self._registrations[key] = Registration(key, lambda c: value, Lifetime.SINGLETON)
        self._singletons[key] = value

    def resolve(self, key):
        if key not in self._registrations:
            raise ResolutionError(f"unknown key: {key}")
        if key in self._resolving:
            raise ResolutionError(f"circular dependency detected for: {key}")
        reg = self._registrations[key]
        if reg.lifetime == Lifetime.SINGLETON and key in self._singletons:
            return self._singletons[key]
        self._resolving.add(key)
        try:
            value = reg.factory(self)
        finally:
            self._resolving.discard(key)
        if reg.lifetime == Lifetime.SINGLETON:
            self._singletons[key] = value
        return value

    def has(self, key):
        return key in self._registrations
''',
}

COMMAND_UNDO = {
    "command_types.py": '''
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time

class Command(ABC):
    @property
    @abstractmethod
    def name(self): ...
    @abstractmethod
    def execute(self, state): ...
    @abstractmethod
    def undo(self, state): ...

@dataclass
class CommandResult:
    command_name: str
    value: Any
    timestamp: float = field(default_factory=time.monotonic)
''',
    "command_invoker.py": '''
from command_types import Command, CommandResult

_MISSING = object()


class SetCommand(Command):
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self._prev = _MISSING

    @property
    def name(self):
        return f"set({self.key!r}, {self.value!r})"

    def execute(self, state):
        self._prev = state.get(self.key, _MISSING)
        state[self.key] = self.value
        return self.value

    def undo(self, state):
        if self._prev is _MISSING:
            state.pop(self.key, None)
        else:
            state[self.key] = self._prev


class DeleteCommand(Command):
    def __init__(self, key):
        self.key = key
        self._existed = False
        self._prev = _MISSING

    @property
    def name(self):
        return f"delete({self.key!r})"

    def execute(self, state):
        if self.key in state:
            self._existed = True
            self._prev = state[self.key]
            del state[self.key]
            return self._prev
        self._existed = False
        return None

    def undo(self, state):
        if self._existed:
            state[self.key] = self._prev


class Invoker:
    def __init__(self, state):
        self.state = state
        self._undo = []
        self._redo = []

    def execute(self, command):
        value = command.execute(self.state)
        self._undo.append(command)
        self._redo.clear()
        return CommandResult(command.name, value)

    def undo(self):
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo(self.state)
        self._redo.append(cmd)
        return True

    def redo(self):
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.execute(self.state)
        self._undo.append(cmd)
        return True

    def can_undo(self):
        return len(self._undo) > 0

    def can_redo(self):
        return len(self._redo) > 0

    def history(self):
        return [c.name for c in self._undo]
''',
}

REFS = {
    "bloom_filter": BLOOM,
    "simple_orm": SIMPLE_ORM,
    "migration_runner": MIGRATION,
    "di_container": DI,
    "command_undo": COMMAND_UNDO,
}
