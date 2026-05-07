"""Reference: state_persistence, stream_pipeline, mini_interpreter, acl_system, workflow_engine."""
import copy

STATE_PERSIST = {
    "snapshot_types.py": '''
from dataclasses import dataclass, field
from typing import Any
import time

@dataclass
class Snapshot:
    id: int
    label: str
    timestamp: float
    data: dict

@dataclass
class ChangeLog:
    snapshot_id: int
    key: str
    old_value: Any
    new_value: Any
    op: str
''',
    "state_store.py": '''
import copy
import time
from snapshot_types import Snapshot, ChangeLog

class Store:
    def __init__(self, initial_state=None):
        self._state = dict(initial_state) if initial_state else {}
        self._snapshots = {}  # id -> Snapshot
        self._snap_order = []
        self._next_id = 1

    def get(self, key):
        return self._state.get(key)

    def set(self, key, value):
        self._state[key] = value

    def delete(self, key):
        if key in self._state:
            del self._state[key]
            return True
        return False

    def snapshot(self, label=""):
        sid = self._next_id
        self._next_id += 1
        snap = Snapshot(sid, label, time.monotonic(), copy.deepcopy(self._state))
        self._snapshots[sid] = snap
        self._snap_order.append(sid)
        return sid

    def restore(self, snapshot_id):
        if snapshot_id not in self._snapshots:
            return False
        self._state = copy.deepcopy(self._snapshots[snapshot_id].data)
        return True

    def history(self):
        return [self._snapshots[i] for i in self._snap_order]

    def diff(self, snapshot_id_a, snapshot_id_b):
        if snapshot_id_a not in self._snapshots or snapshot_id_b not in self._snapshots:
            return []
        a = self._snapshots[snapshot_id_a].data
        b = self._snapshots[snapshot_id_b].data
        keys = sorted(set(a.keys()) | set(b.keys()))
        changes = []
        for k in keys:
            if k in a and k in b:
                if a[k] != b[k]:
                    changes.append(ChangeLog(snapshot_id_b, k, a[k], b[k], "set"))
            elif k in b:
                changes.append(ChangeLog(snapshot_id_b, k, None, b[k], "set"))
            else:
                changes.append(ChangeLog(snapshot_id_b, k, a[k], None, "delete"))
        return changes

    def rollback(self):
        if not self._snap_order:
            return False
        latest = self._snap_order[-1]
        return self.restore(latest)
''',
}

STREAM = {
    "stream_ops.py": '''
from abc import ABC, abstractmethod

class Operator(ABC):
    @abstractmethod
    def apply(self, iterable): ...

class MapOp(Operator):
    def __init__(self, fn): self.fn = fn
    def apply(self, iterable):
        for x in iterable:
            yield self.fn(x)

class FilterOp(Operator):
    def __init__(self, predicate): self.predicate = predicate
    def apply(self, iterable):
        for x in iterable:
            if self.predicate(x):
                yield x

class TakeOp(Operator):
    def __init__(self, n): self.n = n
    def apply(self, iterable):
        for i, x in enumerate(iterable):
            if i >= self.n:
                break
            yield x

class SkipOp(Operator):
    def __init__(self, n): self.n = n
    def apply(self, iterable):
        for i, x in enumerate(iterable):
            if i >= self.n:
                yield x

class DistinctOp(Operator):
    def apply(self, iterable):
        seen = set()
        for x in iterable:
            if x not in seen:
                seen.add(x)
                yield x

class BatchOp(Operator):
    def __init__(self, size): self.size = size
    def apply(self, iterable):
        batch = []
        for x in iterable:
            batch.append(x)
            if len(batch) == self.size:
                yield batch
                batch = []
        if batch:
            yield batch

class FlattenOp(Operator):
    def apply(self, iterable):
        for x in iterable:
            for y in x:
                yield y
''',
    "stream.py": '''
from stream_ops import Operator, MapOp, FilterOp, TakeOp, SkipOp, DistinctOp, BatchOp, FlattenOp

class Stream:
    def __init__(self, source):
        self._source = source
        self._ops = []

    def _chain(self, op):
        new = Stream(self._source)
        new._ops = self._ops + [op]
        return new

    def map(self, fn): return self._chain(MapOp(fn))
    def filter(self, pred): return self._chain(FilterOp(pred))
    def take(self, n): return self._chain(TakeOp(n))
    def skip(self, n): return self._chain(SkipOp(n))
    def distinct(self): return self._chain(DistinctOp())
    def batch(self, size): return self._chain(BatchOp(size))
    def flatten(self): return self._chain(FlattenOp())

    def _iter(self):
        it = iter(self._source)
        for op in self._ops:
            it = op.apply(it)
        return it

    def to_list(self):
        return list(self._iter())

    def count(self):
        return sum(1 for _ in self._iter())

    def reduce(self, fn, initial):
        acc = initial
        for x in self._iter():
            acc = fn(acc, x)
        return acc

    def first(self, default=None):
        for x in self._iter():
            return x
        return default

    def for_each(self, fn):
        for x in self._iter():
            fn(x)
''',
}

INTERPRETER = {
    "lang_lexer.py": '''
from dataclasses import dataclass
from typing import Any

@dataclass
class Token:
    type: str
    value: Any

class Lexer:
    def tokenize(self, text):
        tokens = []
        i = 0
        n = len(text)
        keywords = {"if", "while", "print"}
        while i < n:
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c.isdigit():
                j = i
                while j < n and text[j].isdigit():
                    j += 1
                tokens.append(Token("NUMBER", int(text[i:j])))
                i = j
                continue
            if c.isalpha() or c == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                word = text[i:j]
                if word in keywords:
                    tokens.append(Token(word.upper(), word))
                else:
                    tokens.append(Token("IDENT", word))
                i = j
                continue
            if c == "=" and i + 1 < n and text[i + 1] == "=":
                tokens.append(Token("EQ", "=="))
                i += 2
                continue
            if c == "=":
                tokens.append(Token("ASSIGN", "="))
                i += 1
                continue
            if c == "<":
                tokens.append(Token("LT", "<"))
                i += 1
                continue
            if c == ">":
                tokens.append(Token("GT", ">"))
                i += 1
                continue
            if c in "+-*/":
                tokens.append(Token("OP", c))
                i += 1
                continue
            if c == ";":
                tokens.append(Token("SEMI", ";"))
                i += 1
                continue
            if c == "{":
                tokens.append(Token("LBRACE", "{"))
                i += 1
                continue
            if c == "}":
                tokens.append(Token("RBRACE", "}"))
                i += 1
                continue
            if c == "(":
                tokens.append(Token("LPAREN", "("))
                i += 1
                continue
            if c == ")":
                tokens.append(Token("RPAREN", ")"))
                i += 1
                continue
            raise SyntaxError(f"unexpected char {c!r} at {i}")
        tokens.append(Token("EOF", None))
        return tokens
''',
    "lang_parser.py": '''
class NumberNode:
    def __init__(self, v): self.v = v

class IdentNode:
    def __init__(self, name): self.name = name

class BinaryNode:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

class AssignNode:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class IfNode:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class WhileNode:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class BlockNode:
    def __init__(self, statements):
        self.statements = statements

class PrintNode:
    def __init__(self, value):
        self.value = value


class Parser:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def parse(self, tokens):
        self.tokens = tokens
        self.pos = 0
        statements = []
        while self._peek().type != "EOF":
            statements.append(self._parse_statement())
        return BlockNode(statements)

    def _peek(self, offset=0):
        return self.tokens[self.pos + offset]

    def _eat(self, type_=None):
        t = self.tokens[self.pos]
        if type_ and t.type != type_:
            raise SyntaxError(f"expected {type_} got {t.type}")
        self.pos += 1
        return t

    def _parse_statement(self):
        t = self._peek()
        if t.type == "PRINT":
            self._eat("PRINT")
            value = self._parse_expr()
            self._eat("SEMI")
            return PrintNode(value)
        if t.type == "IF":
            self._eat("IF")
            cond = self._parse_expr()
            self._eat("LBRACE")
            body = []
            while self._peek().type != "RBRACE":
                body.append(self._parse_statement())
            self._eat("RBRACE")
            return IfNode(cond, BlockNode(body))
        if t.type == "WHILE":
            self._eat("WHILE")
            cond = self._parse_expr()
            self._eat("LBRACE")
            body = []
            while self._peek().type != "RBRACE":
                body.append(self._parse_statement())
            self._eat("RBRACE")
            return WhileNode(cond, BlockNode(body))
        if t.type == "IDENT" and self._peek(1).type == "ASSIGN":
            name = self._eat("IDENT").value
            self._eat("ASSIGN")
            value = self._parse_expr()
            self._eat("SEMI")
            return AssignNode(name, value)
        # expression statement (rare)
        expr = self._parse_expr()
        self._eat("SEMI")
        return expr

    def _parse_expr(self):
        return self._parse_compare()

    def _parse_compare(self):
        left = self._parse_addsub()
        while self._peek().type in ("EQ", "LT", "GT"):
            op = self._eat().type
            right = self._parse_addsub()
            left = BinaryNode(op, left, right)
        return left

    def _parse_addsub(self):
        left = self._parse_muldiv()
        while self._peek().type == "OP" and self._peek().value in "+-":
            op = self._eat().value
            right = self._parse_muldiv()
            left = BinaryNode(op, left, right)
        return left

    def _parse_muldiv(self):
        left = self._parse_atom()
        while self._peek().type == "OP" and self._peek().value in "*/":
            op = self._eat().value
            right = self._parse_atom()
            left = BinaryNode(op, left, right)
        return left

    def _parse_atom(self):
        t = self._peek()
        if t.type == "NUMBER":
            self._eat("NUMBER")
            return NumberNode(t.value)
        if t.type == "IDENT":
            self._eat("IDENT")
            return IdentNode(t.value)
        if t.type == "LPAREN":
            self._eat("LPAREN")
            e = self._parse_expr()
            self._eat("RPAREN")
            return e
        raise SyntaxError(f"unexpected token {t.type}")
''',
    "lang_runtime.py": '''
from lang_lexer import Lexer
from lang_parser import Parser, NumberNode, IdentNode, BinaryNode, AssignNode, IfNode, WhileNode, BlockNode, PrintNode

class Interpreter:
    def __init__(self):
        pass

    def run(self, text):
        tokens = Lexer().tokenize(text)
        ast = Parser().parse(tokens)
        env = {}
        output = []
        self._exec(ast, env, output)
        return output

    def _exec(self, node, env, output):
        if isinstance(node, BlockNode):
            for s in node.statements:
                self._exec(s, env, output)
        elif isinstance(node, PrintNode):
            output.append(self._eval(node.value, env))
        elif isinstance(node, AssignNode):
            env[node.name] = self._eval(node.value, env)
        elif isinstance(node, IfNode):
            if self._eval(node.cond, env) != 0:
                self._exec(node.body, env, output)
        elif isinstance(node, WhileNode):
            while self._eval(node.cond, env) != 0:
                self._exec(node.body, env, output)
        else:
            self._eval(node, env)

    def _eval(self, node, env):
        if isinstance(node, NumberNode):
            return node.v
        if isinstance(node, IdentNode):
            return env[node.name]
        if isinstance(node, BinaryNode):
            l = self._eval(node.left, env)
            r = self._eval(node.right, env)
            if node.op == "+": return l + r
            if node.op == "-": return l - r
            if node.op == "*": return l * r
            if node.op == "/": return l // r
            if node.op == "EQ": return 1 if l == r else 0
            if node.op == "LT": return 1 if l < r else 0
            if node.op == "GT": return 1 if l > r else 0
        raise RuntimeError(f"cannot eval {type(node).__name__}")
''',
}

ACL = {
    "acl_types.py": '''
from dataclasses import dataclass, field
from enum import Enum

@dataclass
class Permission:
    resource: str
    action: str

@dataclass
class Role:
    name: str
    permissions: set = field(default_factory=set)
    inherits: list = field(default_factory=list)

class AccessDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"
''',
    "acl_engine.py": '''
from acl_types import Permission, Role, AccessDecision

class Engine:
    def __init__(self):
        self._roles = {}
        self._user_roles = {}  # user -> set of role names
        # role-level explicit grants/denies (in addition to Role.permissions)
        self._grants = {}      # role_name -> set of (resource, action)
        self._denies = {}      # role_name -> set of (resource, action)

    def define_role(self, role):
        self._roles[role.name] = role
        self._grants.setdefault(role.name, set(role.permissions))
        self._denies.setdefault(role.name, set())

    def assign_role(self, user, role_name):
        self._user_roles.setdefault(user, set()).add(role_name)

    def revoke_role(self, user, role_name):
        if user in self._user_roles and role_name in self._user_roles[user]:
            self._user_roles[user].discard(role_name)
            return True
        return False

    def grant(self, role_name, resource, action):
        self._grants.setdefault(role_name, set()).add((resource, action))

    def deny(self, role_name, resource, action):
        self._denies.setdefault(role_name, set()).add((resource, action))

    def _expand_roles(self, role_name, visited=None):
        if visited is None:
            visited = set()
        if role_name in visited:
            return set()
        visited.add(role_name)
        if role_name not in self._roles:
            return set()
        result = {role_name}
        for parent in self._roles[role_name].inherits:
            result |= self._expand_roles(parent, visited)
        return result

    def _all_roles_for_user(self, user):
        result = set()
        for r in self._user_roles.get(user, set()):
            result |= self._expand_roles(r)
        return result

    def check(self, user, resource, action):
        roles = self._all_roles_for_user(user)
        if not roles:
            return AccessDecision.NOT_APPLICABLE
        # explicit DENY anywhere wins
        for r in roles:
            if (resource, action) in self._denies.get(r, set()):
                return AccessDecision.DENY
        for r in roles:
            if (resource, action) in self._grants.get(r, set()):
                return AccessDecision.ALLOW
        return AccessDecision.NOT_APPLICABLE

    def effective_permissions(self, user):
        roles = self._all_roles_for_user(user)
        granted = set()
        denied = set()
        for r in roles:
            granted |= self._grants.get(r, set())
            denied |= self._denies.get(r, set())
        return granted - denied
''',
}

WORKFLOW = {
    "workflow_types.py": '''
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Step:
    name: str
    action: Callable
    condition: Callable | None = None
    on_failure: str = "fail"

@dataclass
class WorkflowResult:
    completed: bool
    step_results: dict = field(default_factory=dict)
''',
    "workflow.py": '''
from workflow_types import StepStatus, Step, WorkflowResult

class Workflow:
    def __init__(self, name):
        self.name = name
        self._steps = []
        self._status = {}

    def add_step(self, step):
        self._steps.append(step)
        self._status[step.name] = StepStatus.PENDING

    def run(self, context):
        result = WorkflowResult(True, {})
        for s in self._status:
            self._status[s] = StepStatus.PENDING
        result.step_results = {}
        completed = True
        skip_rest = False
        for step in self._steps:
            if skip_rest:
                self._status[step.name] = StepStatus.SKIPPED
                result.step_results[step.name] = {"status": "skipped"}
                continue
            if step.condition is not None:
                try:
                    cond_ok = bool(step.condition(context))
                except Exception:
                    cond_ok = False
                if not cond_ok:
                    self._status[step.name] = StepStatus.SKIPPED
                    result.step_results[step.name] = {"status": "skipped"}
                    continue
            self._status[step.name] = StepStatus.RUNNING
            try:
                value = step.action(context)
                self._status[step.name] = StepStatus.COMPLETED
                result.step_results[step.name] = {"status": "completed", "value": value}
            except Exception as e:
                self._status[step.name] = StepStatus.FAILED
                result.step_results[step.name] = {"status": "failed", "error": str(e)}
                if step.on_failure == "fail":
                    completed = False
                    skip_rest = True
                # 'continue' or 'skip' both proceed
        result.completed = completed
        return result

    def status(self, step_name):
        return self._status.get(step_name, StepStatus.PENDING)

    def reset(self):
        for s in self._status:
            self._status[s] = StepStatus.PENDING
''',
}

REFS = {
    "state_persistence": STATE_PERSIST,
    "stream_pipeline": STREAM,
    "mini_interpreter": INTERPRETER,
    "acl_system": ACL,
    "workflow_engine": WORKFLOW,
}
