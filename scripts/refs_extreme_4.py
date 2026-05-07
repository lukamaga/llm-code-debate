"""Reference: lock_manager, b_tree, ast_calculator, template_compiler, mock_framework."""

LOCK = {
    "lock_types.py": '''
from enum import Enum
from dataclasses import dataclass, field

class LockMode(Enum):
    READ = "read"
    WRITE = "write"

@dataclass
class LockResult:
    acquired: bool
    holders: list = field(default_factory=list)
    reason: str = ""

class DeadlockError(Exception):
    pass
''',
    "lock_manager.py": '''
from lock_types import LockMode, LockResult, DeadlockError

class LockManager:
    def __init__(self):
        # resource -> (mode, [holders])
        self._locks = {}

    def try_acquire(self, resource, holder, mode):
        if resource not in self._locks:
            self._locks[resource] = (mode, [holder])
            return LockResult(True, [holder])
        cur_mode, holders = self._locks[resource]
        # idempotent same-mode reacquire
        if holder in holders and cur_mode == mode:
            return LockResult(True, list(holders))
        # upgrade case: holder is sole READ holder asking WRITE
        if holder in holders and cur_mode == LockMode.READ and mode == LockMode.WRITE:
            if len(holders) == 1:
                self._locks[resource] = (LockMode.WRITE, [holder])
                return LockResult(True, [holder])
            return LockResult(False, list(holders), "cannot upgrade with other readers")
        if cur_mode == LockMode.READ and mode == LockMode.READ:
            new_holders = list(holders)
            if holder not in new_holders:
                new_holders.append(holder)
            self._locks[resource] = (LockMode.READ, new_holders)
            return LockResult(True, list(new_holders))
        return LockResult(False, list(holders), "incompatible mode")

    def release(self, resource, holder):
        if resource not in self._locks:
            return False
        mode, holders = self._locks[resource]
        if holder not in holders:
            return False
        new_holders = [h for h in holders if h != holder]
        if not new_holders:
            del self._locks[resource]
        else:
            self._locks[resource] = (mode, new_holders)
        return True

    def held_by(self, holder):
        return [r for r, (_, hs) in self._locks.items() if holder in hs]

    def resources(self):
        return {r: list(hs) for r, (_, hs) in self._locks.items()}

    def detect_deadlock(self, holder, resource, mode):
        # Simple wait-for analysis: would granting create a cycle?
        # For this synchronous bookkeeping we approximate: return False (no real wait queue)
        # but answer True if holder already holds this resource in conflicting mode
        # and someone else holds it.
        if resource not in self._locks:
            return False
        cur_mode, holders = self._locks[resource]
        if holder in holders:
            return False
        # If another holder also waits for one of holder's resources, that's a cycle
        held_by_holder = set(self.held_by(holder))
        for other in holders:
            if other == holder:
                continue
            other_held = set(self.held_by(other))
            if other_held & held_by_holder:
                return True
        return False
''',
}

BTREE = {
    "btree_node.py": '''
class BTreeNode:
    def __init__(self, is_leaf=True):
        self.keys = []
        self.values = []
        self.children = []
        self.is_leaf = is_leaf

    def is_full(self, min_degree):
        return len(self.keys) >= 2 * min_degree - 1

    def split(self, min_degree):
        mid = min_degree - 1
        median_key = self.keys[mid]
        median_value = self.values[mid]
        right = BTreeNode(is_leaf=self.is_leaf)
        right.keys = self.keys[mid + 1:]
        right.values = self.values[mid + 1:]
        if not self.is_leaf:
            right.children = self.children[mid + 1:]
            self.children = self.children[:mid + 1]
        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        return median_key, median_value, right
''',
    "btree.py": '''
from btree_node import BTreeNode

class BTree:
    def __init__(self, min_degree=3):
        self.t = min_degree
        self.root = None
        self._size = 0

    def insert(self, key, value):
        # update existing key
        if self._update_existing(self.root, key, value):
            return
        if self.root is None:
            self.root = BTreeNode(is_leaf=True)
            self.root.keys.append(key)
            self.root.values.append(value)
            self._size += 1
            return
        if self.root.is_full(self.t):
            new_root = BTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_nonfull(self.root, key, value)
        self._size += 1

    def _update_existing(self, node, key, value):
        if node is None:
            return False
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == key:
            node.values[i] = value
            return True
        if node.is_leaf:
            return False
        return self._update_existing(node.children[i], key, value)

    def _split_child(self, parent, idx):
        child = parent.children[idx]
        median_key, median_value, right = child.split(self.t)
        parent.keys.insert(idx, median_key)
        parent.values.insert(idx, median_value)
        parent.children.insert(idx + 1, right)

    def _insert_nonfull(self, node, key, value):
        i = len(node.keys) - 1
        if node.is_leaf:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
            node.values.insert(i + 1, value)
            return
        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1
        if node.children[i].is_full(self.t):
            self._split_child(node, i)
            if key > node.keys[i]:
                i += 1
        self._insert_nonfull(node.children[i], key, value)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return None
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == key:
            return node.values[i]
        if node.is_leaf:
            return None
        return self._search(node.children[i], key)

    def delete(self, key):
        # Simple delete: rebuild from in-order traversal, omitting key.
        if self.search(key) is None:
            return False
        items = self.range_query(float("-inf") if isinstance(key, (int, float)) else None, None)
        items = [(k, v) for k, v in self._inorder(self.root) if k != key]
        self.root = None
        self._size = 0
        for k, v in items:
            self.insert(k, v)
        return True

    def _inorder(self, node):
        if node is None:
            return
        if node.is_leaf:
            for k, v in zip(node.keys, node.values):
                yield k, v
            return
        for i, k in enumerate(node.keys):
            yield from self._inorder(node.children[i])
            yield k, node.values[i]
        yield from self._inorder(node.children[-1])

    def range_query(self, low, high):
        result = []
        for k, v in self._inorder(self.root):
            if (low is None or k >= low) and (high is None or k <= high):
                result.append((k, v))
        return result

    def size(self):
        return self._size

    def height(self):
        if self.root is None:
            return 0
        return self._height(self.root)

    def _height(self, node):
        if node.is_leaf:
            return 1
        return 1 + max(self._height(c) for c in node.children)
''',
}

AST_CALC = {
    "ast_nodes.py": '''
from abc import ABC, abstractmethod

class ASTNode(ABC):
    @abstractmethod
    def evaluate(self, env): ...

class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = value
    def evaluate(self, env):
        return float(self.value)

class VarNode(ASTNode):
    def __init__(self, name):
        self.name = name
    def evaluate(self, env):
        if self.name not in env:
            raise NameError(self.name)
        return float(env[self.name])

class BinaryOpNode(ASTNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
    def evaluate(self, env):
        a = self.left.evaluate(env)
        b = self.right.evaluate(env)
        if self.op == "+": return a + b
        if self.op == "-": return a - b
        if self.op == "*": return a * b
        if self.op == "/":
            if b == 0:
                raise ZeroDivisionError()
            return a / b
        raise ValueError(f"unknown op {self.op}")

class UnaryOpNode(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand
    def evaluate(self, env):
        v = self.operand.evaluate(env)
        if self.op == "-": return -v
        if self.op == "+": return v
        raise ValueError()

class AssignNode(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    def evaluate(self, env):
        v = self.value.evaluate(env)
        env[self.name] = v
        return v
''',
    "expr_parser.py": '''
from ast_nodes import ASTNode, NumberNode, VarNode, BinaryOpNode, UnaryOpNode, AssignNode

class Parser:
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def _peek(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _eat(self, ch=None):
        c = self._peek()
        self.pos += 1
        return c

    def parse(self):
        node = self._parse_assign()
        return node

    def _parse_assign(self):
        save = self.pos
        # try identifier '=' expression
        ident = self._try_identifier()
        if ident is not None:
            if self._peek() == "=":
                # check it's not '=='
                if self.pos + 1 < len(self.text) and self.text[self.pos + 1] != "=":
                    self._eat("=")
                    expr = self._parse_expr()
                    return AssignNode(ident, expr)
        self.pos = save
        return self._parse_expr()

    def _try_identifier(self):
        save = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1
        start = self.pos
        if self.pos < len(self.text) and (self.text[self.pos].isalpha() or self.text[self.pos] == "_"):
            while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
                self.pos += 1
            return self.text[start:self.pos]
        self.pos = save
        return None

    def _parse_expr(self):
        node = self._parse_term()
        while self._peek() and self._peek() in "+-":
            op = self._eat()
            right = self._parse_term()
            node = BinaryOpNode(op, node, right)
        return node

    def _parse_term(self):
        node = self._parse_unary()
        while self._peek() and self._peek() in "*/":
            op = self._eat()
            right = self._parse_unary()
            node = BinaryOpNode(op, node, right)
        return node

    def _parse_unary(self):
        if self._peek() == "-":
            self._eat("-")
            return UnaryOpNode("-", self._parse_unary())
        if self._peek() == "+":
            self._eat("+")
            return self._parse_unary()
        return self._parse_atom()

    def _parse_atom(self):
        c = self._peek()
        if c == "(":
            self._eat("(")
            node = self._parse_expr()
            if self._peek() == ")":
                self._eat(")")
            return node
        if c.isdigit() or c == ".":
            start = self.pos
            while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == "."):
                self.pos += 1
            return NumberNode(self.text[start:self.pos])
        if c.isalpha() or c == "_":
            ident = self._try_identifier()
            return VarNode(ident)
        raise SyntaxError(f"unexpected '{c}' at pos {self.pos}")
''',
    "calculator.py": '''
from expr_parser import Parser

class Calculator:
    def __init__(self):
        self.env = {}

    def eval(self, expr):
        ast = Parser(expr).parse()
        return ast.evaluate(self.env)
''',
}

TEMPLATE = {
    "template_ast.py": '''
from abc import ABC, abstractmethod

class TemplateNode(ABC):
    @abstractmethod
    def render(self, context): ...

class TextNode(TemplateNode):
    def __init__(self, text):
        self.text = text
    def render(self, context):
        return self.text

class VarNode(TemplateNode):
    def __init__(self, name):
        self.name = name
    def render(self, context):
        return str(context.get(self.name, ""))

class IfNode(TemplateNode):
    def __init__(self, var_name, body):
        self.var_name = var_name
        self.body = body
    def render(self, context):
        if context.get(self.var_name):
            return "".join(n.render(context) for n in self.body)
        return ""

class ForNode(TemplateNode):
    def __init__(self, var_name, iterable_name, body):
        self.var_name = var_name
        self.iterable_name = iterable_name
        self.body = body
    def render(self, context):
        items = context.get(self.iterable_name, []) or []
        out = []
        for it in items:
            new_ctx = dict(context)
            new_ctx[self.var_name] = it
            out.append("".join(n.render(new_ctx) for n in self.body))
        return "".join(out)
''',
    "template_engine.py": '''
import re
from template_ast import TemplateNode, TextNode, VarNode, IfNode, ForNode


class _CompiledTemplate:
    def __init__(self, nodes):
        self.nodes = nodes
    def render(self, context):
        return "".join(n.render(context) for n in self.nodes)


def _tokenize(source):
    pattern = re.compile(r"({{.*?}}|{%.*?%})", re.DOTALL)
    parts = []
    last = 0
    for m in pattern.finditer(source):
        if m.start() > last:
            parts.append(("TEXT", source[last:m.start()]))
        token = m.group(0)
        if token.startswith("{{"):
            parts.append(("VAR", token[2:-2].strip()))
        else:
            parts.append(("TAG", token[2:-2].strip()))
        last = m.end()
    if last < len(source):
        parts.append(("TEXT", source[last:]))
    return parts


def _parse(tokens, pos=0, terminator=None):
    nodes = []
    while pos < len(tokens):
        kind, value = tokens[pos]
        if kind == "TAG":
            if terminator and value == terminator:
                return nodes, pos + 1
            if value.startswith("if "):
                var_name = value[3:].strip()
                body, pos = _parse(tokens, pos + 1, "endif")
                nodes.append(IfNode(var_name, body))
                continue
            if value.startswith("for "):
                rest = value[4:].strip()
                var_name, _, iterable = rest.partition(" in ")
                body, pos = _parse(tokens, pos + 1, "endfor")
                nodes.append(ForNode(var_name.strip(), iterable.strip(), body))
                continue
        elif kind == "VAR":
            nodes.append(VarNode(value))
        else:
            nodes.append(TextNode(value))
        pos += 1
    return nodes, pos


class Engine:
    def __init__(self):
        self._cache = {}

    def compile(self, source):
        tokens = _tokenize(source)
        nodes, _ = _parse(tokens, 0, None)
        return _CompiledTemplate(nodes)

    def render(self, source, context):
        if source not in self._cache:
            self._cache[source] = self.compile(source)
        return self._cache[source].render(context)

    def cache_size(self):
        return len(self._cache)
''',
}

MOCK = {
    "mock_types.py": '''
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Call:
    args: tuple
    kwargs: dict
    return_value: Any = None

@dataclass
class MockSpec:
    name: str
    return_value: Any = None
    side_effect: Any = None
''',
    "mock_lib.py": '''
import inspect
from mock_types import Call, MockSpec

_SENTINEL = object()


def _is_exc(v):
    if isinstance(v, BaseException):
        return True
    if inspect.isclass(v) and issubclass(v, BaseException):
        return True
    return False


class Mock:
    def __init__(self, name="mock", return_value=None, side_effect=None):
        self._name = name
        self._return_value = return_value
        self._side_effect = side_effect
        self._side_iter = None
        self._history = []

    def __call__(self, *args, **kwargs):
        if self._side_effect is not None:
            se = self._side_effect
            if _is_exc(se):
                self._history.append(Call(args, dict(kwargs), None))
                raise se if isinstance(se, BaseException) else se()
            if callable(se):
                value = se(*args, **kwargs)
                self._history.append(Call(args, dict(kwargs), value))
                return value
            # iterable
            if self._side_iter is None:
                self._side_iter = iter(se)
            value = next(self._side_iter)
            self._history.append(Call(args, dict(kwargs), value))
            return value
        value = self._return_value
        self._history.append(Call(args, dict(kwargs), value))
        return value

    @property
    def call_count(self):
        return len(self._history)

    @property
    def call_history(self):
        return list(self._history)

    @property
    def called(self):
        return len(self._history) > 0

    def called_with(self, *args, **kwargs):
        for c in self._history:
            if c.args == args and c.kwargs == kwargs:
                return True
        return False

    def called_n_times(self, n):
        return self.call_count == n

    def reset(self):
        self._history = []
        self._side_iter = None

    def configure(self, return_value=_SENTINEL, side_effect=_SENTINEL):
        if return_value is not _SENTINEL:
            self._return_value = return_value
        if side_effect is not _SENTINEL:
            self._side_effect = side_effect
            self._side_iter = None


class Spy:
    def __init__(self, target):
        self._target = target
        self._history = []

    def __call__(self, *args, **kwargs):
        value = self._target(*args, **kwargs)
        self._history.append(Call(args, dict(kwargs), value))
        return value

    @property
    def call_count(self):
        return len(self._history)

    @property
    def call_history(self):
        return list(self._history)

    @property
    def called(self):
        return len(self._history) > 0

    def called_with(self, *args, **kwargs):
        for c in self._history:
            if c.args == args and c.kwargs == kwargs:
                return True
        return False

    def called_n_times(self, n):
        return self.call_count == n

    def reset(self):
        self._history = []
''',
}

REFS = {
    "lock_manager": LOCK,
    "b_tree": BTREE,
    "ast_calculator": AST_CALC,
    "template_compiler": TEMPLATE,
    "mock_framework": MOCK,
}
