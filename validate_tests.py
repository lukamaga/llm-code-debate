#!/usr/bin/env python3

import json
import os
import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path

TASK_DIRS = [
    Path("/Users/lukashm/Desktop/llm-code-debate/tasks/hard"),
    Path("/Users/lukashm/Desktop/llm-code-debate/tasks/extreme"),
]

HARD_IMPLEMENTATIONS = {
    "alien_dictionary": '''
from collections import defaultdict, deque

def alien_order(words: list[str]) -> str:
    adj = defaultdict(set)
    in_degree = {c: 0 for w in words for c in w}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        min_len = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in adj[w1[j]]:
                    adj[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for n in sorted(adj[c]):
            in_degree[n] -= 1
            if in_degree[n] == 0:
                queue.append(n)

    if len(result) != len(in_degree):
        return ""
    return "".join(result)
''',

    "count_inversions": '''
def count_inversions(arr: list[int]) -> int:
    if len(arr) <= 1:
        return 0
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    inv = count_inversions(left) + count_inversions(right)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            inv += len(left) - i
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
    return inv
''',

    "lfu_cache": '''
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.min_freq = 0
        self.key_val = {}
        self.key_freq = {}
        self.freq_keys = defaultdict(OrderedDict)

    def get(self, key: int) -> int:
        if key not in self.key_val:
            return -1
        freq = self.key_freq[key]
        del self.freq_keys[freq][key]
        if not self.freq_keys[freq]:
            del self.freq_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.key_freq[key] = freq + 1
        self.freq_keys[freq + 1][key] = None
        return self.key_val[key]

    def put(self, key: int, value: int) -> None:
        if self.cap <= 0:
            return
        if key in self.key_val:
            self.key_val[key] = value
            self.get(key)
            return
        if len(self.key_val) >= self.cap:
            evict_key, _ = self.freq_keys[self.min_freq].popitem(last=False)
            if not self.freq_keys[self.min_freq]:
                del self.freq_keys[self.min_freq]
            del self.key_val[evict_key]
            del self.key_freq[evict_key]
        self.key_val[key] = value
        self.key_freq[key] = 1
        self.freq_keys[1][key] = None
        self.min_freq = 1
''',

    "linked_list_lib": '''
class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val: int) -> None:
        if not self.head:
            self.head = Node(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = Node(val)

    def to_list(self) -> list[int]:
        result = []
        curr = self.head
        while curr:
            result.append(curr.val)
            curr = curr.next
        return result

    def reverse(self) -> None:
        prev = None
        curr = self.head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        self.head = prev

    def remove(self, val: int) -> bool:
        if not self.head:
            return False
        if self.head.val == val:
            self.head = self.head.next
            return True
        curr = self.head
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
                return True
            curr = curr.next
        return False

    def find(self, val: int) -> bool:
        curr = self.head
        while curr:
            if curr.val == val:
                return True
            curr = curr.next
        return False

    def size(self) -> int:
        count = 0
        curr = self.head
        while curr:
            count += 1
            curr = curr.next
        return count

    def insert_at(self, index: int, val: int) -> bool:
        if index < 0:
            return False
        if index == 0:
            self.head = Node(val, self.head)
            return True
        curr = self.head
        for _ in range(index - 1):
            if not curr:
                return False
            curr = curr.next
        if not curr:
            return False
        curr.next = Node(val, curr.next)
        return True

    def get_at(self, index: int) -> int:
        curr = self.head
        for _ in range(index):
            if not curr:
                return -1
            curr = curr.next
        return curr.val if curr else -1
''',

    "max_profit_k_transactions": '''
def max_profit(k: int, prices: list[int]) -> int:
    n = len(prices)
    if n <= 1 or k == 0:
        return 0
    if k >= n // 2:
        return sum(max(prices[i+1] - prices[i], 0) for i in range(n-1))
    dp = [[0] * n for _ in range(k + 1)]
    for t in range(1, k + 1):
        max_diff = -prices[0]
        for d in range(1, n):
            dp[t][d] = max(dp[t][d-1], prices[d] + max_diff)
            max_diff = max(max_diff, dp[t-1][d] - prices[d])
    return dp[k][n-1]
''',

    "median_sorted_arrays": '''
def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = (m + n + 1) // 2 - i
        left1 = nums1[i-1] if i > 0 else float('-inf')
        right1 = nums1[i] if i < m else float('inf')
        left2 = nums2[j-1] if j > 0 else float('-inf')
        right2 = nums2[j] if j < n else float('inf')
        if left1 <= right2 and left2 <= right1:
            if (m + n) % 2 == 0:
                return (max(left1, left2) + min(right1, right2)) / 2
            else:
                return max(left1, left2)
        elif left1 > right2:
            hi = i - 1
        else:
            lo = i + 1
    return 0.0
''',

    "merge_intervals": '''
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged
''',

    "merge_k_sorted": '''
import heapq

def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    result = []
    while heap:
        val, li, idx = heapq.heappop(heap)
        result.append(val)
        if idx + 1 < len(lists[li]):
            heapq.heappush(heap, (lists[li][idx+1], li, idx+1))
    return result
''',

    "minimum_window_substring": '''
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not t or not s:
        return ""
    need = Counter(t)
    have = {}
    formed = 0
    required = len(need)
    l = 0
    ans = (float('inf'), 0, 0)
    for r, c in enumerate(s):
        have[c] = have.get(c, 0) + 1
        if c in need and have[c] == need[c]:
            formed += 1
        while formed == required:
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)
            have[s[l]] -= 1
            if s[l] in need and have[s[l]] < need[s[l]]:
                formed -= 1
            l += 1
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2]+1]
''',

    "regex_matching": '''
def is_match(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2]
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]
''',

    "shortest_path": '''
import heapq
from collections import defaultdict

def shortest_path(n: int, edges: list[list[int]], source: int, target: int) -> int:
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    dist = [float('inf')] * n
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == target:
            return d
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return -1
''',

    "sliding_window_max": '''
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    if not nums or k == 0:
        return []
    dq = deque()
    result = []
    for i, n in enumerate(nums):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < n:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
''',

    "text_justification": '''
def full_justify(words: list[str], maxWidth: int) -> list[str]:
    result = []
    i = 0
    while i < len(words):
        line = [words[i]]
        line_len = len(words[i])
        i += 1
        while i < len(words) and line_len + 1 + len(words[i]) <= maxWidth:
            line_len += 1 + len(words[i])
            line.append(words[i])
            i += 1
        if i == len(words) or len(line) == 1:
            left = " ".join(line)
            result.append(left + " " * (maxWidth - len(left)))
        else:
            total_spaces = maxWidth - sum(len(w) for w in line)
            gaps = len(line) - 1
            space_per = total_spaces // gaps
            extra = total_spaces % gaps
            s = ""
            for j, w in enumerate(line):
                s += w
                if j < gaps:
                    s += " " * (space_per + (1 if j < extra else 0))
            result.append(s)
    return result
''',

    "topological_sort": '''
from collections import defaultdict, deque

def topological_sort(num_nodes: int, edges: list[list[int]]) -> list[int]:
    graph = defaultdict(list)
    in_degree = [0] * num_nodes
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result if len(result) == num_nodes else []
''',

    "word_ladder": '''
from collections import deque

def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    queue = deque([(begin_word, 1)])
    visited = {begin_word}
    while queue:
        word, length = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word == end_word:
                    return length + 1
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, length + 1))
    return 0
''',
}

EXTREME_IMPLEMENTATIONS = {
    "calculator": {
        "tokenizer.py": '''
from enum import Enum
from dataclasses import dataclass

class TokenType(Enum):
    NUMBER = 'NUMBER'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    EOF = 'EOF'

@dataclass
class Token:
    type: TokenType
    value: str | int

class Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isspace():
                self.pos += 1
            elif c.isdigit():
                num = ''
                while self.pos < len(self.text) and self.text[self.pos].isdigit():
                    num += self.text[self.pos]
                    self.pos += 1
                tokens.append(Token(TokenType.NUMBER, int(num)))
            elif c == '+':
                tokens.append(Token(TokenType.PLUS, '+'))
                self.pos += 1
            elif c == '-':
                tokens.append(Token(TokenType.MINUS, '-'))
                self.pos += 1
            elif c == '*':
                tokens.append(Token(TokenType.MULTIPLY, '*'))
                self.pos += 1
            elif c == '/' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '/':
                tokens.append(Token(TokenType.DIVIDE, '//'))
                self.pos += 2
            elif c == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
                self.pos += 1
            elif c == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
                self.pos += 1
            else:
                self.pos += 1
        tokens.append(Token(TokenType.EOF, None))
        return tokens
''',
        "evaluator.py": '''
from tokenizer import Tokenizer, Token, TokenType

class ExpressionCalculator:
    def evaluate(self, expr: str) -> int:
        tokenizer = Tokenizer(expr)
        self.tokens = tokenizer.tokenize()
        self.pos = 0
        result = self._expr()
        return result

    def _current(self):
        return self.tokens[self.pos]

    def _eat(self, tt):
        if self._current().type == tt:
            t = self._current()
            self.pos += 1
            return t
        raise ValueError(f"Expected {tt}")

    def _expr(self):
        result = self._term()
        while self._current().type in (TokenType.PLUS, TokenType.MINUS):
            if self._current().type == TokenType.PLUS:
                self._eat(TokenType.PLUS)
                result += self._term()
            else:
                self._eat(TokenType.MINUS)
                result -= self._term()
        return result

    def _term(self):
        result = self._unary()
        while self._current().type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            if self._current().type == TokenType.MULTIPLY:
                self._eat(TokenType.MULTIPLY)
                result *= self._unary()
            else:
                self._eat(TokenType.DIVIDE)
                divisor = self._unary()
                result = int(result / divisor)
        return result

    def _unary(self):
        if self._current().type == TokenType.MINUS:
            self._eat(TokenType.MINUS)
            return -self._unary()
        return self._factor()

    def _factor(self):
        if self._current().type == TokenType.NUMBER:
            return self._eat(TokenType.NUMBER).value
        if self._current().type == TokenType.LPAREN:
            self._eat(TokenType.LPAREN)
            result = self._expr()
            self._eat(TokenType.RPAREN)
            return result
        raise ValueError(f"Unexpected token: {self._current()}")
''',
    },

    "state_machine": {
        "state_types.py": '''
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class Action:
    name: str
    handler: Callable[[dict], None]

@dataclass
class Transition:
    event: str
    source: str
    target: str
    guard: Callable[[dict], bool] | None = None
    actions: list[Action] = field(default_factory=list)

@dataclass
class State:
    name: str
    on_enter: Callable[[dict], None] | None = None
    on_exit: Callable[[dict], None] | None = None
''',
        "fsm_engine.py": '''
from state_types import State, Transition, Action

class StateMachine:
    def __init__(self, initial_state: str, context: dict = None):
        self._current = initial_state
        self.context = context or {}
        self.states = {}
        self.transitions = []
        self._history = [initial_state]

    def add_state(self, state: State) -> None:
        self.states[state.name] = state

    def add_transition(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def send(self, event: str) -> bool:
        for t in self.transitions:
            if t.event == event and t.source == self._current:
                if t.guard and not t.guard(self.context):
                    return False
                old = self.states.get(self._current)
                if old and old.on_exit:
                    old.on_exit(self.context)
                for action in t.actions:
                    action.handler(self.context)
                self._current = t.target
                new = self.states.get(self._current)
                if new and new.on_enter:
                    new.on_enter(self.context)
                self._history.append(self._current)
                return True
        return False

    @property
    def current_state(self) -> str:
        return self._current

    def get_available_events(self) -> list[str]:
        return [t.event for t in self.transitions if t.source == self._current]

    def get_history(self) -> list[str]:
        return self._history
''',
    },

    "event_system": {
        "event_types.py": '''
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class Event:
    name: str
    data: dict = field(default_factory=dict)
    _stopped: bool = field(default=False, repr=False)

    def stop_propagation(self):
        self._stopped = True

    @property
    def is_stopped(self):
        return self._stopped

@dataclass
class Middleware:
    name: str
    handler: Callable[['Event'], 'Event | None']
''',
        "event_bus.py": '''
from event_types import Event, Middleware
from collections import defaultdict

class EventBus:
    def __init__(self):
        self._handlers = defaultdict(list)
        self._middlewares = []
        self._history = []

    def on(self, event_name: str, handler, priority: int = 0):
        self._handlers[event_name].append((priority, handler))
        self._handlers[event_name].sort(key=lambda x: -x[0])

    def off(self, event_name: str, handler):
        self._handlers[event_name] = [(p, h) for p, h in self._handlers[event_name] if h != handler]

    def emit(self, event: Event) -> Event:
        for mw in self._middlewares:
            result = mw.handler(event)
            if result is None:
                return event
            event = result
        self._history.append(event)
        for _, handler in self._handlers.get(event.name, []):
            if event.is_stopped:
                break
            handler(event)
        return event

    def use(self, middleware: Middleware):
        self._middlewares.append(middleware)

    def get_history(self) -> list[Event]:
        return self._history

    def clear(self):
        self._handlers.clear()
        self._middlewares.clear()
        self._history.clear()
''',
    },

    "json_parser": {
        "tokenizer.py": '''
from enum import Enum
from dataclasses import dataclass
from typing import Any

class TokenType(Enum):
    LBRACE = '{'
    RBRACE = '}'
    LBRACKET = '['
    RBRACKET = ']'
    COLON = ':'
    COMMA = ','
    STRING = 'STRING'
    NUMBER = 'NUMBER'
    TRUE = 'true'
    FALSE = 'false'
    NULL = 'null'
    EOF = 'EOF'

@dataclass
class Token:
    type: TokenType
    value: Any

class Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.pos < len(self.text):
            c = self.text[self.pos]
            if c.isspace():
                self.pos += 1
            elif c == '{':
                tokens.append(Token(TokenType.LBRACE, '{'))
                self.pos += 1
            elif c == '}':
                tokens.append(Token(TokenType.RBRACE, '}'))
                self.pos += 1
            elif c == '[':
                tokens.append(Token(TokenType.LBRACKET, '['))
                self.pos += 1
            elif c == ']':
                tokens.append(Token(TokenType.RBRACKET, ']'))
                self.pos += 1
            elif c == ':':
                tokens.append(Token(TokenType.COLON, ':'))
                self.pos += 1
            elif c == ',':
                tokens.append(Token(TokenType.COMMA, ','))
                self.pos += 1
            elif c == '"':
                tokens.append(self._string())
            elif c == '-' or c.isdigit():
                tokens.append(self._number())
            elif self.text[self.pos:self.pos+4] == 'true':
                tokens.append(Token(TokenType.TRUE, True))
                self.pos += 4
            elif self.text[self.pos:self.pos+5] == 'false':
                tokens.append(Token(TokenType.FALSE, False))
                self.pos += 5
            elif self.text[self.pos:self.pos+4] == 'null':
                tokens.append(Token(TokenType.NULL, None))
                self.pos += 4
            else:
                raise ValueError(f"Unexpected character: {c}")
        tokens.append(Token(TokenType.EOF, None))
        return tokens

    def _string(self) -> Token:
        self.pos += 1  # skip opening quote
        result = ''
        while self.pos < len(self.text) and self.text[self.pos] != '"':
            if self.text[self.pos] == '\\\\':
                self.pos += 1
                c = self.text[self.pos]
                if c == 'n': result += '\\n'
                elif c == 't': result += '\\t'
                elif c == '"': result += '"'
                elif c == '\\\\': result += '\\\\'
                elif c == '/': result += '/'
                else: result += c
            else:
                result += self.text[self.pos]
            self.pos += 1
        self.pos += 1  # skip closing quote
        return Token(TokenType.STRING, result)

    def _number(self) -> Token:
        start = self.pos
        if self.text[self.pos] == '-':
            self.pos += 1
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        is_float = False
        if self.pos < len(self.text) and self.text[self.pos] == '.':
            is_float = True
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        if self.pos < len(self.text) and self.text[self.pos] in ('e', 'E'):
            is_float = True
            self.pos += 1
            if self.pos < len(self.text) and self.text[self.pos] in ('+', '-'):
                self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        num_str = self.text[start:self.pos]
        if is_float:
            return Token(TokenType.NUMBER, float(num_str))
        return Token(TokenType.NUMBER, int(num_str))
''',
        "parser.py": '''
from tokenizer import Tokenizer, Token, TokenType

class JSONParser:
    def parse(self, text: str):
        tokenizer = Tokenizer(text)
        self.tokens = tokenizer.tokenize()
        self.pos = 0
        result = self._value()
        return result

    def _current(self):
        return self.tokens[self.pos]

    def _eat(self, tt):
        if self._current().type == tt:
            t = self._current()
            self.pos += 1
            return t
        raise ValueError(f"Expected {tt}, got {self._current().type}")

    def _value(self):
        t = self._current()
        if t.type == TokenType.LBRACE:
            return self._object()
        elif t.type == TokenType.LBRACKET:
            return self._array()
        elif t.type == TokenType.STRING:
            self.pos += 1
            return t.value
        elif t.type == TokenType.NUMBER:
            self.pos += 1
            return t.value
        elif t.type == TokenType.TRUE:
            self.pos += 1
            return True
        elif t.type == TokenType.FALSE:
            self.pos += 1
            return False
        elif t.type == TokenType.NULL:
            self.pos += 1
            return None
        raise ValueError(f"Unexpected token: {t}")

    def _object(self):
        self._eat(TokenType.LBRACE)
        result = {}
        if self._current().type != TokenType.RBRACE:
            key = self._eat(TokenType.STRING).value
            self._eat(TokenType.COLON)
            value = self._value()
            result[key] = value
            while self._current().type == TokenType.COMMA:
                self._eat(TokenType.COMMA)
                key = self._eat(TokenType.STRING).value
                self._eat(TokenType.COLON)
                value = self._value()
                result[key] = value
        self._eat(TokenType.RBRACE)
        return result

    def _array(self):
        self._eat(TokenType.LBRACKET)
        result = []
        if self._current().type != TokenType.RBRACKET:
            result.append(self._value())
            while self._current().type == TokenType.COMMA:
                self._eat(TokenType.COMMA)
                result.append(self._value())
        self._eat(TokenType.RBRACKET)
        return result
''',
    },

    "mini_database": {
        "storage.py": '''
import json
import os

class Storage:
    def __init__(self):
        self.tables = {}

    def create_table(self, name: str, columns: list[str]):
        self.tables[name] = {"columns": columns, "rows": [], "auto_id": 1}

    def table_exists(self, name: str) -> bool:
        return name in self.tables

    def get_table(self, name: str):
        return self.tables.get(name)

    def insert(self, table_name: str, row: dict) -> int:
        table = self.tables[table_name]
        row_id = table["auto_id"]
        table["auto_id"] += 1
        row["id"] = row_id
        table["rows"].append(row)
        return row_id

    def select(self, table_name: str, where=None):
        table = self.tables[table_name]
        rows = table["rows"]
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        return rows

    def update(self, table_name: str, values: dict, where: dict) -> int:
        table = self.tables[table_name]
        count = 0
        for row in table["rows"]:
            if all(row.get(k) == v for k, v in where.items()):
                row.update(values)
                count += 1
        return count

    def delete(self, table_name: str, where: dict) -> int:
        table = self.tables[table_name]
        before = len(table["rows"])
        table["rows"] = [r for r in table["rows"] if not all(r.get(k) == v for k, v in where.items())]
        return before - len(table["rows"])
''',
        "database.py": '''
from storage import Storage

class Database:
    def __init__(self):
        self.storage = Storage()

    def create_table(self, name: str, columns: list[str]):
        if self.storage.table_exists(name):
            raise ValueError(f"Table {name} already exists")
        self.storage.create_table(name, columns)

    def insert(self, table_name: str, row: dict) -> int:
        if not self.storage.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        return self.storage.insert(table_name, row)

    def select(self, table_name: str, where: dict = None) -> list[dict]:
        if not self.storage.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        return self.storage.select(table_name, where)

    def update(self, table_name: str, values: dict, where: dict) -> int:
        if not self.storage.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        return self.storage.update(table_name, values, where)

    def delete(self, table_name: str, where: dict) -> int:
        if not self.storage.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        return self.storage.delete(table_name, where)

    def count(self, table_name: str, where: dict = None) -> int:
        return len(self.select(table_name, where))
''',
    },

    "plugin_system": {
        "plugin_base.py": '''
from dataclasses import dataclass, field
from typing import Any

@dataclass
class HookResult:
    modified: bool = False
    data: Any = None
    errors: list[str] = field(default_factory=list)

class Plugin:
    name: str = "base"
    version: str = "1.0.0"
    enabled: bool = True

    def activate(self, context: dict) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def process(self, hook_name: str, data: Any) -> HookResult:
        return HookResult()
''',
        "plugin_manager.py": '''
from plugin_base import Plugin, HookResult
from typing import Any

class PluginManager:
    def __init__(self):
        self._plugins = {}
        self._hooks = {}
        self._context = {}

    def register(self, plugin: Plugin) -> None:
        self._plugins[plugin.name] = plugin
        plugin.activate(self._context)

    def unregister(self, name: str) -> bool:
        if name in self._plugins:
            self._plugins[name].deactivate()
            del self._plugins[name]
            return True
        return False

    def get_plugin(self, name: str) -> Plugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        return list(self._plugins.keys())

    def execute_hook(self, hook_name: str, data: Any = None) -> list[HookResult]:
        results = []
        for plugin in self._plugins.values():
            if plugin.enabled:
                result = plugin.process(hook_name, data)
                results.append(result)
        return results

    def enable(self, name: str) -> bool:
        if name in self._plugins:
            self._plugins[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        if name in self._plugins:
            self._plugins[name].enabled = False
            return True
        return False
''',
    },

    "task_scheduler": {
        "task_model.py": '''
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
import time

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    name: str
    priority: Priority
    handler: Callable[[], Any]
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    result: Any = None
    error: str | None = None
''',
        "scheduler.py": '''
from task_model import Task, Priority
from collections import defaultdict, deque
from typing import Any

class TaskScheduler:
    def __init__(self):
        self._tasks = {}
        self._results = {}
        self._execution_order = []

    def add_task(self, task: Task) -> None:
        self._tasks[task.name] = task

    def remove_task(self, name: str) -> bool:
        if name in self._tasks:
            del self._tasks[name]
            return True
        return False

    def get_task(self, name: str) -> Task | None:
        return self._tasks.get(name)

    def run_all(self) -> dict[str, Any]:
        order = self._topological_sort()
        results = {}
        for name in order:
            task = self._tasks[name]
            deps_ok = all(
                self._tasks[d].status == "completed"
                for d in task.dependencies
                if d in self._tasks
            )
            if not deps_ok:
                task.status = "failed"
                task.error = "dependency failed"
                results[name] = None
                continue
            try:
                task.result = task.handler()
                task.status = "completed"
                results[name] = task.result
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                results[name] = None
            self._execution_order.append(name)
        self._results = results
        return results

    def _topological_sort(self) -> list[str]:
        in_degree = {name: 0 for name in self._tasks}
        graph = defaultdict(list)
        for name, task in self._tasks.items():
            for dep in task.dependencies:
                if dep in self._tasks:
                    graph[dep].append(name)
                    in_degree[name] += 1
        queue = []
        for name in self._tasks:
            if in_degree[name] == 0:
                queue.append(self._tasks[name])
        queue.sort(key=lambda t: -t.priority.value)
        q = deque([t.name for t in queue])
        result = []
        while q:
            name = q.popleft()
            result.append(name)
            candidates = []
            for neighbor in graph[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    candidates.append(self._tasks[neighbor])
            candidates.sort(key=lambda t: -t.priority.value)
            for c in candidates:
                q.append(c.name)
        return result

    def get_execution_order(self) -> list[str]:
        return self._execution_order

    def get_results(self) -> dict[str, Any]:
        return self._results
''',
    },

    "http_router": {
        "route_types.py": '''
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class Request:
    method: str
    path: str
    headers: dict = field(default_factory=dict)
    body: Any = None
    params: dict = field(default_factory=dict)

@dataclass
class Response:
    status: int = 200
    body: Any = None
    headers: dict = field(default_factory=dict)

@dataclass
class Route:
    method: str
    path: str
    handler: Callable[[Request], Response]

@dataclass
class Middleware:
    name: str
    handler: Callable[[Request, Callable], Response]
''',
        "router.py": '''
import re
from route_types import Request, Response, Route, Middleware

class Router:
    def __init__(self):
        self._routes = []
        self._middlewares = []
        self._prefix = ""

    def add_route(self, method: str, path: str, handler):
        full_path = self._prefix + path
        self._routes.append(Route(method.upper(), full_path, handler))

    def get(self, path: str, handler):
        self.add_route("GET", path, handler)

    def post(self, path: str, handler):
        self.add_route("POST", path, handler)

    def put(self, path: str, handler):
        self.add_route("PUT", path, handler)

    def delete(self, path: str, handler):
        self.add_route("DELETE", path, handler)

    def use(self, middleware: Middleware):
        self._middlewares.append(middleware)

    def group(self, prefix: str) -> 'Router':
        sub = Router()
        sub._routes = self._routes
        sub._middlewares = self._middlewares
        sub._prefix = self._prefix + prefix
        return sub

    def handle(self, request: Request) -> Response:
        for route in self._routes:
            params = self._match(route, request)
            if params is not None:
                request.params = params
                handler = route.handler
                for mw in reversed(self._middlewares):
                    prev_handler = handler
                    handler = lambda req, h=prev_handler, m=mw: m.handler(req, h)
                return handler(request)
        return Response(status=404, body="Not Found")

    def _match(self, route: Route, request: Request):
        if route.method != request.method:
            return None
        pattern_parts = route.path.strip('/').split('/')
        path_parts = request.path.strip('/').split('/')
        if len(pattern_parts) != len(path_parts):
            return None
        params = {}
        for pp, rp in zip(pattern_parts, path_parts):
            if pp.startswith(':'):
                params[pp[1:]] = rp
            elif pp != rp:
                return None
        return params
''',
    },

    "schema_validator": {
        "schema_types.py": '''
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class SchemaType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ANY = "any"

@dataclass
class ValidationError:
    path: str
    message: str

@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

@dataclass
class Schema:
    type: SchemaType
    required: bool = True
    properties: dict[str, 'Schema'] = field(default_factory=dict)
    items: 'Schema | None' = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None
    enum: list | None = None
''',
        "validator.py": '''
import re
from schema_types import Schema, SchemaType, ValidationResult, ValidationError

class SchemaValidator:
    def validate(self, data, schema: Schema, path: str = "$") -> ValidationResult:
        errors = []
        self._validate(data, schema, path, errors)
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _validate(self, data, schema: Schema, path: str, errors: list):
        if data is None:
            if schema.required:
                errors.append(ValidationError(path, "Value is required"))
            return

        type_map = {
            SchemaType.STRING: str,
            SchemaType.INTEGER: int,
            SchemaType.FLOAT: (int, float),
            SchemaType.BOOLEAN: bool,
            SchemaType.ARRAY: list,
            SchemaType.OBJECT: dict,
        }

        if schema.type != SchemaType.ANY:
            expected = type_map.get(schema.type)
            if expected:
                if schema.type == SchemaType.INTEGER and isinstance(data, bool):
                    errors.append(ValidationError(path, f"Expected {schema.type.value}, got boolean"))
                    return
                if schema.type == SchemaType.BOOLEAN:
                    if not isinstance(data, bool):
                        errors.append(ValidationError(path, f"Expected {schema.type.value}, got {type(data).__name__}"))
                        return
                elif not isinstance(data, expected):
                    errors.append(ValidationError(path, f"Expected {schema.type.value}, got {type(data).__name__}"))
                    return

        if schema.type == SchemaType.STRING and isinstance(data, str):
            if schema.min_length is not None and len(data) < schema.min_length:
                errors.append(ValidationError(path, f"String too short"))
            if schema.max_length is not None and len(data) > schema.max_length:
                errors.append(ValidationError(path, f"String too long"))
            if schema.pattern is not None and not re.search(schema.pattern, data):
                errors.append(ValidationError(path, f"Does not match pattern"))

        if schema.type in (SchemaType.INTEGER, SchemaType.FLOAT) and isinstance(data, (int, float)):
            if schema.minimum is not None and data < schema.minimum:
                errors.append(ValidationError(path, f"Value below minimum"))
            if schema.maximum is not None and data > schema.maximum:
                errors.append(ValidationError(path, f"Value above maximum"))

        if schema.enum is not None and data not in schema.enum:
            errors.append(ValidationError(path, f"Value not in enum"))

        if schema.type == SchemaType.ARRAY and isinstance(data, list):
            if schema.min_length is not None and len(data) < schema.min_length:
                errors.append(ValidationError(path, f"Array too short"))
            if schema.max_length is not None and len(data) > schema.max_length:
                errors.append(ValidationError(path, f"Array too long"))
            if schema.items:
                for i, item in enumerate(data):
                    self._validate(item, schema.items, f"{path}[{i}]", errors)

        if schema.type == SchemaType.OBJECT and isinstance(data, dict):
            for key, prop_schema in schema.properties.items():
                if key in data:
                    self._validate(data[key], prop_schema, f"{path}.{key}", errors)
                elif prop_schema.required:
                    errors.append(ValidationError(f"{path}.{key}", "Value is required"))
''',
    },

    "template_engine": {
        "lexer.py": '''
from enum import Enum
from dataclasses import dataclass
from typing import Any

class TokenType(Enum):
    TEXT = "TEXT"
    VARIABLE = "VARIABLE"
    IF_START = "IF_START"
    IF_END = "IF_END"
    ELSE = "ELSE"
    FOR_START = "FOR_START"
    FOR_END = "FOR_END"
    COMMENT = "COMMENT"

@dataclass
class Token:
    type: TokenType
    value: Any = None

def tokenize(template: str) -> list[Token]:
    tokens = []
    i = 0
    while i < len(template):
        if template[i:i+2] == '{{':
            end = template.find('}}', i+2)
            if end == -1:
                tokens.append(Token(TokenType.TEXT, template[i:]))
                break
            content = template[i+2:end].strip()
            tokens.append(Token(TokenType.VARIABLE, content))
            i = end + 2
        elif template[i:i+2] == '{%':
            end = template.find('%}', i+2)
            if end == -1:
                tokens.append(Token(TokenType.TEXT, template[i:]))
                break
            content = template[i+2:end].strip()
            if content.startswith('if '):
                tokens.append(Token(TokenType.IF_START, content[3:].strip()))
            elif content == 'endif':
                tokens.append(Token(TokenType.IF_END))
            elif content == 'else':
                tokens.append(Token(TokenType.ELSE))
            elif content.startswith('for '):
                tokens.append(Token(TokenType.FOR_START, content[4:].strip()))
            elif content == 'endfor':
                tokens.append(Token(TokenType.FOR_END))
            i = end + 2
        elif template[i:i+2] == '{#':
            end = template.find('#}', i+2)
            if end == -1:
                tokens.append(Token(TokenType.TEXT, template[i:]))
                break
            tokens.append(Token(TokenType.COMMENT, template[i+2:end].strip()))
            i = end + 2
        else:
            end = i
            while end < len(template) and template[end:end+2] not in ('{{', '{%', '{#'):
                end += 1
            tokens.append(Token(TokenType.TEXT, template[i:end]))
            i = end
    return tokens
''',
        "renderer.py": '''
from lexer import tokenize, Token, TokenType

class TemplateEngine:
    def render(self, template: str, context: dict = None) -> str:
        context = context or {}
        tokens = tokenize(template)
        result, _ = self._render_tokens(tokens, 0, context)
        return result

    def _render_tokens(self, tokens, pos, context):
        result = ''
        while pos < len(tokens):
            token = tokens[pos]
            if token.type == TokenType.TEXT:
                result += token.value
                pos += 1
            elif token.type == TokenType.VARIABLE:
                result += str(self._resolve(token.value, context))
                pos += 1
            elif token.type == TokenType.IF_START:
                cond = self._resolve(token.value, context)
                pos += 1
                if_body, pos = self._render_tokens(tokens, pos, context)
                else_body = ''
                if pos < len(tokens) and tokens[pos].type == TokenType.ELSE:
                    pos += 1
                    else_body, pos = self._render_tokens(tokens, pos, context)
                if pos < len(tokens) and tokens[pos].type == TokenType.IF_END:
                    pos += 1
                result += if_body if cond else else_body
            elif token.type == TokenType.FOR_START:
                parts = token.value.split(' in ')
                var_name = parts[0].strip()
                iter_name = parts[1].strip()
                pos += 1
                loop_start = pos
                loop_body_tokens = []
                depth = 1
                while pos < len(tokens) and depth > 0:
                    if tokens[pos].type == TokenType.FOR_START:
                        depth += 1
                    elif tokens[pos].type == TokenType.FOR_END:
                        depth -= 1
                        if depth == 0:
                            break
                    loop_body_tokens.append(tokens[pos])
                    pos += 1
                if pos < len(tokens):
                    pos += 1  # skip FOR_END
                items = self._resolve(iter_name, context)
                if items:
                    for item in items:
                        sub_ctx = dict(context)
                        sub_ctx[var_name] = item
                        body, _ = self._render_tokens(loop_body_tokens, 0, sub_ctx)
                        result += body
            elif token.type in (TokenType.IF_END, TokenType.FOR_END, TokenType.ELSE):
                break
            elif token.type == TokenType.COMMENT:
                pos += 1
            else:
                pos += 1
        return result, pos

    def _resolve(self, expr: str, context: dict):
        parts = expr.split('.')
        val = context
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = getattr(val, part, None)
            if val is None:
                return ''
        return val
''',
    },

    "data_pipeline": {
        "transforms.py": '''
from typing import Any, Callable

class MapTransform:
    def __init__(self, func: Callable):
        self.func = func

    def apply(self, data: list) -> list:
        return [self.func(item) for item in data]

class FilterTransform:
    def __init__(self, predicate: Callable):
        self.predicate = predicate

    def apply(self, data: list) -> list:
        return [item for item in data if self.predicate(item)]

class AggregateTransform:
    def __init__(self, func: Callable, initial=None):
        self.func = func
        self.initial = initial

    def apply(self, data: list):
        result = self.initial
        for item in data:
            if result is None:
                result = item
            else:
                result = self.func(result, item)
        return result

class SortTransform:
    def __init__(self, key: Callable = None, reverse: bool = False):
        self.key = key
        self.reverse = reverse

    def apply(self, data: list) -> list:
        return sorted(data, key=self.key, reverse=self.reverse)
''',
        "validators.py": '''
from typing import Any, Callable

class DataValidator:
    def __init__(self):
        self._rules = []

    def add_rule(self, name: str, check: Callable[[Any], bool], message: str = ""):
        self._rules.append((name, check, message))

    def validate(self, data) -> tuple[bool, list[str]]:
        errors = []
        for name, check, message in self._rules:
            try:
                if not check(data):
                    errors.append(message or f"Validation failed: {name}")
            except Exception as e:
                errors.append(f"Validation error in {name}: {str(e)}")
        return (len(errors) == 0, errors)
''',
        "pipeline.py": '''
from typing import Any

class Pipeline:
    def __init__(self):
        self._steps = []
        self._validator = None

    def add_step(self, transform) -> 'Pipeline':
        self._steps.append(transform)
        return self

    def set_validator(self, validator):
        self._validator = validator
        return self

    def execute(self, data):
        if self._validator:
            valid, errors = self._validator.validate(data)
            if not valid:
                raise ValueError(f"Validation failed: {errors}")
        result = data
        for step in self._steps:
            result = step.apply(result)
        return result

class PipelineBuilder:
    def __init__(self):
        self._pipeline = Pipeline()

    def add_step(self, transform) -> 'PipelineBuilder':
        self._pipeline.add_step(transform)
        return self

    def set_validator(self, validator) -> 'PipelineBuilder':
        self._pipeline.set_validator(validator)
        return self

    def build(self) -> Pipeline:
        return self._pipeline
''',
    },

    "expression_evaluator": {
        "ast_nodes.py": '''
from dataclasses import dataclass
from typing import Any

@dataclass
class NumberNode:
    value: float

@dataclass
class BinaryOpNode:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOpNode:
    op: str
    operand: Any

@dataclass
class VariableNode:
    name: str

@dataclass
class FunctionCallNode:
    name: str
    args: list

@dataclass
class AssignmentNode:
    name: str
    value: Any
''',
        "interpreter.py": '''
import math
from ast_nodes import NumberNode, BinaryOpNode, UnaryOpNode, VariableNode, FunctionCallNode, AssignmentNode

class Interpreter:
    def __init__(self):
        self.variables = {}
        self.functions = {
            'abs': abs,
            'min': min,
            'max': max,
            'sqrt': math.sqrt,
            'pow': pow,
        }

    def evaluate(self, expr: str):
        expr = expr.strip()
        if '=' in expr and not any(expr.split('=')[0].strip().endswith(c) for c in ['!', '<', '>']):
            parts = expr.split('=', 1)
            if not parts[0].strip().endswith(('!', '<', '>')) and '==' not in expr[:expr.index('=')+1]:
                name = parts[0].strip()
                value = self._parse_and_eval(parts[1].strip())
                self.variables[name] = value
                return value
        return self._parse_and_eval(expr)

    def _parse_and_eval(self, expr: str):
        tokens = self._tokenize(expr)
        self._tokens = tokens
        self._pos = 0
        result = self._parse_expr()
        return result

    def _tokenize(self, expr):
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i].isspace():
                i += 1
            elif expr[i].isdigit() or (expr[i] == '.' and i+1 < len(expr) and expr[i+1].isdigit()):
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(('NUM', float(expr[i:j])))
                i = j
            elif expr[i].isalpha() or expr[i] == '_':
                j = i
                while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                    j += 1
                tokens.append(('ID', expr[i:j]))
                i = j
            elif expr[i] in '+-*/%':
                tokens.append(('OP', expr[i]))
                i += 1
            elif expr[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif expr[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            elif expr[i] == ',':
                tokens.append(('COMMA', ','))
                i += 1
            elif expr[i] == '*' and i+1 < len(expr) and expr[i+1] == '*':
                tokens.append(('OP', '**'))
                i += 2
            else:
                i += 1
        tokens.append(('EOF', None))
        return tokens

    def _current(self):
        return self._tokens[self._pos]

    def _eat(self, type_=None):
        t = self._tokens[self._pos]
        self._pos += 1
        return t

    def _parse_expr(self):
        left = self._parse_term()
        while self._current()[0] == 'OP' and self._current()[1] in ('+', '-'):
            op = self._eat()[1]
            right = self._parse_term()
            if op == '+':
                left = left + right
            else:
                left = left - right
        return left

    def _parse_term(self):
        left = self._parse_power()
        while self._current()[0] == 'OP' and self._current()[1] in ('*', '/', '%'):
            op = self._eat()[1]
            right = self._parse_power()
            if op == '*':
                left = left * right
            elif op == '/':
                left = left / right
            else:
                left = left % right
        return left

    def _parse_power(self):
        base = self._parse_unary()
        if self._current()[0] == 'OP' and self._current()[1] == '**':
            self._eat()
            exp = self._parse_power()
            return base ** exp
        return base

    def _parse_unary(self):
        if self._current()[0] == 'OP' and self._current()[1] == '-':
            self._eat()
            return -self._parse_unary()
        if self._current()[0] == 'OP' and self._current()[1] == '+':
            self._eat()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self):
        if self._current()[0] == 'NUM':
            val = self._eat()[1]
            return val if val != int(val) else int(val)
        if self._current()[0] == 'ID':
            name = self._eat()[1]
            if self._current()[0] == 'LPAREN':
                self._eat()
                args = []
                if self._current()[0] != 'RPAREN':
                    args.append(self._parse_expr())
                    while self._current()[0] == 'COMMA':
                        self._eat()
                        args.append(self._parse_expr())
                self._eat()  # RPAREN
                if name in self.functions:
                    return self.functions[name](*args)
                raise ValueError(f"Unknown function: {name}")
            if name in self.variables:
                return self.variables[name]
            raise ValueError(f"Unknown variable: {name}")
        if self._current()[0] == 'LPAREN':
            self._eat()
            val = self._parse_expr()
            self._eat()  # RPAREN
            return val
        raise ValueError(f"Unexpected token: {self._current()}")
''',
    },

    "graph_algorithms": {
        "graph_types.py": '''
from dataclasses import dataclass
from enum import Enum
from typing import Any

class EdgeType(Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"

@dataclass
class Vertex:
    id: str
    data: Any = None

@dataclass
class Edge:
    source: str
    target: str
    weight: float = 1.0
    edge_type: EdgeType = EdgeType.UNDIRECTED
''',
        "graph.py": '''
from graph_types import Vertex, Edge, EdgeType
from collections import defaultdict

class Graph:
    def __init__(self, directed: bool = False):
        self.directed = directed
        self._vertices = {}
        self._adj = defaultdict(list)
        self._edges = []

    def add_vertex(self, vertex: Vertex):
        self._vertices[vertex.id] = vertex

    def add_edge(self, edge: Edge):
        self._edges.append(edge)
        self._adj[edge.source].append((edge.target, edge.weight))
        if not self.directed:
            self._adj[edge.target].append((edge.source, edge.weight))

    def get_neighbors(self, vertex_id: str) -> list[tuple[str, float]]:
        return self._adj.get(vertex_id, [])

    def get_vertices(self) -> list[str]:
        return list(self._vertices.keys())

    def get_edges(self) -> list[Edge]:
        return self._edges

    def has_vertex(self, vertex_id: str) -> bool:
        return vertex_id in self._vertices

    def has_edge(self, source: str, target: str) -> bool:
        return any(t == target for t, _ in self._adj.get(source, []))
''',
        "algorithms.py": '''
from graph_types import Vertex, Edge, EdgeType
from graph import Graph
from collections import deque, defaultdict
import heapq

def bfs(graph: Graph, start: str) -> list[str]:
    visited = set()
    result = []
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor, _ in graph.get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result

def dfs(graph: Graph, start: str) -> list[str]:
    visited = set()
    result = []
    def _dfs(node):
        visited.add(node)
        result.append(node)
        for neighbor, _ in graph.get_neighbors(node):
            if neighbor not in visited:
                _dfs(neighbor)
    _dfs(start)
    return result

def dijkstra(graph: Graph, start: str, end: str) -> tuple[float, list[str]]:
    dist = {start: 0}
    prev = {}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        if u == end:
            break
        for v, w in graph.get_neighbors(u):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if end not in dist:
        return (float('inf'), [])
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev.get(node)
    return (dist[end], list(reversed(path)))

def has_cycle(graph: Graph) -> bool:
    vertices = graph.get_vertices()
    if graph.directed:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {v: WHITE for v in vertices}
        def _dfs(v):
            color[v] = GRAY
            for neighbor, _ in graph.get_neighbors(v):
                if color.get(neighbor) == GRAY:
                    return True
                if color.get(neighbor) == WHITE and _dfs(neighbor):
                    return True
            color[v] = BLACK
            return False
        for v in vertices:
            if color[v] == WHITE:
                if _dfs(v):
                    return True
        return False
    else:
        visited = set()
        def _dfs(v, parent):
            visited.add(v)
            for neighbor, _ in graph.get_neighbors(v):
                if neighbor not in visited:
                    if _dfs(neighbor, v):
                        return True
                elif neighbor != parent:
                    return True
            return False
        for v in vertices:
            if v not in visited:
                if _dfs(v, None):
                    return True
        return False

def topological_sort(graph: Graph) -> list[str]:
    vertices = graph.get_vertices()
    in_degree = {v: 0 for v in vertices}
    for v in vertices:
        for neighbor, _ in graph.get_neighbors(v):
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    queue = deque([v for v in vertices if in_degree[v] == 0])
    result = []
    while queue:
        v = queue.popleft()
        result.append(v)
        for neighbor, _ in graph.get_neighbors(v):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result if len(result) == len(vertices) else []
''',
    },

    "http_client": {
        "http_types.py": '''
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class Header:
    name: str
    value: str

@dataclass
class HttpRequest:
    method: HttpMethod
    url: str
    headers: list[Header] = field(default_factory=list)
    body: Any = None

@dataclass
class HttpResponse:
    status_code: int
    headers: list[Header] = field(default_factory=list)
    body: Any = None
''',
        "mock_server.py": '''
from http_types import HttpMethod, HttpRequest, HttpResponse, Header

class MockServer:
    def __init__(self):
        self._routes = {}
        self._requests = []

    def register(self, method: str, path: str, response: HttpResponse):
        key = (method.upper(), path)
        self._routes[key] = response

    def handle(self, request: HttpRequest) -> HttpResponse:
        self._requests.append(request)
        path = request.url.split("://", 1)[-1]
        path = "/" + path.split("/", 1)[-1] if "/" in path else "/"
        key = (request.method.value, path)
        if key in self._routes:
            return self._routes[key]
        return HttpResponse(status_code=404, body="Not Found")

    def get_requests(self) -> list[HttpRequest]:
        return self._requests

    def reset(self):
        self._requests.clear()
''',
        "client.py": '''
from http_types import HttpMethod, HttpRequest, HttpResponse, Header
from mock_server import MockServer

class HttpClient:
    def __init__(self, server: MockServer = None):
        self._server = server
        self._default_headers = []
        self._interceptors = []

    def set_default_header(self, name: str, value: str):
        self._default_headers.append(Header(name, value))

    def get(self, url: str, headers: list[Header] = None) -> HttpResponse:
        return self._send(HttpMethod.GET, url, headers)

    def post(self, url: str, body=None, headers: list[Header] = None) -> HttpResponse:
        return self._send(HttpMethod.POST, url, headers, body)

    def put(self, url: str, body=None, headers: list[Header] = None) -> HttpResponse:
        return self._send(HttpMethod.PUT, url, headers, body)

    def delete(self, url: str, headers: list[Header] = None) -> HttpResponse:
        return self._send(HttpMethod.DELETE, url, headers)

    def _send(self, method: HttpMethod, url: str, headers=None, body=None) -> HttpResponse:
        all_headers = list(self._default_headers)
        if headers:
            all_headers.extend(headers)
        request = HttpRequest(method=method, url=url, headers=all_headers, body=body)
        for interceptor in self._interceptors:
            request = interceptor(request)
        if self._server:
            return self._server.handle(request)
        return HttpResponse(status_code=500, body="No server configured")

    def add_interceptor(self, interceptor):
        self._interceptors.append(interceptor)
''',
    },

    "query_engine": {
        "query_parser.py": '''
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Condition:
    field: str
    operator: str
    value: Any

@dataclass
class SelectQuery:
    table: str
    columns: list[str] = field(default_factory=lambda: ['*'])
    conditions: list[Condition] = field(default_factory=list)
    order_by: str | None = None
    order_dir: str = "ASC"
    limit: int | None = None

def parse(query: str) -> SelectQuery:
    query = query.strip()
    # Parse SELECT ... FROM ...
    upper = query.upper()
    select_idx = upper.index("SELECT") + 6
    from_idx = upper.index("FROM")
    cols_str = query[select_idx:from_idx].strip()
    columns = [c.strip() for c in cols_str.split(',')]

    rest = query[from_idx + 4:].strip()
    # Get table name
    parts = rest.split()
    table = parts[0]

    conditions = []
    order_by = None
    order_dir = "ASC"
    limit = None

    rest_upper = rest.upper()

    # Parse WHERE
    if 'WHERE' in rest_upper:
        where_idx = rest_upper.index('WHERE') + 5
        end_idx = len(rest)
        for kw in ['ORDER', 'LIMIT']:
            if kw in rest_upper[where_idx:]:
                end_idx = rest_upper.index(kw, where_idx)
                break
        where_clause = rest[where_idx:end_idx].strip()
        # Split by AND
        cond_parts = where_clause.split(' AND ')
        for cp in cond_parts:
            cp = cp.strip()
            for op in ['>=', '<=', '!=', '=', '>', '<', 'LIKE']:
                op_upper = op
                idx = cp.upper().find(op_upper)
                if idx != -1:
                    field_name = cp[:idx].strip()
                    value_str = cp[idx+len(op):].strip()
                    if value_str.startswith("'") and value_str.endswith("'"):
                        value = value_str[1:-1]
                    elif '.' in value_str:
                        value = float(value_str)
                    else:
                        try:
                            value = int(value_str)
                        except ValueError:
                            value = value_str
                    conditions.append(Condition(field_name, op, value))
                    break

    # Parse ORDER BY
    if 'ORDER BY' in rest_upper:
        ob_idx = rest_upper.index('ORDER BY') + 8
        end_idx = len(rest)
        if 'LIMIT' in rest_upper[ob_idx:]:
            end_idx = rest_upper.index('LIMIT', ob_idx)
        ob_str = rest[ob_idx:end_idx].strip()
        ob_parts = ob_str.split()
        order_by = ob_parts[0]
        if len(ob_parts) > 1 and ob_parts[1].upper() == 'DESC':
            order_dir = "DESC"

    # Parse LIMIT
    if 'LIMIT' in rest_upper:
        lim_idx = rest_upper.index('LIMIT') + 5
        limit = int(rest[lim_idx:].strip())

    return SelectQuery(table=table, columns=columns, conditions=conditions,
                       order_by=order_by, order_dir=order_dir, limit=limit)
''',
        "executor.py": '''
from query_parser import parse, SelectQuery, Condition

class QueryEngine:
    def __init__(self):
        self._tables = {}

    def create_table(self, name: str, rows: list[dict]):
        self._tables[name] = rows

    def execute(self, query_str: str) -> list[dict]:
        query = parse(query_str)
        if query.table not in self._tables:
            raise ValueError(f"Table {query.table} not found")
        rows = list(self._tables[query.table])

        # Apply conditions
        for cond in query.conditions:
            rows = [r for r in rows if self._eval_condition(r, cond)]

        # Apply ORDER BY
        if query.order_by:
            reverse = query.order_dir == "DESC"
            rows.sort(key=lambda r: r.get(query.order_by, ''), reverse=reverse)

        # Apply LIMIT
        if query.limit is not None:
            rows = rows[:query.limit]

        # Select columns
        if query.columns != ['*']:
            rows = [{c: r.get(c) for c in query.columns} for r in rows]

        return rows

    def _eval_condition(self, row, cond: Condition) -> bool:
        val = row.get(cond.field)
        if cond.operator == '=':
            return val == cond.value
        elif cond.operator == '!=':
            return val != cond.value
        elif cond.operator == '>':
            return val > cond.value
        elif cond.operator == '<':
            return val < cond.value
        elif cond.operator == '>=':
            return val >= cond.value
        elif cond.operator == '<=':
            return val <= cond.value
        elif cond.operator.upper() == 'LIKE':
            import re
            pattern = cond.value.replace('%', '.*').replace('_', '.')
            return bool(re.match(f'^{pattern}$', str(val)))
        return False
''',
    },
}

def validate_hard_task(task_path: Path) -> dict:
    with open(task_path) as f:
        task = json.load(f)

    task_id = task["id"]
    impl = HARD_IMPLEMENTATIONS.get(task_id)
    if not impl:
        return {"task": task_id, "status": "SKIP", "message": "No reference implementation"}

    with tempfile.TemporaryDirectory() as tmpdir:
        impl_file = os.path.join(tmpdir, "solution.py")
        with open(impl_file, "w") as f:
            f.write(impl)

        test_file = os.path.join(tmpdir, "test_solution.py")
        sig = task["signature"]
        test_code = "import sys\nsys.path.insert(0, '" + tmpdir + "')\nfrom solution import *\n\n"
        for t in task["tests"]:
            test_code += t + "\n\n"

        with open(test_file, "w") as f:
            f.write(test_code)

        result = subprocess.run(
            ["python3", "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=30,
            cwd=tmpdir
        )

        passed = result.returncode == 0
        return {
            "task": task_id,
            "difficulty": "hard",
            "status": "PASS" if passed else "FAIL",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }


def validate_extreme_task(task_path: Path) -> dict:
    with open(task_path) as f:
        task = json.load(f)

    task_id = task["id"]
    files = EXTREME_IMPLEMENTATIONS.get(task_id)
    if not files:
        return {"task": task_id, "status": "SKIP", "message": "No reference implementation"}

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in files.items():
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(content)

        test_file = os.path.join(tmpdir, "test_task.py")
        test_code = f"import sys\nsys.path.insert(0, '{tmpdir}')\n"

        for imp in task.get("test_imports", []):
            test_code += imp + "\n"
        test_code += "\n"

        for t in task["tests"]:
            test_code += t + "\n\n"

        with open(test_file, "w") as f:
            f.write(test_code)

        result = subprocess.run(
            ["python3", "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=30,
            cwd=tmpdir
        )

        passed = result.returncode == 0
        return {
            "task": task_id,
            "difficulty": "extreme",
            "status": "PASS" if passed else "FAIL",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }


def main():
    results = []

    hard_dir = Path("/Users/lukashm/Desktop/llm-code-debate/tasks/hard")
    for task_file in sorted(hard_dir.glob("*.json")):
        print(f"Validating hard/{task_file.name}...")
        try:
            r = validate_hard_task(task_file)
            results.append(r)
            print(f"  -> {r['status']}")
        except Exception as e:
            results.append({"task": task_file.stem, "status": "ERROR", "message": str(e)})
            print(f"  -> ERROR: {e}")

    extreme_dir = Path("/Users/lukashm/Desktop/llm-code-debate/tasks/extreme")
    for task_file in sorted(extreme_dir.glob("*.json")):
        print(f"Validating extreme/{task_file.name}...")
        try:
            r = validate_extreme_task(task_file)
            results.append(r)
            print(f"  -> {r['status']}")
        except Exception as e:
            results.append({"task": task_file.stem, "status": "ERROR", "message": str(e)})
            print(f"  -> ERROR: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]
    skipped = [r for r in results if r["status"] == "SKIP"]
    errors = [r for r in results if r["status"] == "ERROR"]

    print(f"Total: {len(results)}, PASS: {len(passed)}, FAIL: {len(failed)}, SKIP: {len(skipped)}, ERROR: {len(errors)}")

    if failed:
        print("\n--- FAILURES ---")
        for r in failed:
            print(f"\n{'='*60}")
            print(f"FAILED: {r['difficulty']}/{r['task']}")
            print(f"{'='*60}")
            print(r.get("stdout", ""))
            if r.get("stderr"):
                print("STDERR:", r["stderr"][:500])

    if errors:
        print("\n--- ERRORS ---")
        for r in errors:
            print(f"ERROR: {r['task']}: {r.get('message', 'unknown')}")

    return len(failed) + len(errors)


if __name__ == "__main__":
    sys.exit(main())
