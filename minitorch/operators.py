"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List, TypeVar, Optional

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(a: float, b: float) -> float:
    return a * b

def id(x: float) -> float:
    return x

def add(a: float, b: float) -> float:
    return a + b

def neg(x: float) -> float: 
    return -x

def lt(a: float, b: float) -> float:
    return 1. if a < b else 0.

def eq(a: float, b: float) -> float:
    return 1. if a == b else 0.

def max(a: float, b: float) -> float:
    return a if a > b else b

def is_close(a: float, b: float) -> float:
    return 1. if (abs(a - b) < 0.01) else 0

def sigmoid(x: float) -> float:
    return 1./(1.0 + math.exp(-x))

def relu(x: float) -> float:
    return float(x) if x >= 0 else 0.

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1./x

def log_back(x: float, g: float) -> float:
    return inv(x) * g

def inv_back(x: float, g: float) -> float:
    return -1 / (x ** 2) * g

def relu_back(x: float, g: float) -> float:
    return g * (1 if x > 0 else 0)

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

# TODO: Implement for Task 0.3.
def map(fn: Callable[[T], U]) -> Callable[[Iterable[T]], List[U]]:
    def apply(ls: Iterable[T]) -> List[U]:
        return [fn(l) for l in ls]
    
    return apply
    
def zipWith(fn: Callable[[T, T], U]) -> Callable[[Iterable[T], Iterable[T]], List[U]]:
    def apply(a: Iterable[T], b: Iterable[T]) -> List[U]:
        out = []
        a_iter = iter(a)
        b_iter = iter(b)

        while True:
            try:
                x = next(a_iter)
                y = next(b_iter)
            except StopIteration:
                return out

            out.append(fn(x, y))
    return apply


def reduce(fn: Callable[[T, T], T]) -> Callable[[Iterable[T]], Optional[T]]:
    def apply(l: Iterable[T]) -> Optional[T]:
        prev_val = None
        a_iter = iter(l)
        while True:
            if prev_val is None:
                first = None
                try:
                    first = next(a_iter)
                    second = next(a_iter)
                except StopIteration:
                    return first

                prev_val = fn(first, second)
            else:
                try:
                    v = next(a_iter)
                    prev_val = fn(v, prev_val)
                except StopIteration:
                    return prev_val
    return apply

def negList(l: List[float]) -> List[float]:
    return map(neg)(l)

def addLists(a: List[float], b: List[float]) -> List[float]:
    return zipWith(add)(a, b)

def sum(l: List[float]) -> float:
    o = reduce(add)(l)
    return o if o is not None else 0


def prod(l: List[float]) -> float:
    o = reduce(mul)(l)
    return o if o is not None else 1
