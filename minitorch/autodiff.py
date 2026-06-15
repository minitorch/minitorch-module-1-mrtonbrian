from cgitb import lookup
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from pandas.core.indexing import is_label_like
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_lower = [x if i != arg else x - epsilon / 2. for i, x in enumerate[Any](vals)]
    vals_upper = [x if i != arg else x + epsilon / 2. for i, x in enumerate[Any](vals)]

    return (f(*vals_upper) - f(*vals_lower)) / float(epsilon)

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Kahn's
    indegrees = defaultdict[int, int](int)
    var_lookup: dict[int, Variable] = {}

    print("rightmost id:", variable.unique_id)
    stack = [variable]

    seen = set[int]()
    while len(stack) > 0:
        top = stack.pop()
        var_lookup[top.unique_id] = top

        if top.unique_id in seen:
            continue

        seen.add(top.unique_id)

        if top.unique_id not in indegrees:
            indegrees[top.unique_id] = 0

        if top.is_constant():
            continue

        for parent in top.parents:
            stack.append(parent)
            indegrees[parent.unique_id] += 1

    working_set = [var_lookup[variable] for variable, deg in indegrees.items() if deg == 0]
    while len(working_set) > 0:
        top = working_set.pop()

        if top.is_constant():
            continue

        yield top

        for parent in top.parents:
            indegrees[parent.unique_id] -= 1
            if indegrees[parent.unique_id] == 0:
                working_set.append(parent)

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    deriv_vals = defaultdict[int, float](float)

    deriv_vals[variable.unique_id] = deriv

    for var in topological_sort(variable):    
        print(var, var.history, var.is_constant(), var.is_leaf(), deriv_vals)
        if var.is_leaf():
            continue

        derivatives = var.chain_rule(deriv_vals[var.unique_id])

        for par_variable, d_val in derivatives:
            if par_variable.is_leaf():
                par_variable.accumulate_derivative(d_val)
            else:
                deriv_vals[par_variable.unique_id] += d_val

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
