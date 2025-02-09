---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Custom methods

As the purpose of pyscfad is to provide a framework for developing new methods that are automatically differentiable,
it also offers several useful functionalities to simplify such development.
This is based on the powerful AD tools like JAX.
Typically, developing a new method involves the following steps.

## Class definition

We assume the new method is defined in a custom class.
JAX function transformations are applied to functions that operate over
[pytrees](https://jax.readthedocs.io/en/latest/pytrees.html).
Although not required, it may be convenient to convert the class into a pytree,
so that the class instance can be passed to the functions being transformed.
This conversion can be achieved by subclassing the `PytreeNode` class.

```{code-cell}
from pyscfad import numpy as np
from pyscfad.pytree import PytreeNode

class PowerSum(PytreeNode):
    _dynamic_attr = {'array'}

    def __init__(self, array, order=2):
        self.array = array
        self.order = order

    def kernel(self):
        return np.sum(self.array**self.order)
```

In the example above, we define a class whose `kernel` function performs the calculation of
element-wise power then summation for the input array.
Note the class attribute `_dynamic_attr` in the definition,
which labels the names of dynamic attributes of the class.
These attributes are considered as the leaves of the pytree,
which are traced variables in the computational graph.
Whereas the other attributes of the object are static, which means
that they are kept as constants during the computation.

## Function transformation

With the class registered as a pytree,
it is possible to apply function transformations
to the functions that take the class instance as the input.

```{code-cell}
import jax

a = PowerSum(np.eye(2), order=4)
grad = jax.jit(jax.grad(PowerSum.kernel))(a)
print(grad.array)
```

Here, both [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) 
and [`jax.grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) 
can be applied to the `kernel` function,
taking `a` (an instance of `PowerSum`) as the input.
Note that the static attribute `order` must be kept unmodified within the function
being transformed, i.e., `kernel` in this example. Otherwise, unpredicted behavior may occur.


## Subclassing

A subclass of `PytreeNode` can be further subclassed.
In addition, only the newly added dynamic attributes need to be registered.
For example, to subclass `PowerSum`, and to add a new dynamic variable `array1`,
we can simply do the following.

```{code-cell}
class MultiplyPowerSum(PowerSum):
    _dynamic_attr = {'array1'}

    def __init__(self, array, array1, order=2):
        super().__init__(array, order=order)
        self.array1 = array1

    def kernel(self):
        return np.sum((self.array*self.array1)**self.order)
```

Now, both `array` and `array1` will be correctly traced.

```{code-cell}
a = MultiplyPowerSum(np.eye(2), np.eye(2)*2, order=4)
grad = jax.jit(jax.grad(MultiplyPowerSum.kernel))(a)
print(grad.array)
print(grad.array1)
```

