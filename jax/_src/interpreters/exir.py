# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""APIs for lowering jaxprs to EXIR.

See https://pytorch.org/executorch/stable/ir-exir.html#aten-dialect.
"""
import dataclasses
import string
from collections.abc import Sequence
from functools import partial
from typing import Any, Optional, Callable

import torch
from executorch import exir
from torch import export
from torch import fx
from torch.utils import dlpack as torch_dlpack

import jax
import jax.numpy as jnp
from jax._src import ad_util, dtypes
from jax._src import core
from jax._src import dlpack as jax_dlpack
from jax._src import prng
from jax._src.lax import lax
from jax._src.typing import ArrayLike, DTypeLike


@dataclasses.dataclass(frozen=True, slots=True)
class Context:
  in_avals: Sequence[core.ShapedArray]
  out_avals: Sequence[core.ShapedArray]


torch_impl = {}


def register_torch_impl(prim: core.Primitive) -> Callable[[Any], None]:
  return partial(torch_impl.__setitem__, prim)


def forward_to_torch(prim: core.Primitive, torch_fn: Callable[..., Any]) -> None:
  torch_impl[prim] = lambda ctx, *in_vals, **params: torch_fn(
      *in_vals,
      **params,
  )


forward_to_torch(lax.add_p, torch.add)
forward_to_torch(lax.sub_p, torch.sub)
forward_to_torch(lax.mul_p, torch.mul)
forward_to_torch(lax.div_p, torch.div)

forward_to_torch(lax.sqrt_p, torch.sqrt)
forward_to_torch(lax.log_p, torch.log)
forward_to_torch(lax.log1p_p, torch.log1p)
forward_to_torch(lax.exp_p, torch.exp)
forward_to_torch(lax.expm1_p, torch.expm1)


@register_torch_impl(lax.reduce_max_p)
def _reduce_max_torch_impl(ctx: Context, in_val: torch.Tensor, axes):
  return torch.amax(in_val, dim=axes)


@register_torch_impl(lax.reduce_sum_p)
def _reduce_max_torch_impl(ctx: Context, in_val: torch.Tensor, axes):
  return torch.sum(in_val, dim=axes)


@register_torch_impl(lax.transpose_p)
def _transpose_torch_impl(ctx: Context, in_val: torch.Tensor, *, permutation):
  return torch.permute(in_val, permutation)


@register_torch_impl(lax.reshape_p)
def _reshape_torch_impl(
    ctx: Context,
    in_val: torch.Tensor,
    *,
    new_sizes,
    dimensions,
) -> torch.Tensor:
  if dimensions is not None:
    in_val = torch.permute(in_val, dimensions)
  return torch.reshape(in_val, new_sizes)


@register_torch_impl(lax.broadcast_in_dim_p)
def _broadcast_in_dim_torch_impl(
    ctx: Context,
    in_val: torch.Tensor,
    *,
    shape,
    broadcast_dimensions,
) -> torch.Tensor:
  with_1s_shape = [1] * len(shape)
  for i, dim in enumerate(broadcast_dimensions):
    with_1s_shape[dim] = ctx.in_avals[0].shape[i]
  with_1s = torch.reshape(in_val, with_1s_shape)
  # ExecuTorch does not support 0-strided tensors.
  return torch.broadcast_to(with_1s, shape).contiguous()


@register_torch_impl(lax.dot_general_p)
def _dot_general_torch_impl(
    ctx: Context,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    dimension_numbers: lax.DotDimensionNumbers,
    precision: lax.Precision,
    preferred_element_type: Optional[DTypeLike],
) -> torch.Tensor:
  # This implementation trick was borrowed from jax2tf.
  del precision, preferred_element_type  # Unused.
  lhs_aval, rhs_aval = ctx.in_avals
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers

  next_id = iter(string.ascii_letters).__next__
  lhs_axis_ids = [next_id() for _ in lhs_aval.shape]
  rhs_axis_ids = [next_id() for _ in rhs_aval.shape]
  lhs_out_axis_ids = lhs_axis_ids[:]
  rhs_out_axis_ids = rhs_axis_ids[:]

  for l, r in zip(lhs_contracting, rhs_contracting):
    contracted_id = next_id()
    lhs_axis_ids[l] = rhs_axis_ids[r] = contracted_id
    lhs_out_axis_ids[l] = rhs_out_axis_ids[r] = None

  batch_ids = []
  for l, r in zip(lhs_batch, rhs_batch):
    batch_id = next_id()
    lhs_axis_ids[l] = rhs_axis_ids[r] = batch_id
    lhs_out_axis_ids[l] = rhs_out_axis_ids[r] = None
    batch_ids.append(batch_id)

  out_axis_ids = [
      d for d in batch_ids + lhs_out_axis_ids + rhs_out_axis_ids
      if d is not None
  ]

  spec = "{},{}->{}".format("".join(lhs_axis_ids),
                            "".join(rhs_axis_ids),
                            "".join(out_axis_ids))
  return torch.einsum(spec, lhs, rhs)


@register_torch_impl(prng.random_wrap_p)
def _random_wrap_torch_impl(ctx: Context, in_val, *, impl):
  del g, impl  # Unused.
  return in_val


@register_torch_impl(prng.random_unwrap_p)
def _random_wrap_torch_impl(ctx: Context, in_val):
  del g  # Unused.
  return in_val


@register_torch_impl(prng.random_fold_in_p)
def _random_fold_in_torch_impl(
    ctx: Context,
    keys: torch.Tensor,
    msgs: torch.Tensor,
) -> torch.Tensor:
  # TODO(slebedev): How to call into prng.random_fold_in_impl_base here?
  return keys


@register_torch_impl(ad_util.stop_gradient_p)
def _stop_gradient_torch_impl(ctx: Context, in_val: torch.Tensor):
  return in_val.detach()


jax_to_torch_dtype = {
    jnp.dtype("float32"): torch.float32,
    jnp.dtype("int32"): torch.int32,
    jnp.dtype("int64"): torch.int64,
}


@register_torch_impl(lax.convert_element_type_p)
def _convert_element_type_torch_impl(ctx: Context, in_val, new_dtype, weak_type):
  assert not weak_type
  return in_val.to(jax_to_torch_dtype[new_dtype])


def to_torch_tensor(x: ArrayLike) -> torch.Tensor:
  arr = jnp.asarray(x)
  if dtypes.issubdtype(arr.dtype, dtypes.prng_key):
    # TODO(slebedev): Find a way to support PRNG keys.
    return torch.tensor(0xDEADC0DE)
  elif not arr.shape:
    # Let PyTorch choose the dtype for scalars.
    # See https://github.com/pytorch/pytorch/issues/58734.
    return torch.as_tensor(arr.item())
  t = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(arr))
  # ExecuTorch does not support 0-strided tensors.
  return t.contiguous()


@dataclasses.dataclass
class Env:
  data: dict[core.Var, Any]

  def read(self, v: core.Atom) -> Any:
    if type(v) is core.Literal:
      return to_torch_tensor(v.val)
    assert isinstance(v, core.Var)
    return self.data[v]

  def write(self, v: core.Var, val: Any) -> None:
    self.data[v] = val


def interpret_closed_jaxpr(
    closed_jaxpr: core.ClosedJaxpr,
) -> Callable[..., Sequence[Any]]:
  jaxpr = closed_jaxpr.jaxpr
  consts = closed_jaxpr.consts

  def inner(*args: Any):
    # torch.fx.Proxy does not support iteration, so we need to recreate
    # the args via manual indexing.
    args = [args[i] for i in range(len(jaxpr.invars))]

    env = Env({})
    for v, val in zip(jaxpr.invars, args):
      env.write(v, val)
    for v, val in zip(jaxpr.constvars, consts):
      env.write(v, to_torch_tensor(val))

    for eqn in jaxpr.eqns:
      ctx = Context(
          in_avals=[v.aval for v in eqn.invars],
          out_avals=[v.aval for v in eqn.outvars],
      )
      out = torch_impl[eqn.primitive](
          ctx,
          *(env.read(var) for var in eqn.invars),
          **eqn.params,
      )
      if not eqn.primitive.multiple_results:
        out = [out]
      for v, val in zip(eqn.outvars, out):
        env.write(v, val)

    return [env.read(v) for v in jaxpr.outvars]

  return inner


def closed_jaxpr_to_exir(
    closed_jaxpr: core.ClosedJaxpr,
    *args: jax.Array,
) -> exir.ExecutorchProgramManager:
  gm = torch.fx.symbolic_trace(interpret_closed_jaxpr(closed_jaxpr))
  torch_args = tuple(to_torch_tensor(jnp.ones_like(arg)) for arg in args)
  aten_program = export.export(gm, torch_args)
  edge_program = exir.to_edge(aten_program)
  executorch_program = edge_program.to_executorch()
  return executorch_program


if __name__ == "__main__":
  from flax import linen as nn


  def f(x, y):
    z = jax.nn.softmax(x) + y
    return z @ z.T


  def g(q, rng):
    attn = nn.SelfAttention(
        num_heads=8,
        qkv_features=16,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros,
        deterministic=True,
    )
    y, _ = attn.init_with_output(rng, q)
    return y


  # x = jnp.ones([4, 2])
  # print(f(x, x))
  # closed_jaxpr = jax.make_jaxpr(f)(x, x)
  # p = closed_jaxpr_to_exir(closed_jaxpr, x, x)

  q = jnp.ones((4, 2, 3, 5))
  rng = jax.random.key(1)
  print(g(q, rng))
  closed_jaxpr = jax.make_jaxpr(g)(q, rng)
  p = closed_jaxpr_to_exir(closed_jaxpr, q, rng)

  with open("/tmp/model.pte", "wb") as file:
    file.write(p.buffer)
  print(p)
