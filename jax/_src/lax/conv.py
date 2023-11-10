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
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.lib.mlir.dialects import hlo
from jax._src.typing import Array, ArrayLike, DTypeLike, DType


@dataclass(frozen=True, unsafe_hash=True)
class ConvDimensionNumbers:
  in_batch: Sequence[int]
  in_spatial: Sequence[int]
  in_feature: int
  kernel_batch: Sequence[int]
  kernel_in_feature: int
  kernel_out_feature: int
  kernel_spatial: Sequence[int]
  out_batch: Sequence[int]
  out_feature: int
  out_spatial: Sequence[int]

  def normalized(self) -> "ConvDimensionNumbers":
    def drop_dim(dim, bs):
      return int(dim - np.greater_equal(dim, bs).sum())

    def drop_dims(dims, bs):
      return np.subtract(
          dims,
          np.greater_equal(dims, np.array(bs)[:, None]).sum(axis=0),
      )

    return ConvDimensionNumbers(
        in_batch=tuple(drop_dims(self.in_batch[:1], self.in_batch[1:])),
        in_feature=drop_dim(self.in_feature, self.in_batch[1:]),
        in_spatial=tuple(drop_dims(self.in_spatial, self.in_batch[1:])),
        kernel_batch=(),
        kernel_in_feature=drop_dim(self.kernel_in_feature, self.kernel_batch),
        kernel_out_feature=drop_dim(self.kernel_out_feature, self.kernel_batch),
        kernel_spatial=tuple(drop_dims(self.kernel_spatial, self.kernel_batch)),
        out_batch=tuple(drop_dims(self.out_batch[:1], self.out_batch[1:])),
        out_feature=drop_dim(self.out_feature, self.out_batch[1:]),
        out_spatial=tuple(drop_dims(self.out_spatial, self.out_batch[1:])),
    )


def conv_dimension_numbers(
    in_shape: core.Shape,
    kernel_shape: core.Shape,
) -> ConvDimensionNumbers:
  ...


_Padding = str | Sequence[tuple[int, int]]


def conv_general(
    input: ArrayLike,
    kernel: ArrayLike,
    *,
    window_strides: Sequence[int],
    in_dilation: Sequence[int] | None = None,
    kernel_dilation: Sequence[int] | None = None,
    padding: _Padding,
    dimension_numbers: ConvDimensionNumbers | None = None,
    batch_group_count: int = 1,
    feature_group_count: int = 1,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
):
  dnums = dimension_numbers or conv_dimension_numbers(
      input.shape,
      kernel.shape,
  )
  if in_dilation is None:
    in_dilation = (1,) * len(dnums.in_spatial)
  if kernel_dilation is None:
    kernel_dilation = (1,) * len(dnums.kernel_spatial)
  if isinstance(padding, str):
    raise NotImplementedError
  if preferred_element_type is not None:
    preferred_element_type = dtypes.canonicalize_dtype(
        np.dtype(preferred_element_type)
    )
  return conv_general_p.bind(
      input,
      kernel,
      window_strides=tuple(window_strides),
      in_dilation=tuple(in_dilation),
      kernel_dilation=tuple(kernel_dilation),
      padding=tuple(padding),
      dimension_numbers=dnums,
      batch_group_count=batch_group_count,
      feature_group_count=feature_group_count,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )


def _conv_general_shape_rule(
    in_aval: core.ShapedArray,
    kernel_aval: core.ShapedArray,
    *,
    window_strides: Sequence[int],
    in_dilation: Sequence[int],
    kernel_dilation: Sequence[int],
    padding: _Padding,
    dimension_numbers: ConvDimensionNumbers,
    batch_group_count: int,
    **params,
):
  del params  # Unused.

  # TODO(slebedev): Validate the parameters.
  dnums = dimension_numbers

  in_dilated = np.array(in_aval.shape)
  in_dilated[list(dnums.in_spatial)] = list(map(
      core.dilate_dim,
      np.take(in_aval.shape, dnums.in_spatial),
      in_dilation
  ))

  kernel_dilated = np.array(kernel_aval.shape)
  kernel_dilated[list(dnums.kernel_spatial)] = list(map(
      core.dilate_dim,
      np.take(kernel_aval.shape, dnums.kernel_spatial),
      kernel_dilation,
  ))

  if isinstance(padding, str):
    padding = lax.padtype_to_pads(
        np.take(in_dilated, dnums.in_spatial),
        np.take(kernel_dilated, dnums.kernel_spatial),
        window_strides,
        padding,
    )

  in_padded = np.copy(in_dilated)
  in_padded[list(dnums.in_spatial)] = np.add(
      np.take(in_dilated, dnums.in_spatial),
      np.sum(np.asarray(padding).reshape(-1, 2), axis=1),
  )

  out_shape = (
      *np.take(in_aval.shape, dnums.out_batch) // batch_group_count,
      kernel_aval.shape[dnums.kernel_out_feature],
      *map(
          core.stride_dim,
          np.take(in_padded, dnums.in_spatial),
          np.take(kernel_dilated, dnums.kernel_spatial),
          window_strides,
      ))
  return tuple(np.take(
      out_shape,
      np.argsort([*dnums.out_batch, dnums.out_feature, *dnums.out_spatial]),
  ))


def _conv_general_dtype_rule(
    in_aval: core.ShapedArray,
    kernel_aval: core.ShapedArray,
    *,
    preferred_element_type: DTypeLike | None,
    **params: Any,
) -> DType:
  del params  # Unused.
  out_dtype = lax.naryop_dtype_rule(
      lax._input_dtype,
      [lax._any, lax._any],
      "conv_general",
      in_aval,
      kernel_aval,
  )
  if preferred_element_type is None:
    return out_dtype
  lax._validate_preferred_element_type(out_dtype, preferred_element_type)
  return preferred_element_type


conv_general_p = lax.standard_primitive(
    _conv_general_shape_rule, _conv_general_dtype_rule, "conv_general")


def _conv_general_batch_rule(
    batched_args: Sequence[core.ShapedArray],
    batch_dims: Sequence[int],
    *,
    window_strides: Sequence[int],
    padding: _Padding,
    in_dilation: Sequence[int],
    kernel_dilation: Sequence[int],
    dimension_numbers: ConvDimensionNumbers,
    batch_group_count: int,
    feature_group_count: int,
    precision: lax.PrecisionLike,
    preferred_element_type: DTypeLike | None,
) -> tuple[Array, int]:
  in_aval, kernel_aval = batched_args
  new_dnums, out_bdim = _conv_general_batch_dimension_numbers(
      *batch_dims,
      dimension_numbers=dimension_numbers,
  )
  batched_out = conv_general(
      *batched_args,
      window_strides=window_strides,
      padding=padding,
      in_dilation=in_dilation,
      kernel_dilation=kernel_dilation,
      dimension_numbers=new_dnums,
      batch_group_count=batch_group_count,
      feature_group_count=feature_group_count,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )
  out_shape = _conv_general_shape_rule(
      in_aval,
      kernel_aval,
      window_strides=window_strides,
      padding=padding,
      in_dilation=in_dilation,
      kernel_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      batch_group_count=batch_group_count,
      feature_group_count=feature_group_count,
  )
  return batched_out, batching.shape_as_bdim(out_bdim, out_shape)


def _conv_general_batch_dimension_numbers(
    in_bdim: int | None,
    kernel_bdim: int | None,
    *,
    dimension_numbers: ConvDimensionNumbers,
) -> tuple[ConvDimensionNumbers, int]:
  dnums = dimension_numbers

  def bump_dim(dim, b):
    return dim + int(dim >= b)

  def bump_dims(dims, b):
    return np.add(dims, np.greater_equal(dims, b))

  if in_bdim is not None and kernel_bdim is not None:
    out_bdim = len(dnums.out_batch)
    new_dnums = ConvDimensionNumbers(
        in_batch=(*bump_dims(dnums.in_batch, in_bdim), in_bdim),
        in_feature=bump_dim(dnums.in_feature, in_bdim),
        in_spatial=tuple(bump_dims(dnums.in_spatial, in_bdim)),
        kernel_batch=(*bump_dims(dnums.kernel_batch, kernel_bdim), kernel_bdim),
        kernel_in_feature=bump_dim(dnums.kernel_in_feature, kernel_bdim),
        kernel_out_feature=bump_dim(dnums.kernel_out_feature, kernel_bdim),
        kernel_spatial=tuple(bump_dims(dnums.kernel_spatial, kernel_bdim)),
        out_batch=(*bump_dims(dnums.out_batch, out_bdim), out_bdim),
        out_feature=bump_dim(dnums.out_feature, out_bdim),
        out_spatial=tuple(bump_dims(dnums.out_spatial, out_bdim)),
    )
    return new_dnums, 0
  elif in_bdim is not None:
    ...
  elif kernel_bdim is not None:
    ...
  else:
    assert False  # Impossible.


batching.primitive_batchers[conv_general_p] = _conv_general_batch_rule


def _reshape_axes_into(srcs: Sequence[int], dst: int, val: mlir.Value, aval: core.ShapedArray):
  perm = [i for i in range(aval.ndim) if i not in srcs]
  for idx, src in enumerate(srcs):
    perm.insert(dst + idx, src)
  new_shape = list(np.delete(aval.shape, srcs))
  for src in srcs:
    new_shape[dst - int(dst > src)] *= aval.shape[src]
  new_val = hlo.TransposeOp(val, mlir.dense_int_elements(perm))
  new_val = hlo.ReshapeOp(
      mlir.aval_to_ir_type(core.ShapedArray(new_shape, aval.dtype)),
      new_val,
  )
  return new_val, new_shape


def _reshape_axes_out_of(src: int, dst: int, sizes: Sequence[int], val: mlir.Value, aval: core.ShapedArray):
  new_shape = list(aval.shape)
  last_size = new_shape[src]
  for size in sizes:
    last_size, ragged = divmod(last_size, size)
    assert not ragged
  new_shape[src] = last_size
  new_shape[dst:dst] = sizes
  return hlo.ReshapeOp(
      mlir.aval_to_ir_type(core.ShapedArray(new_shape, aval.dtype)),
      val,
  )


def _conv_general_lower(
    ctx: mlir.LoweringRuleContext,
    in_val: mlir.Value,
    kernel_val: mlir.Value,
    *,
    window_strides: Sequence[int],
    in_dilation: Sequence[int],
    kernel_dilation: Sequence[int],
    padding: _Padding,
    dimension_numbers: ConvDimensionNumbers,
    batch_group_count: int,
    feature_group_count: int,
    precision: lax.PrecisionLike,
    preferred_element_type: DTypeLike,
):
  in_aval, kernel_aval = ctx.avals_in
  [out_aval] = ctx.avals_out
  dnums = dimension_numbers

  if len(dnums.in_batch) > 1 and dnums.kernel_batch:
    assert len(dnums.in_batch) == len(dnums.kernel_batch) + 1
    assert (
        [in_aval.shape[dim] for dim in dnums.in_batch[1:]] ==
        [kernel_aval.shape[dim] for dim in dnums.kernel_batch]
    )
    if batch_group_count > 1:
      new_in_val, new_in_shape = _reshape_axes_into(
          dnums.in_batch[1:], dnums.in_batch[0], in_val, in_aval)
      for dim in dnums.in_batch[1:]:
        batch_group_count *= in_aval.shape[dim]
    else:
      new_in_val, new_in_shape = _reshape_axes_into(
          dnums.in_batch[1:], dnums.in_feature, in_val, in_aval)
      for dim in dnums.in_batch[1:]:
        feature_group_count *= in_aval.shape[dim]
    new_kernel_val, new_kernel_shape = _reshape_axes_into(
        dnums.kernel_batch,
        dnums.kernel_out_feature,
        kernel_val,
        kernel_aval
    )
    new_in_aval = core.ShapedArray(new_in_shape, in_aval.dtype)
    new_kernel_aval = core.ShapedArray(new_kernel_shape, kernel_aval.dtype)
    new_dnums = dnums.normalized()
    new_out_shape = _conv_general_shape_rule(
        new_in_aval,
        new_kernel_aval,
        window_strides=window_strides,
        in_dilation=in_dilation,
        kernel_dilation=kernel_dilation,
        padding=padding,
        dimension_numbers=new_dnums,
        batch_group_count=batch_group_count,
    )
    new_out_aval = core.ShapedArray(new_out_shape, out_aval.dtype)
    new_ctx = dataclasses.replace(
        ctx,
        avals_in=[new_in_aval, new_kernel_aval],
        avals_out=[new_out_aval],
    )
    [new_out_val] = _conv_general_lower(
        new_ctx,
        new_in_val,
        new_kernel_val,
        window_strides=window_strides,
        in_dilation=in_dilation,
        kernel_dilation=kernel_dilation,
        padding=padding,
        dimension_numbers=new_dnums,
        batch_group_count=batch_group_count,
        feature_group_count=feature_group_count,
        precision=precision,
        preferred_element_type=preferred_element_type
    )
    out_val = _reshape_axes_out_of(
        new_dnums.out_feature,
        new_dnums.out_batch[0],
        [in_aval.shape[dim] for dim in dnums.in_batch[1:]],
        new_out_val,
        new_out_aval
    )
    return [out_val.result]
  elif len(dnums.in_batch) > 1:
    raise NotImplementedError
  elif dnums.kernel_batch:
    raise NotImplementedError

  hlo_dnums = hlo.ConvDimensionNumbers.get(
      input_batch_dimension=dnums.in_batch[0],
      input_feature_dimension=dnums.in_feature,
      input_spatial_dimensions=dnums.in_spatial,
      kernel_input_feature_dimension=dnums.kernel_in_feature,
      kernel_output_feature_dimension=dnums.kernel_out_feature,
      kernel_spatial_dimensions=dnums.kernel_spatial,
      output_batch_dimension=dnums.out_batch[0],
      output_feature_dimension=dnums.out_feature,
      output_spatial_dimensions=dnums.out_spatial,
  )

  if len(padding) == 0:
    padding = np.zeros((0, 2), dtype=np.int64)
  else:
    assert all(core.is_constant_shape(p) for p in padding)

  window_reversal = mlir.dense_bool_elements([False] * len(dnums.in_spatial))
  return [
      hlo.ConvolutionOp(
          mlir.aval_to_ir_type(out_aval),
          in_val,
          kernel_val,
          dimension_numbers=hlo_dnums,
          feature_group_count=mlir.i64_attr(feature_group_count),
          batch_group_count=mlir.i64_attr(batch_group_count),
          window_strides=mlir.dense_int_elements(window_strides),
          padding=mlir.dense_int_elements(padding),
          lhs_dilation=mlir.dense_int_elements(in_dilation),
          rhs_dilation=mlir.dense_int_elements(kernel_dilation),
          window_reversal=window_reversal,
          precision_config=lax.precision_attr(precision),
      ).result
  ]


mlir.register_lowering(conv_general_p, _conv_general_lower, platform="cpu")


def main():
  in_shape = [1, 1, 4, 4]
  kernel_shape = [1, 1, 3, 3]
  kernel_dilation = [1, 1]
  window_strides = [4, 4]

  dnums = ConvDimensionNumbers(
      in_batch=(0,),
      in_feature=1,
      in_spatial=(2, 3),
      kernel_batch=(),
      kernel_in_feature=0,
      kernel_out_feature=1,
      kernel_spatial=(2, 3),
      out_batch=(0,),
      out_feature=1,
      out_spatial=(2, 3),
  )

  import jax
  from jax._src.lax import convolution as legacy_conv

  legacy_dnums = legacy_conv.ConvDimensionNumbers(
      (0, 1, 2, 3),
      (1, 0, 2, 3),
      (0, 1, 2, 3),
  )

  for padding, in_dilation in [
      ([(0, 0), (0, 0)], [2, 2]),
      ("VALID", [1, 1]),
      ("SAME", [1, 1]),
      ("SAME_LOWER", [1, 1]),
  ]:
    fn = partial(
        conv_general,
        window_strides=window_strides,
        in_dilation=in_dilation,
        kernel_dilation=kernel_dilation,
        padding=padding,
        dimension_numbers=dnums,
    )
    legacy_fn = partial(
        legacy_conv.conv_general_dilated,
        window_strides=window_strides,
        lhs_dilation=in_dilation,
        rhs_dilation=kernel_dilation,
        padding=padding,
        dimension_numbers=legacy_dnums,
    )
    jax.vmap(fn)(
        np.ones([2] + in_shape),
        np.ones([2] + kernel_shape),
    )  # want: 2x1x1x2x2xf32
    expected = legacy_conv.conv_general_dilated(
        np.ones(in_shape),
        np.ones(kernel_shape),
        window_strides=window_strides,
        lhs_dilation=in_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=legacy_dnums,
        padding=padding,
    )
    out_shape = _conv_general_shape_rule(
        np.ones(in_shape),
        np.ones(kernel_shape),
        window_strides=window_strides,
        in_dilation=in_dilation,
        kernel_dilation=kernel_dilation,
        padding=padding,
        dimension_numbers=dnums,
        batch_group_count=1,
        feature_group_count=1,
    )
    print(padding, out_shape, expected.shape)

    print(jax.jit(jax.vmap(fn)).lower(np.ones([2] + in_shape), np.ones([2] + kernel_shape)).as_text())


if __name__ == "__main__":
  main()
