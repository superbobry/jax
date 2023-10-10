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
import contextlib
import functools
import logging
import math
import re
from typing import Optional, Sequence
import unittest

from absl.testing import absltest
import numpy as np

import jax
from jax import numpy as jnp
from jax import tree_util
from jax.config import config
from jax.experimental.export import export

from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.lib import version as jaxlib_version
from jax._src.lib.mlir.dialects import hlo

config.parse_flags_with_absl()

# A primitive for testing multi-platform lowering. Takes one argument and
# adds a different value to it: cpu=2., tpu=3., cuda=.4, rocm=5.
_testing_multi_platform_p = core.Primitive("testing_multi_platform")
_testing_multi_platform_to_add = dict(cpu=2., tpu=3., cuda=4., rocm=5.)

@_testing_multi_platform_p.def_abstract_eval
def _testing_multi_platform_abstract_eval(xaval: core.AbstractValue):
  assert xaval.dtype == np.float32  # type: ignore
  return xaval

def _testing_multi_platform_lowering(ctx: mlir.LoweringRuleContext,
                                     x: mlir.Value,
                                     *,
                                     platform: str) -> Sequence[mlir.Value]:
  to_add = _testing_multi_platform_to_add[platform]
  to_add_value = mlir.broadcast_in_dim(ctx,
                                       mlir.ir_constant(np.float32(to_add)),
                                       ctx.avals_in[0],
                                       broadcast_dimensions=())
  return mlir.hlo.AddOp(x, to_add_value).results

# Register a default rule for cuda, to test the default-platform rule selection.
mlir.register_lowering(_testing_multi_platform_p,
                       functools.partial(_testing_multi_platform_lowering,
                                         platform="cuda"))
for platform in ["cpu", "tpu", "rocm"]:
  mlir.register_lowering(_testing_multi_platform_p,
                         functools.partial(_testing_multi_platform_lowering,
                                           platform=platform),
                         platform=platform)

def _testing_multi_platform_func(x):
  return _testing_multi_platform_p.bind(x)

def _testing_multi_platform_fun_expected(x):
  return x + _testing_multi_platform_to_add[xb.canonicalize_platform(jtu.device_under_test())]


class JaxExportTest(jtu.JaxTestCase):

  def override_serialization_version(self, version_override: int):
      version = config.jax_serialization_version
      if version != version_override:
        self.addCleanup(functools.partial(config.update,
                                          "jax_serialization_version",
                                          version_override))
        config.update("jax_serialization_version", version_override)
      logging.info(
        "Using JAX serialization version %s",
        config.jax_serialization_version)

  def setUp(self):
    super().setUp()
    # Run tests with the maximum supported version by default
    self.override_serialization_version(
      export.maximum_supported_serialization_version)

  def test_basic_export_only(self):
    def my_fun(x):
      return jnp.sin(x)
    exp = export.export(my_fun)(jax.ShapeDtypeStruct((4,), dtype=np.float32))
    self.assertEqual("my_fun", exp.fun_name)
    self.assertEqual(export.default_lowering_platform(), exp.lowering_platform)
    self.assertEqual(tree_util.tree_flatten(((1,), {}))[1], exp.in_tree)
    self.assertEqual((core.ShapedArray((4,), dtype=np.float32),), exp.in_avals)
    self.assertEqual((core.ShapedArray((4,), dtype=np.float32),), exp.out_avals)

  def test_pytree_export_only(self):
    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    def f(a_b_pair, *, a, b):
      return (dict(res=a_b_pair, a=a, b=b), jnp.sin(a), jnp.cos(b))

    exp = export.export(f, lowering_platform="cpu")((a, b), a=a, b=b)
    a_aval = core.ShapedArray(a.shape, a.dtype)
    b_aval = core.ShapedArray(b.shape, b.dtype)
    self.assertEqual(exp.lowering_platform, "cpu")
    args = ((a, b),)
    kwargs = dict(a=a, b=b)
    self.assertEqual(exp.in_tree, tree_util.tree_flatten((args, kwargs))[1])
    self.assertEqual(exp.in_avals, (a_aval, b_aval, a_aval, b_aval))
    self.assertEqual(exp.out_tree, tree_util.tree_flatten(f(*args, **kwargs))[1])
    self.assertEqual(exp.out_avals, (a_aval, b_aval, a_aval, b_aval, a_aval, b_aval))

  def test_poly_export_only(self):
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f(a, b):  # a: f32[2w,h]  b: f32[w,h]
      return jnp.concatenate([a, b], axis=0)

    exp = export.export(f)(
        export.poly_spec(a.shape, a.dtype, "(2*w, h)"),
        export.poly_spec(a.shape, a.dtype, "(w, h)"))
    self.assertEqual("(2*w, h)", str(exp.in_avals[0].shape))
    self.assertEqual("(w, h)", str(exp.in_avals[1].shape))
    self.assertEqual("(3*w, h)", str(exp.out_avals[0].shape))

  def test_poly_pytree_export_only(self):
    a = np.arange(12, dtype=np.float32).reshape((3, 4))
    def f(a0, a1, *, ak):
      return jnp.concatenate([a0, a1, ak], axis=0)

    a_poly_spec = export.poly_spec(a.shape, a.dtype, "(w, h)")
    exp = export.export(f)(a_poly_spec, a_poly_spec, ak=a_poly_spec)
    self.assertEqual("(w, h)", str(exp.in_avals[0].shape))
    self.assertEqual("(3*w, h)", str(exp.out_avals[0].shape))

  def test_basic(self):
    f = jnp.sin
    x = np.arange(4, dtype=np.float32)
    exp_f = export.export(f)(x)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(f(x), f1(x))

  def test_call_exported_lambda(self):
    # When we export a lambda, the exported.fun_name is not a valid MLIR function name
    f = lambda x: jnp.sin(x)
    x = np.arange(4, dtype=np.float32)
    exp_f = export.export(f)(x)
    f1 = export.call_exported(exp_f)
    self.assertAllClose(f(x), f1(x))

  def test_call_twice_exported(self):
    def f(x): return jnp.sin(x)
    x = np.arange(4, dtype=np.float32)

    @jax.jit
    def f1(x):
      exp_f = export.export(f)(x)
      return export.call_exported(exp_f)(x) + export.call_exported(exp_f)(x)

    self.assertAllClose(2. * f(x), f1(x))

  def test_unused_args(self):
    f = lambda x, y: jnp.sin(x)
    x = np.arange(4, dtype=np.float32)
    y = np.arange(6, dtype=np.float32)
    exp_f = export.export(f)(x, y)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(f(x, y), f1(x, y))

  def test_pytree(self):
    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    def f(a_b_pair, a, b):
      return (dict(res=a_b_pair, a=a, b=b), jnp.sin(a), jnp.cos(b))

    exp_f = export.export(f)((a, b), a=a, b=b)
    f1 = export.call_exported(exp_f)
    self.assertAllClose(f((a, b), a=a, b=b),
                        f1((a, b), a=a, b=b))

  def test_error_wrong_intree(self):
    def f(a_b_pair, *, c):
      return jnp.sin(a_b_pair[0]) + jnp.cos(a_b_pair[1]) + c
    a = b = c = np.arange(4, dtype=np.float32)
    exp_f = export.export(f)((a, b), c=c)

    with self.assertRaisesRegex(
        ValueError,
        "The invocation args and kwargs must have the same pytree structure"):
      export.call_exported(exp_f)(a, b, c=(a, b))

  def test_error_wrong_avals(self):
    def f(a, *, b):  # a: f32[4] and b: f32[4]
      return jnp.sin(a) + jnp.cos(b)
    f32_4 = np.arange(4, dtype=np.float32)
    exp_f = export.export(f)(f32_4, b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Shape mismatch for args\[0\].shape\[0\]"):
      export.call_exported(exp_f)(np.arange(6, dtype=np.float32), b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Shape mismatch for kwargs\['b'\].shape\[0\]"):
      export.call_exported(exp_f)(f32_4, b=np.arange(6, dtype=np.float32))

    with self.assertRaisesRegex(ValueError,
        r"Rank mismatch for args\[0\]"):
      export.call_exported(exp_f)(f32_4.reshape((1, 4)), b=f32_4)

    with self.assertRaisesRegex(ValueError,
        r"Dtype mismatch for args\[0\]"):
      export.call_exported(exp_f)(f32_4.astype(np.float16), b=f32_4)

  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["platform"],
    kwargs=[dict(platform=p)
            for p in ("cpu", "cuda", "rocm", "tpu")])
  def test_error_wrong_platform(self, platform):
    a = np.arange(4, dtype=np.float32)

    exp_f = export.export(jnp.sin, lowering_platform=platform)(a)
    if xb.canonicalize_platform(jtu.device_under_test()) == platform:
      raise unittest.SkipTest("Uninteresting scenario")

    with self.assertRaisesRegex(
        ValueError, "The exported function .* was lowered for platform"):
      export.call_exported(exp_f)(a)

    # Now try with the platform check disabled
    exp_f_no_platform_check = export.export(
      jnp.sin, lowering_platform=platform,
      disabled_checks=[export.DisabledSafetyCheck.platform()])(a)
    res = export.call_exported(exp_f_no_platform_check)(a)
    self.assertAllClose(res, jnp.sin(a))

  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["dialect"],
    kwargs=[dict(dialect=dialect)
            for dialect in ("mhlo", "stablehlo")]
  )
  def test_error_disallowed_custom_call(self, dialect):
    # If we use hlo.custom_call or mhlo.custom_call we detect
    # invalid custom call targets.
    # Set up a primitive with custom lowering rules
    test_primitive = core.Primitive("_test_primitive_disallowed_custom_call")
    test_primitive.def_abstract_eval(lambda in_aval: in_aval)
    def test_primitive_lowering(ctx, arg):
      from jax._src.lib.mlir.dialects import mhlo
      op = dict(stablehlo=hlo.CustomCallOp, mhlo=mhlo.CustomCallOp)[dialect]
      return op([arg.type], [arg], "disallowed_call_target").results
    mlir.register_lowering(test_primitive, test_primitive_lowering)
    self.addCleanup(lambda: mlir.register_lowering(test_primitive, None))

    a = np.arange(3, dtype=np.float32)
    with self.assertRaisesRegex(ValueError,
        "Cannot serialize code with custom calls whose targets .*"):
      export.export(
        lambda a: a + test_primitive.bind(a)
      )(a)

    # Now try again with the safety check disabled
    exp = export.export(
      lambda a: a + test_primitive.bind(a),
      disabled_checks=[export.DisabledSafetyCheck.custom_call("disallowed_call_target")]
    )(a)
    self.assertIn("disallowed_call_target", exp.mlir_module())

  def test_grad(self):
    f = lambda x: jnp.sum(jnp.sin(x))
    x = np.arange(4, dtype=np.float32)
    exp_f = export.export(f)(x)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(jax.grad(f)(x), jax.grad(f1)(x))

  def test_higher_order_grad(self):
    f = lambda x: x ** 3
    x = np.float32(4.)
    exp_f = export.export(f)(x)

    f1 = export.call_exported(exp_f)
    self.assertAllClose(jax.grad(f)(x),
                        jax.grad(f1)(x))
    self.assertAllClose(jax.grad(jax.grad(f))(x),
                        jax.grad(jax.grad(f1))(x))
    self.assertAllClose(jax.grad(jax.grad(jax.grad(f)))(x),
                        jax.grad(jax.grad(jax.grad(f1)))(x))

  def test_pytree_vjp(self):
    def f(a_b_pair, *, a, b):
      return (dict(res=a_b_pair, a=2. * a, b=3. * b),
              jnp.sin(4. * a))

    a = np.arange(4, dtype=np.float32)
    b = np.arange(6, dtype=np.float32)
    exp_f = export.export(f)((a, b), a=a, b=b)

    out_ct = f((a, b), a=a, b=b)  # The output has the right structure as the cotangent
    def f1_jax(a, b):  # For VJP, make a function without kwargs
      res = f((a, b), a=a, b=b)
      return res
    def f1_exp(a, b):  # For VJP, make a function without kwargs
      res = export.call_exported(exp_f)((a, b), a=a, b=b)
      return res
    jax_vjp = jax.vjp(f1_jax, a, b)[1](out_ct)
    exp_vjp = jax.vjp(f1_exp, a, b)[1](out_ct)
    self.assertAllClose(jax_vjp, exp_vjp)

  def test_roundtrip(self):
    def f1(x):
      return jnp.sin(x)
    a = np.arange(4, dtype=np.float32)
    exp_f1 = export.export(f1)(a)
    def f2(x):
      res1 = export.call_exported(exp_f1)(x)
      res2 = export.call_exported(exp_f1)(res1)
      return jnp.cos(res2)
    exp_f2 = export.export(f2)(a)

    self.assertAllClose(jnp.cos(jnp.sin(jnp.sin(a))),
                        export.call_exported(exp_f2)(a))

  @jtu.parameterized_filterable(
    kwargs=[
      dict(v=v)
      for v in range(export.minimum_supported_serialization_version - 1,
                     export.maximum_supported_serialization_version + 2)])
  def test_shape_poly_basic_versions(self, v: int):
    self.override_serialization_version(v)
    with contextlib.ExitStack() as e:
      if not (export.minimum_supported_serialization_version <= v
              <= export.maximum_supported_serialization_version):
        e.enter_context(self.assertRaisesRegex(
          ValueError,
          f"The requested jax_serialization version {v} is outside the range of supported versions"))

      exp = export.export(jnp.sin)(
        export.poly_spec((3, 4), np.float32, "w, h"))
      # Peek at the module
      module_str = exp.mlir_module()
      self.assertEqual(config.jax_serialization_version >= 7,
                       "shape_assertion" in module_str)
      self.assertIn("jax.uses_shape_polymorphism = true",
                    module_str)
      dim_vars = re.findall(
        r"(%arg\d):\s*tensor<i..>\s*{jax.dimension_variable = true}",
        module_str)
      self.assertEqual(["%arg0", "%arg1"], dim_vars,
                       f"Found {dim_vars} in {module_str}")
      x = np.arange(30, dtype=np.float32).reshape((5, 6))
      res = export.call_exported(exp)(x)
      self.assertAllClose(res, np.sin(x))

  # A function is exported with f32[poly_spec] and is called with different arg
  # shapes. We use export.call_exported and we also run the shape check
  # module.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw:f"poly_spec={kw['poly_spec']}_arg_shape={kw['arg_shape']}",  # type: ignore
    kwargs=[
      dict(poly_spec="3,4,12", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,4,12", arg_shape=(3, 4, 13),
           # The shape check module does not test constant dimensions
           expect_error=re.escape(
               r"Shape mismatch for args[0].shape[2] (expected same constant)")),
      dict(poly_spec="3,4,6*a", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,a,a+8", arg_shape=(3, 4, 12)),
      dict(poly_spec="3,4,a+1", arg_shape=(3, 4, 1),
           expect_error=re.escape(
               "Expected value >= 1 for dimension variable 'a'. "
               "Using the following polymorphic shapes specifications: args[0].shape = (3, 4, a + 1). "
               "Obtained dimension variables: 'a' = 0"
          )),
      dict(poly_spec="3,4,6*a", arg_shape=(3, 4, 13),
           expect_error=re.escape(
              "Division had remainder 1 when computing the value of 'a'"
          )),
      dict(poly_spec="3,a,a+8", arg_shape=(3, 4, 13),
           expect_error=re.escape(
             "Found inconsistency between dimension size "
             "args[0].shape[2] (= 13) and the specification 'a + 8' (= 12)"
          )),
  ])
  def test_poly_shape_checks(
      self, poly_spec="3,a,a+8",
      arg_shape=(3, 4, 12), arg_dtype=np.float32,
      expect_error=None):  # If given, error from running the exported module

    def f(x):  # x: f32[poly_spec]
      return jnp.reshape(x, (-1, x.shape[1]))

    disabled_checks = ()
    exp_f = export.export(f, disabled_checks=disabled_checks)(
        export.poly_spec((3, 4, 12), np.float32, poly_spec))
    self.assertEqual(exp_f.uses_shape_polymorphism, poly_spec != "3,4,12")
    arg = np.arange(np.prod(arg_shape),
                    dtype=arg_dtype).reshape(arg_shape)  # arg : f32[3,4,12]

    with contextlib.ExitStack() as stack:
      if expect_error is not None:
        stack.push(self.assertRaisesRegex(Exception, expect_error))

      assert core.is_constant_shape(arg.shape)
      res = export.call_exported(exp_f)(arg)

    if not expect_error:
      self.assertAllClose(res, f(arg))

  # An inner function is exported with polymorphic shapes inner_poly_spec, and
  # is called from an outer function, which is exported with outer_poly_spec.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw:f"inner={kw['inner_poly_spec']}_outer={kw['outer_poly_spec']}",  # type: ignore
    #one_containing="",
    # By default arg_shape = (3, 4, 12) for both the outer function and the inner
    # The inner function is exported for f32.
    kwargs=[
      # Both inner and outer are static shapes
      dict(inner_poly_spec="3,4,12", outer_poly_spec="3,4,12"),
      # Inner has poly shapes but outer has static shapes. When we call inner
      # we do the shape constraint checking
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,12"),
      dict(inner_poly_spec="3,4,3*a", outer_poly_spec="3,4,12"),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
             "Found inconsistency between dimension size "
             "args[0].shape[2] (= 12) and the specification 'a' (= 4)")),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
             "Division had remainder 2 when computing the value of 'a'")),
      dict(inner_poly_spec="3,4,12+a", outer_poly_spec="3,4,12",
           expect_error_outer_exp=re.escape(
              "Expected value >= 1 for dimension variable 'a'. "
              "Using the following polymorphic shapes specifications: args[0].shape = (3, 4, a + 12). "
              "Obtained dimension variables: 'a' = 0 from specification "
              "'a + 12' for dimension args[0].shape[2] (= 12)")),
      # Both inner and outer have poly shapes.
      dict(inner_poly_spec="3,a,b", outer_poly_spec="3,4,c"),
      dict(inner_poly_spec="3,4,3*a", outer_poly_spec="3,4,6*c"),
      dict(inner_poly_spec="3,a,a+8", outer_poly_spec="3,c+2,c+10"),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
             "Expected value >= 1 for dimension variable 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (3, a, a + b). "
             "Obtained dimension variables: 'a' = 4 from specification "
             "'a' for dimension args[0].shape[1] (= 4), "
             "'b' = c + -4 from specification 'a + b' for dimension args[0].shape[2] (= c),")),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
             "Found inconsistency between dimension size "
             "args[0].shape[2] (= c) and the specification 'a' (= 4)")),
      dict(inner_poly_spec="3,a,a", arg_shape=(3, 4),
           outer_poly_spec="3,c",
           expect_error_outer_exp=r"Rank mismatch for args\[0\]"),
      dict(inner_poly_spec="3,a,a+b", arg_dtype=np.int32,
           outer_poly_spec="3,c,d",
           expect_error_outer_exp=r"Dtype mismatch for args\[0\]"),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,c",
           expect_error_outer_exp=re.escape(
              "Division had remainder mod(c, 5) when computing the value of 'a'")),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="3,c,c",
           expect_error_outer_exp=re.escape(
               "Expected value >= 1 for dimension variable 'b'. "
               "Using the following polymorphic shapes specifications: args[0].shape = (3, a, a + b). "
               "Obtained dimension variables: 'a' = c from "
               "specification 'a' for dimension args[0].shape[1] (= c), "
               "'b' = 0 from specification 'a + b' for dimension args[0].shape[2] (= c)")),
      dict(inner_poly_spec="3,a,a+b", outer_poly_spec="c,4,12",
           expect_error_outer_exp=re.escape(
               "Shape mismatch for args[0].shape[0] (expected same constant)")),
      dict(inner_poly_spec="3,4,5*a", outer_poly_spec="3,4,25*c",
           expect_error_run=re.escape(
              "Division had remainder 12 when computing the value of 'c'")),
      dict(inner_poly_spec="3,a,b", outer_poly_spec="3,c+4,12",
           expect_error_run=re.escape(
               "Expected value >= 1 for dimension variable 'c'. "
               "Using the following polymorphic shapes specifications: args[0].shape = (3, c + 4, 12). "
               "Obtained dimension variables: 'c' = 0")),
      dict(inner_poly_spec="3,a,a", outer_poly_spec="3,a,a",
           expect_error_run=re.escape(
               "Found inconsistency between dimension size "
               "args[0].shape[2] (= 12) and the specification 'a' (= 4)")),
  ])
  def test_poly_shape_checks_nested(
      self, inner_poly_spec="3,4,5*a",
      arg_shape=(3, 4, 12), arg_dtype=np.float32,
      outer_poly_spec="3,4,25*c",
      expect_error_outer_exp=None,
      expect_error_run=None):
    # Polymorphic export called with static or polymorphic shapes
    def inner(x):  # x: inner_poly_spec
      return jnp.reshape(x, (-1, x.shape[1]))

    arg = np.arange(np.prod(arg_shape),
                    dtype=arg_dtype).reshape(arg_shape)  # x : f32[3,4,12]
    inner_exp = export.export(inner)(
        export.poly_spec((3, 4, 12), np.float32, inner_poly_spec))

    self.assertEqual(inner_exp.uses_shape_polymorphism,
                     (inner_poly_spec != "3,4,12"))
    def outer(x):  # x: outer_poly_spec
      # Use an addition to test that the shapes are refined properly for the
      # result of the call_exported.
      return export.call_exported(inner_exp)(x) + inner(x)

    with contextlib.ExitStack() as stack:
      if expect_error_outer_exp is not None:
        stack.push(self.assertRaisesRegex(ValueError, expect_error_outer_exp))

      # Call it after exporting again, with polymorphic shapes
      outer_exp = export.export(outer)(
          export.poly_spec(arg.shape, arg.dtype, outer_poly_spec))

    if expect_error_outer_exp is not None:
      return

    self.assertEqual(outer_exp.uses_shape_polymorphism,
                     (inner_poly_spec != "3,4,12" or outer_poly_spec != "3,4,12"))

    with contextlib.ExitStack() as stack:
      if expect_error_run is not None:
        stack.push(self.assertRaisesRegex(Exception, expect_error_run))

      res = export.call_exported(outer_exp)(arg)

    if expect_error_run is not None:
      return
    self.assertAllClose(2. * inner(arg), res)

  # Tests details of the shape constraints errors
  # This test exists also in shape_poly_test.py. Here we test the
  # call_exported error reporting.
  @jtu.parameterized_filterable(
    testcase_name=lambda kw: kw["shape"],  # assume "shape" is unique
    kwargs=[
      dict(shape=(8, 2, 9),  # a = 2, b = 3, c = 4
           poly_spec="(a + 2*b, a, a + b + c)"),
      dict(shape=(2, 2, 6),  # a = 2, b = 0, c = 4
           poly_spec="(a + 2*b, a, a + b + c)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Expected value >= 1 for dimension variable 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (a + 2*b, a, a + b + c). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), "
             "'b' = 0 from specification 'a + 2*b' for dimension args[0].shape[0] (= 2), . "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details."
           )),
      dict(shape=(3, 2, 6),  # a = 2, b = 0.5, c = 4 - b is not integer
           poly_spec="(a + 2*b, a, a + b + c)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Division had remainder 1 when computing the value of 'b'. "
             "Using the following polymorphic shapes specifications: args[0].shape = (a + 2*b, a, a + b + c). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), . "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details."
           )),
      dict(shape=(8, 2, 6),  # a = 2, b = 3 - inconsistency
           poly_spec="(a + 2*b, a, a + b)",
           expect_error=(
             "Input shapes do not match the polymorphic shapes specification. "
             "Found inconsistency between dimension size args[0].shape[0] (= 8) and the specification 'a + 2*b' (= 10). "
             "Using the following polymorphic shapes specifications: args[0].shape = (a + 2*b, a, a + b). "
             "Obtained dimension variables: 'a' = 2 from specification 'a' for dimension args[0].shape[1] (= 2), "
             "'b' = 4 from specification 'a + b' for dimension args[0].shape[2] (= 6), . "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-assertion-errors for more details."
           )),
      dict(shape=(7, 2, 36),  # a = 2, b = 3, c = 6 - cannot solve c
           poly_spec="(2 * a + b, a, c * c)",
           expect_error=(
             "Cannot solve for values of dimension variables {'c'}. "
             "We can only solve linear uni-variate constraints. "
             "Using the following polymorphic shapes specifications: args[0].shape = (2*a + b, a, c^2). "
             "Unprocessed specifications: 'c^2' for dimension size args[0].shape[2]. "
             "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#dimension-variables-must-be-solvable-from-the-input-shapes for more details."
           )),
  ])
  def test_shape_constraints_errors(self, *,
      shape, poly_spec: str, expect_error: Optional[str] = None):
    def f_jax(x):  # x: f32[a + 2*b, a, a + b + c]
      return 0.

    if shape == (8, 2, 6) and jaxlib_version <= (0, 4, 14):
      raise unittest.SkipTest("Test requires jaxlib >= 0.4.14")
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    with contextlib.ExitStack() as stack:
      if expect_error is not None:
        stack.push(self.assertRaisesRegex(Exception, re.escape(expect_error)))
      exp = export.export(f_jax)(
          export.poly_spec(x.shape, x.dtype, poly_spec))
      export.call_exported(exp)(x)

  def test_multi_platform(self):
    if jtu.test_device_matches(["gpu"]):
      # The export is not applicable to GPU
      raise unittest.SkipTest("Not intended for running on GPU")
    x = np.arange(5, dtype=np.float32)
    exp = export.export(_testing_multi_platform_func,
                        lowering_platforms=('cpu', 'tpu'))(x)
    self.assertEqual(exp.lowering_platforms, ('cpu', 'tpu'))
    module_str = str(exp.mlir_module())
    expected_main_re = (
      r"@main\("
      r"%arg0: tensor<i..> {jax.platform_index = true}.*, "
      r"%arg1: tensor<5xf32>.* ->")
    self.assertRegex(module_str, expected_main_re)

    self.assertIn("jax.uses_shape_polymorphism = true",
                  module_str)

    res = export.call_exported(exp)(x)
    self.assertAllClose(res, _testing_multi_platform_fun_expected(x))

  def test_multi_platform_nested(self):
    if jtu.test_device_matches(["tpu"]):
      # The outer export is not applicable to TPU
      raise unittest.SkipTest("Not intended for running on TPU")
    x = np.arange(5, dtype=np.float32)
    exp = export.export(_testing_multi_platform_func,
                        lowering_platforms=('cpu', 'tpu', 'cuda'))(x)
    self.assertEqual(exp.lowering_platforms, ('cpu', 'tpu', 'cuda'))

    # Now serialize the call to the exported using a different sequence of
    # lowering platforms, but included in the lowering platforms for the
    # nested exported.
    exp2 = export.export(export.call_exported(exp),
                         lowering_platforms=('cpu', 'cuda'))(x)
    res2 = export.call_exported(exp2)(x)
    self.assertAllClose(res2, _testing_multi_platform_fun_expected(x))

  def test_multi_platform_nested_inside_single_platform_export(self):
    x = np.arange(5, dtype=np.float32)
    exp = export.export(_testing_multi_platform_func,
                        lowering_platforms=('cpu', 'tpu', 'cuda'))(x)
    self.assertEqual(exp.lowering_platforms, ('cpu', 'tpu', 'cuda'))

    # Now serialize the call for the current platform.
    exp2 = export.export(export.call_exported(exp))(x)
    module_str = str(exp2.mlir_module())
    self.assertIn("jax.uses_shape_polymorphism = true",
                  module_str)
    res2 = export.call_exported(exp2)(x)
    self.assertAllClose(res2, _testing_multi_platform_fun_expected(x))

  def test_multi_platform_and_poly(self):
    if jtu.test_device_matches(["gpu"]):
      # The export is not applicable to GPU
      raise unittest.SkipTest("Not intended for running on GPU")
    exp = export.export(lambda x: jnp.reshape(_testing_multi_platform_func(x), (-1,)),
                        lowering_platforms=('cpu', 'tpu'))(
        export.poly_spec((5, 6), np.float32, "b1, b2")
    )
    x = np.arange(12, dtype=np.float32).reshape((3, 4))
    res = export.call_exported(exp)(x)
    self.assertAllClose(res, _testing_multi_platform_fun_expected(x).reshape((-1,)))
    # Now serialize the call to the exported
    exp2 = export.export(export.call_exported(exp))(x)
    res2 = export.call_exported(exp2)(x)
    self.assertAllClose(res2, _testing_multi_platform_fun_expected(x).reshape((-1,)))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
