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

from collections.abc import Generator, Iterable, Iterator, Sequence
from functools import partial
import itertools as it
import math
import operator as op
import os
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
from jax import config
from jax import lax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax._src import core
from jax._src import linear_util as lu
from jax._src import test_util as jtu
from jax._src import tree_util
from jax._src import xla_bridge
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.util import merge_lists, partition_list, safe_map, safe_zip

config.parse_flags_with_absl()

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# Helper for some tests.
def create_inputs(a_sharding, b_sharding):
  x, y, z = 2, 2, 2  # pylint: disable=invalid-name
  devices = np.array(jax.devices()[:x * y * z]).reshape((x, y, z))
  mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
  b, e, f = 8, 8, 8  # pylint: disable=invalid-name
  m1 = jax.device_put(
      jnp.arange(b * e).reshape((b, e)),
      jax.sharding.NamedSharding(mesh, a_sharding))
  m2 = jax.device_put(
      jnp.arange(e * f).reshape((e, f)),
      jax.sharding.NamedSharding(mesh, b_sharding))
  return mesh, m1, m2

# Run all tests with 8 CPU devices.
prev_xla_flags = None

# Run all tests with 8 CPU devices.
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()

  if len(jax.devices()) < 8:
    raise unittest.SkipTest("tests require 8 devices")

# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class ShardMapTest(jtu.JaxTestCase):

  def test_identity(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    def identity(x):
      return x

    @jax.jit
    def fwd(a):
      c = shard_map(
          lambda x: x,
          mesh,
          in_specs=(P('z', ('x', 'y')),),
          out_specs=P('z', ('x', 'y')))(a)
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 2))

  def test_all_gather(self):
    mesh, a, _ = create_inputs(P('z', ('x', 'y')), P(None, None))
    assert a.device_buffers[0].shape == (4, 2)

    # NOTE(mattjj): to use out_specs=P(None, ('x', 'y')), we need to use
    # all_gather_invariant primitive, which differs in its output replication
    # type compared to all_gather.
    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', ('x', 'y')),), out_specs=P('z', ('x', 'y')))
    def fwd(a):
      return lax.all_gather(a, 'z', axis=0, tiled=True)

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (8, 2))

  def test_matmul_partial(self):
    raise unittest.SkipTest("invalid replication asserted by out_spec?")

    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)), out_specs=P('z', None))
    def fwd(a):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return c

    c = fwd(a)
    self.assertEqual(c.device_buffers[0].shape, (4, 8))

  def test_matmul_reduce_scatter(self):
    mesh, a, b = create_inputs(P('z', 'y'), P('y', None))
    assert a.device_buffers[0].shape == (4, 4)

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('z', 'y'), P('y', None)),
             out_specs=P(('z', 'y'), None))
    def fwd(a, b):
      c = jnp.matmul(a, b)  # [B.z, F] {y.unreduced}
      return lax.psum_scatter(c, 'y', scatter_dimension=0, tiled=True)

    c = fwd(a, b)
    self.assertEqual(c.device_buffers[0].shape, (2, 8))

  def test_collective_permute(self):
    devices = np.array(jax.devices()[:8]) # Take up to 8 devices
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P('x', None),),
             out_specs=P('x', None))
    def fwd(a):
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(a, 'x', perm=perm)

    c = fwd(a)
    self.assertAllClose(c[1, :], a[0, :])

  def test_all_to_all(self):
    devices = np.array(jax.devices()[:8]) # Take up to 8 devices
    mesh = Mesh(devices, axis_names=('x'))
    a = jax.device_put(
        jnp.arange(8 * 8).reshape((8, 8)),
        jax.sharding.NamedSharding(mesh, P('x', None)))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P(None, 'x'))
    def fwd(a):
      return lax.all_to_all(a, 'x', split_axis=1, concat_axis=1, tiled=True)

    c = fwd(a)
    assert (c == jnp.reshape(a.T, (1, 64))).all()

  def test_eager_repr(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    s = None

    @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'))
    def f(x):
      nonlocal s
      s = str(x)
      return x
    _ = f(np.arange(8 * 8.).reshape(8, 8))

    self.assertIsInstance(s, str)
    self.assertIn('at mesh coordinates', s)

  def test_jvp_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    args = np.arange(4 * 4.).reshape(4, 4),
    jtu.check_grads(g, args, 2, ['fwd'])
    jtu.check_grads(jax.jit(g), args, 2, ['fwd'])

  def test_linearize_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
    x = np.arange(4 * 4.).reshape(4, 4)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jax.lax.sin(jax.lax.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_linearize_basic_repres_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    g = shard_map(lambda x: jnp.sin(jnp.cos(x)), mesh,
                  in_specs=(P('x',),), out_specs=P('x',))
    x = np.arange(4.)

    y, y_dot = jax.jvp(g, [x], [x])

    y_, g_lin = jax.linearize(g, x)
    y_dot_ = g_lin(x)

    self.assertAllClose(y, y_, check_dtypes=False)
    self.assertAllClose(y_dot, y_dot_, check_dtypes=False)

  def test_replication_checker_eager(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      g(x)

    def f2(x):
      return jax.lax.psum(x, 'x')
    def g2(x):
      return shard_map(f2, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = g2(x)  # doesn't crash

  def test_replication_checker_jit(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = np.arange(8 * 8.).reshape(8, 8)

    def f(x):
      return 2 * x
    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)

    with self.assertRaisesRegex(ValueError, 'statically inferred'):
      jax.jit(g)(x)

    def f2(x):
      return jax.lax.psum(x, 'x')
    def g2(x):
      return shard_map(f2, mesh, in_specs=(P('x', 'y'),), out_specs=P(None, 'y'))(x)
    _ = jax.jit(g2)(x)  # doesn't crash

  def test_process_env_traces(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))
    x = np.arange(8.)

    def g(x):
      y = (3. * x).sum()
      z = shard_map(lambda x: 2 * x * y, mesh,
                    in_specs=(P('x'),), out_specs=P('x'))(np.arange(8.))
      return z

    jtu.check_grads(g, (x,), modes=['fwd'], order=2)

  def test_eager_control_flow(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(2 * 2.).reshape(2, 2)

    def f(x):
      y = jax.lax.psum(x, ('x', 'y'))
      if y < 0:
        return x
      else:
        return -x

    def g(x):
      return shard_map(f, mesh, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))(x)
    y = g(x)
    self.assertAllClose(y, -x, check_dtypes=False)

  def test_outer_jit_detects_shard_map_mesh(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    f = shard_map(lambda x: x.reshape(1, *x.shape), mesh, P(), P('x'))
    _ = jax.jit(f)(jnp.array(2.0))  # doesn't crash

  def test_vmap_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(8 * 8.).reshape(8, 8)

    def g(x):
      return shard_map(lambda x: 2. * x, mesh,
                       in_specs=P('y'), out_specs=P('y'))(x)
    y = jax.vmap(g)(x)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

  def test_vmap_basic_axis_name(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(8 * 8.).reshape(8, 8)

    def g(x):
      return shard_map(lambda x: 2. * x, mesh,
                       in_specs=P('y'), out_specs=P('y'))(x)
    y = jax.vmap(g, axis_name='i')(x)
    self.assertAllClose(y, 2 * x, check_dtypes=False)

  def test_vmap_basic_axis_name_reuse_mesh_name(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    x = jnp.arange(8 * 8.).reshape(8, 8)

    def g(x):
      return shard_map(lambda x: 2. * x, mesh,
                       in_specs=P('y'), out_specs=P('y'))(x)
    y = jax.vmap(g, axis_name='x')(x)  # NOTE reuse same 'x' as on mesh
    self.assertAllClose(y, 2 * x, check_dtypes=False)

  def test_tree_prefix_error(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @partial(shard_map, mesh=mesh, in_specs=([P('x', 'y')],), out_specs=P('x', 'y'))
    def f(x):
      return x

    x = jnp.arange(8 * 8.).reshape(8, 8)
    with self.assertRaisesRegex(ValueError, r'shard_map in_specs\[0\]'):
      f([x, x])

  def test_rank_errors(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    def foo():
      return {'hi': [3.]}

    with self.assertRaisesRegex(ValueError, 'which has length 1'):
      shard_map(foo, mesh=mesh, in_specs=(), out_specs={'hi': P('x')})()

    with self.assertRaisesRegex(ValueError, 'which has length 1'):
      jax.jit(lambda: shard_map(foo, mesh=mesh,
                                in_specs=(), out_specs={'hi': P('x')})())()

    with self.assertRaisesRegex(ValueError, 'which has rank 0'):
      shard_map(foo, mesh=mesh, in_specs=({'hi': P('x')},), out_specs=())(
          {'hi': [jnp.array(3.)]})

    with self.assertRaisesRegex(ValueError,
                                r'consider using an in_specs entry of `P\(\)`'):
      shard_map(foo, mesh=mesh, in_specs=P(None), out_specs=())(3.)

  def test_reverse_mode_ad(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x',), P(None)), out_specs=P('x',))
    def f(x, y):
      return jnp.sin(x) + 3 + jnp.tan(2.) * jnp.cos(x) + y

    x = jnp.arange(8.) / 10.
    y = jnp.arange(4.) / 10.
    jtu.check_grads(f, (x, y), modes=['fwd', 'rev'], order=2)

  def test_post_process(self):
    # JVPTrace.post_process_shard_map and JaxprTrace.post_process_shard_map
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    def f(x):
      @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
      def g(y):
        return jnp.sin(y) * jnp.sin(x).sum()
      return g(jnp.arange(8.))

    x = jnp.arange(8.)
    _, f_lin = jax.linearize(f, x)
    y_dot = f_lin(x)

    y_dot_expected = jnp.sin(jnp.arange(8.)) * (jnp.cos(x) * x).sum()
    self.assertAllClose(y_dot, y_dot_expected, check_dtypes=False)

  @jtu.run_on_devices('gpu', 'tpu')
  def test_axis_index(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P('x'))
    def f():
      return jax.lax.axis_index('x')[None]

    x = f()
    self.assertAllClose(x, jnp.arange(4), check_dtypes=False)

  def test_remat_basic(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    # check param updating is handled
    @jax.remat
    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f(x):
      return jnp.sin(x)

    x = jnp.arange(4.)
    g = jax.grad(lambda x: f(x).sum())(x)  # doesn't crash
    self.assertAllClose(g, jnp.cos(x), check_dtypes=False)

    # also check residuals are handled correctly
    @partial(jax.remat, policy=jax.checkpoint_policies.everything_saveable)
    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f2(x):
      return jnp.sin(x)

    g2 = jax.grad(lambda x: f2(x).sum())(x)  # doesn't crash
    self.assertAllClose(g2, jnp.cos(x), check_dtypes=False)

  def test_remat_scalar_residuals(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    @partial(jax.remat, policy=jax.checkpoint_policies.everything_saveable)
    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f(x):
      return jnp.sin(jnp.sin(jnp.sin(x.sum()))[None])

    x = jnp.arange(8.)
    _ = jax.grad(lambda x: f(x).sum())(x)  # doesn't crash
    jtu.check_grads(f, (x,), modes=['rev'], order=2, atol=1e-2, rtol=1e-2)

  def test_check_rep_false_doesnt_hit_rep_rules(self):
    mesh = Mesh(np.array(jax.devices()[:4]), ('x',))

    prim = core.Primitive('prim')  # no rep rule here!
    prim.multiple_results = True
    prim.def_impl(lambda: [])
    prim.def_abstract_eval(lambda: [])

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=None, check_rep=True)
    def f():
      prim.bind()

    with self.assertRaises(NotImplementedError):
      f()
    with self.assertRaises(NotImplementedError):
      jax.jit(f)()

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=None, check_rep=False)
    def f2():
      prim.bind()

    f2()
    jax.jit(f2)()

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=None, check_rep=False)
    def f3():
      jax.jit(prim.bind)()

    f3()
    jax.jit(f3)()

  def test_vmap_spmd_axis_name(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f(x):
      return x

    x = jnp.arange(4 * 4).reshape(4, 4)
    jaxpr = jax.make_jaxpr(jax.vmap(f, spmd_axis_name='y'))(x).jaxpr
    e, = jaxpr.eqns
    self.assertIn('in_names', e.params)
    self.assertEqual(e.params['in_names'], ({0: ('y',), 1: ('x',)},))
    self.assertIn('out_names', e.params)
    self.assertEqual(e.params['out_names'], ({0: ('y',), 1: ('x',)},))

  def test_vmap_spmd_axis_name_pair(self):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))

    @partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P())
    def f(x):
      return x

    x = jnp.arange(4 * 4).reshape(4, 4)
    jaxpr = jax.make_jaxpr(jax.vmap(f, spmd_axis_name=('x', 'y')))(x).jaxpr
    e, = jaxpr.eqns
    self.assertIn('in_names', e.params)
    self.assertEqual(e.params['in_names'], ({0: ('x', 'y',)},))
    self.assertIn('out_names', e.params)
    self.assertEqual(e.params['out_names'], ({0: ('x', 'y',)},))

  @jtu.run_on_devices('cpu', 'gpu', 'tpu')
  def test_debug_print_jit(self):
    mesh = Mesh(jax.devices(), ('i',))

    @jax.jit  # NOTE: axis_index requires jit (at time of writing)
    @partial(shard_map, mesh=mesh, in_specs=P('i'), out_specs=P('i'))
    def f(x):
      idx = jax.lax.axis_index('i')
      jax.debug.print("instance {i} has value x={x}", i=idx, x=x)
      y = jnp.cos(x)
      jax.debug.print("instance {i} has value y={y}", i=idx, y=y)
      return y

    x = jnp.arange(2 * len(jax.devices()))

    with jtu.capture_stdout() as output:
      f(x)
      jax.effects_barrier()
    for i in range(len(jax.devices())):
      self.assertIn(f'instance {i} has value', output())

  def test_debug_print_eager(self):
    mesh = Mesh(jax.devices(), ('i',))

    @partial(shard_map, mesh=mesh, in_specs=P('i'), out_specs=P('i'))
    def f(x):
      jax.debug.print("x={x}", x=x)
      y = jnp.cos(x)
      jax.debug.print("y={y}", y=y)
      return y

    x = jnp.arange(2 * len(jax.devices()))

    with jtu.capture_stdout() as output:
      f(x)
      jax.effects_barrier()
    for i in range(len(jax.devices())):
      self.assertIn(f'x=[{2*i} {2*i+1}]', output())

  def test_partial_eval_custom_axis_env(self):
    mesh = Mesh(jax.devices(), ('i',))

    @partial(shard_map, mesh=mesh, in_specs=P('i'), out_specs=P('i'))
    def f(_):
      _, idx = jax.lax.scan(lambda _, __: (None, jax.lax.axis_index('i')),
                            None, None, length=1)
      return idx

    xs = jnp.arange(16.)
    jax.eval_shape(jax.grad(lambda x: jax.remat(f)(x).sum().astype('float32')),
                   xs)

  @jax.legacy_prng_key('allow')
  def test_prngkeyarray_eager(self):
    # https://github.com/google/jax/issues/15398
    mesh = jtu.create_global_mesh((4,), ('x',))
    sharding = jax.sharding.NamedSharding(mesh, P('x'))

    rng = jax.random.PRNGKey(0)
    sharded_rng = jax.random.split(rng, num=4)
    sharded_rng = jax.device_put(sharded_rng, sharding)

    def f(key):
      return jax.random.randint(key[0], shape=(1, 16), minval=0, maxval=16,
                                dtype=jnp.int32)

    pspec = P('x') if config.jax_enable_custom_prng else P('x', None)
    g = shard_map(f, mesh, in_specs=(pspec,), out_specs=pspec)
    _ = g(sharded_rng)  # don't crash!

  def test_functools_partial_rank_error(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    @partial
    def f(x):
      return x

    g = shard_map(f, mesh, in_specs=(P('x', None),), out_specs=P('x',))
    x = jnp.arange(4)
    with self.assertRaises(ValueError):
      g(x)

  def test_in_specs_none_error(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    def f(x): return x

    with self.assertRaisesRegex(TypeError, "but it was None"):
      shard_map(f, mesh, in_specs=None, out_specs=P())(3.)

    # TODO(mattjj): enable this test once we fix the tree_map(f, None, 3.0) bug
    # with self.assertRaises(TypeError):
    #   shard_map(f, mesh, in_specs=(None,), out_specs=P())(3.)

    shard_map(f, mesh, in_specs=P(), out_specs=P())(3.)  # doesn't crash

  def test_scan_rep_rule(self):
    mesh = jtu.create_global_mesh((2, 2,), ('x', 'y'))

    def f(x, y, z):
      x, y, z = x.sum(), y.sum(), z.sum()
      def body(c, _):
        c, *cs = c
        return (*cs, c), None
      out, _  = jax.lax.scan(body, (x, y, z), None, length=3)
      return [jnp.expand_dims(a, 0) for a in out]

    x = jnp.arange(4)

    # doesn't crash, because out_spec assumes no replication (and there is none)
    shard_map(f, mesh, in_specs=(P(None), P('x'), P(('x', 'y'))),
              out_specs=P(('x', 'y')))(x, x, x)

    # does crash, because output incorrectly promises replication
    with self.assertRaisesRegex(ValueError, "require replication"):
      shard_map(f, mesh, in_specs=(P(None), P('x'), P(('x', 'y'))),
                out_specs=P('x'))(x, x, x)
    with self.assertRaisesRegex(ValueError, "require replication"):
      shard_map(f, mesh, in_specs=(P(None), P('x'), P(('x', 'y'))),
                out_specs=P('y'))(x, x, x)
    with self.assertRaisesRegex(ValueError, "require replication"):
      shard_map(f, mesh, in_specs=(P(None), P('x'), P(('x', 'y'))),
                out_specs=P(None))(x, x, x)

    def g(x, y, z):
      x, y, z = x.sum(), y.sum(), z.sum()
      def body(c, _):
        return c, None
      out, _  = jax.lax.scan(body, (x, y, z), None, length=1)
      return [jnp.expand_dims(a, 0) for a in out]

    # doesn't crash, because everything matches
    shard_map(g, mesh, in_specs=(P(None), P('x'), P(('x', 'y'))),
              out_specs=[P(None), P('x'), P(('x', 'y'))])(x, x, x)

    # does crash, because the second guy is wrong
    with self.assertRaisesRegex(ValueError, "require replication"):
      shard_map(g, mesh, in_specs=(P(None), P('x'), P(('x', 'y'))),
                out_specs=[P(None), P(None), P(('x', 'y'))])(x, x, x)

  def test_eager_notimplemented_error_message_custom_jvp(self):
    @jax.custom_jvp
    def foo(x):
      return 2. * x

    @foo.defjvp
    def foo_jvp(primals, tangents):
      (x,), (x_dot,) = primals, tangents
      return foo(x), 2. * x_dot

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(foo, mesh, in_specs=(P('x'),), out_specs=P('x'))
    x = jnp.arange(4.)
    with self.assertRaisesRegex(NotImplementedError, 'custom_jvp'):
      g(x)

  def test_eager_notimplemented_error_message_custom_vjp(self):
    @jax.custom_vjp
    def foo(x):
      return 2. * x

    def foo_fwd(x):
      return x, None

    def foo_bwd(_, y_bar):
      return 2. * y_bar,

    foo.defvjp(foo_fwd, foo_bwd)

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(foo, mesh, in_specs=(P('x'),), out_specs=P('x'))
    x = jnp.arange(4.)
    with self.assertRaisesRegex(NotImplementedError, 'custom_vjp'):
      g(x)

  def test_eager_notimplemented_error_message_axis_index(self):
    def foo(x):
      return x + jax.lax.axis_index('x')

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(foo, mesh, in_specs=(P('x'),), out_specs=P('x'))
    x = jnp.arange(4.)
    with self.assertRaisesRegex(NotImplementedError, 'axis_index'):
      g(x)

  def test_jaxpr_shardings_with_no_outputs(self):
    # https://github.com/google/jax/issues/15385
    mesh = jtu.create_global_mesh((4,), ('i',))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P('i'))
    def f():
      return jax.lax.iota(jnp.dtype('int32'), 4)
    f()  # don't crash

    @partial(shard_map, mesh=mesh, in_specs=(P('i'),), out_specs=P('i'))
    def g(a_block):
      i = jnp.arange(a_block.shape[0])
      return i + a_block

    g(np.arange(32))  # don't crash

  def test_device_put(self):
    mesh = jtu.create_global_mesh((4,), ('i',))

    @partial(shard_map, mesh=mesh, in_specs=P('i'), out_specs=P('i'))
    def f(x):
      return x + jax.device_put(1)

    x = jnp.arange(32.)
    f(x)  # doesn't crash
    jax.jit(f)(x)  # doesn't crash

    @partial(shard_map, mesh=mesh, in_specs=P('i'), out_specs=P('i'))
    def g(x):
      return x + jax.device_put(1, jax.devices()[0])

    with self.assertRaisesRegex(ValueError, "got device"):
      g(x)

    # jit means device_puts are ignored, even those within shmap bodies, so no
    # error!
    jax.jit(g)(x)  # doesn't crash

  @jtu.run_on_devices('cpu', 'gpu', 'tpu')
  def test_key_array_with_replicated_last_tile_dim(self):
    # See https://github.com/google/jax/issues/16137

    mesh = jtu.create_global_mesh((2, 4), ('i', 'j'))

    def f(rng):
      @partial(shard_map, mesh=mesh, in_specs=P('i'), out_specs=P('i'),
               check_rep=False)
      def g(rng):
        return jnp.array([jax.random.normal(rng[0])])
      return g(jax.random.split(rng, 4))

    jax.jit(f)(jax.random.key(0))  # doesn't crash

  # same method appears in api_test.py:DCETest
  # TODO(mattjj): consider moving this method to be a helper in jtu
  def assert_dce_result(self, jaxpr: core.Jaxpr, used_outputs: list[bool],
                        expected_used_inputs: list[bool],
                        expected_num_eqns: Optional[int] = None,
                        check_diff: bool = True):
    jaxpr_dce, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs)
    core.check_jaxpr(jaxpr_dce)
    self.assertEqual(used_inputs, expected_used_inputs)
    if expected_num_eqns is not None:
      all_jaxprs = it.chain([jaxpr_dce], core.subjaxprs(jaxpr_dce))
      num_eqns = sum(len(subjaxpr.eqns) for subjaxpr in all_jaxprs)
      self.assertEqual(num_eqns, expected_num_eqns, msg=str(jaxpr_dce))

    rand_ = jtu.rand_small(np.random.RandomState(0))
    rand  = lambda v: rand_(v.aval.shape, v.aval.dtype)
    consts = [rand(v) for v in jaxpr.constvars]
    inputs = [rand(v) for v in jaxpr.invars   ]
    inputs_dce = [x for x, used in zip(inputs, used_inputs) if used]
    full_outs = core.eval_jaxpr(jaxpr    , consts, *inputs)
    expected_outs_dce = [y for y, used in zip(full_outs, used_outputs) if used]
    outs = core.eval_jaxpr(jaxpr_dce, consts, *inputs_dce)
    self.assertAllClose(outs, expected_outs_dce)

    if check_diff and expected_num_eqns != 0:
      f = lambda *args: core.eval_jaxpr(jaxpr_dce, consts, *args)
      jtu.check_grads(f, inputs_dce, order=2, modes=['rev'])

  def test_dce(self):
    mesh = jtu.create_global_mesh((4, 2), ('i', 'j'))

    def f(x, y, z):
      @partial(shard_map, mesh=mesh, in_specs=(P('i', 'j'), P(None, 'i')),
               out_specs=(P(None, None), P(None, 'i'), P('i', 'j')))
      def g(y, z): return jnp.sin(x), jnp.cos(z), jnp.tan(y)
      return g(y, z)

    x = jnp.zeros((4, 4))
    y = jnp.zeros((8, 8))
    z = jnp.zeros((16, 16))
    jaxpr = jax.make_jaxpr(f)(x, y, z).jaxpr
    self.assertLen(jaxpr.eqns, 1)
    self.assertLen(jaxpr.eqns[0].params['jaxpr'].eqns, 3)

    # If we use all outputs, nothing should be deleted.
    self.assert_dce_result(
        jaxpr,  used_outputs=[True, True, True],
        expected_used_inputs=[True, True, True],
        expected_num_eqns=1 + 3,  # one outer eqn, three remain in body
        check_diff=False)

    # If we drop the last output, the second input should be dropped.
    self.assert_dce_result(
        jaxpr,  used_outputs=[True, True, False],
        expected_used_inputs=[True, False, True],
        expected_num_eqns=1 + 2,  # one outer eqn, two remain in body
        check_diff=False)
    # If we drop the second output, the last input should be dropped.
    self.assert_dce_result(
        jaxpr,  used_outputs=[True, False, True],
        expected_used_inputs=[True, True, False],
        expected_num_eqns=1 + 2,  # one outer eqn, two remain in body
        check_diff=False)
    # If we drop the latter two outputs, the latter two inputs should be dropped
    self.assert_dce_result(
        jaxpr,  used_outputs=[True, False, False],
        expected_used_inputs=[True, False, False],
        expected_num_eqns=1 + 1,  # one outer eqn, two remain in body
        check_diff=False)

    # Finally, try dropping the closed-over value.
    self.assert_dce_result(
        jaxpr,  used_outputs=[False, True, False],
        expected_used_inputs=[False, False, True],
        expected_num_eqns=1 + 1,  # one outer eqn, two remain in body
        check_diff=False)

  def test_post_process_partial_eval_with_scalar_res(self):
    mesh = jtu.create_global_mesh((4, 2), ('i', 'j'))
    g = jax.grad(lambda x: shard_map(lambda: jnp.sin(x), mesh=mesh,
                                     in_specs=P(), out_specs=P())())(2.0)
    self.assertAllClose(g, jnp.cos(2.0), check_dtypes=False)

  def test_sharding_metadata_in_mhlo_attrs(self):
    mesh = Mesh(jax.devices(), ('i',))
    x = jnp.arange(len(jax.devices()), dtype='float32')
    y = jnp.array([3.], dtype='float32')

    def foo(x):
      x = jnp.sin(x)
      x = shard_map(lambda x: jnp.cos(x * y), mesh,
                    in_specs=P('i'), out_specs=P('i'))(x)
      x = shard_map(lambda x: jnp.cos(x * y), mesh,
                    in_specs=P('i'), out_specs=P('i'))(x)
      return x

    mhlo_str = mlir.module_to_string(jax.jit(foo).lower(x).compiler_ir('mhlo'))
    self.assertIn("call @shmap_body", mhlo_str)
    self.assertIn("call @shmap_body_0", mhlo_str)
    self.assertIn("%arg0: tensor<1xf32>", mhlo_str)
    self.assertIn("\"[None]\"", mhlo_str)
    self.assertIn("%arg1: tensor<1xf32>", mhlo_str)
    self.assertIn("\"[('i',)]\"", mhlo_str)
    self.assertIn("-> (tensor<1xf32> {jax.result_info = \"[('i',)]\"})", mhlo_str)

  def test_rewrite_process_call(self):
    def f(x):
      return core.call_p.bind(lu.wrap_init(lambda x: [2. * x]), x)[0] * x

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(f, mesh, in_specs=(P('x'),), out_specs=P('x'))
    x = jnp.arange(4.)
    y = jax.jit(g)(x)  # eager requires shmap to have ShardMapTrace.process_call
    self.assertAllClose(y, 2 * x * x, check_dtypes=True)

  def test_rewrite_post_process_call(self):
    # We shouldn't hit post_process_call here because of RewriteTrace's dynamic
    # behavior (i.e. no data dependence).
    mesh = jtu.create_global_mesh((4,), ('x',))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P('x'),), out_specs=P('x'))
    def f(x):
      return core.call_p.bind(lu.wrap_init(lambda: [2. * x]))[0] * x

    x = jnp.arange(4.)
    y = f(x)
    self.assertAllClose(y, 2 * x * x, check_dtypes=True)

  def test_rewrite_process_custom_jvp_call(self):
    @jax.custom_jvp
    def foo(x):
      return 2. * x

    @foo.defjvp
    def foo_jvp(primals, tangents):
      (x,), (x_dot,) = primals, tangents
      return foo(x), 2. * x_dot

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(lambda x: foo(x) * x, mesh,
                  in_specs=(P('x'),), out_specs=P('x'))
    x = jnp.arange(4.)

    y = jax.jit(g)(x)
    self.assertAllClose(y, 2 * x * x, check_dtypes=True)

    y2, y_dot = jax.jvp(jax.jit(g), (x,), (3 * x,))
    self.assertAllClose(y2, 2 * x * x, check_dtypes=True)
    self.assertAllClose(y_dot, 2 * 2 * 3 * x * x, check_dtypes=True)

  def test_rewrite_process_custom_vjp_call(self):
    @jax.custom_vjp
    def foo(x):
      return 2. * x

    def foo_fwd(x):
      return foo(x), None

    def foo_bwd(_, y_bar):
      return 2. * y_bar,

    foo.defvjp(foo_fwd, foo_bwd)

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(lambda x: foo(x) * x, mesh,
                  in_specs=(P('x'),), out_specs=P('x'))

    x = jnp.arange(4.)
    y = jax.jit(g)(x)
    self.assertAllClose(y, 2 * x * x, check_dtypes=True)

    y_, x_bar = jax.value_and_grad(lambda x: jax.jit(g)(x).sum())(x)
    self.assertAllClose(y_, (2 * x * x).sum(), check_dtypes=True)
    self.assertAllClose(x_bar, 2 * 2 * x, check_dtypes=True)

  def test_rewrite_process_custom_vjp_call_match_more_replicated(self):
    @jax.custom_vjp
    def foo(x):
      return 2. * x

    def foo_fwd(x):
      return foo(x), None

    def foo_bwd(_, y_bar):
      return jnp.ones_like(y_bar),  # diff! more replicated than primal/tangent

    foo.defvjp(foo_fwd, foo_bwd)

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(lambda x: foo(x) * x, mesh,
                  in_specs=(P('x'),), out_specs=P('x'))
    x = jnp.arange(4.)

    y = jax.jit(g)(x)
    self.assertAllClose(y, 2 * x * x, check_dtypes=True)

    y_, x_bar = jax.value_and_grad(lambda x: jax.jit(g)(x).sum())(x)
    self.assertAllClose(y_, (2 * x * x).sum(), check_dtypes=True)
    self.assertAllClose(x_bar, jnp.ones_like(x) + 2 * x, check_dtypes=True)

  def test_rewrite_process_custom_vjp_call_match_less_replicated(self):
    @jax.custom_vjp
    def foo(x, y):
      del y
      return 2. * x

    def foo_fwd(x, y):
      return foo(x, y), y

    def foo_bwd(y, _):
      return y, None  # diff! x_bar less replicated than primal/tangent

    foo.defvjp(foo_fwd, foo_bwd)

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(lambda x, y: foo(x, y) * y, mesh,
                  in_specs=(P(), P('x')), out_specs=P('x'))
    x = jnp.arange(4.)
    y = jnp.arange(4 * 4.)

    z = jax.jit(g)(x, y)
    self.assertAllClose(z, 2 * jnp.tile(x, (4,)) * y, check_dtypes=False)

    z_, x_bar = jax.value_and_grad(lambda x, y: jax.jit(g)(x, y).sum())(x, y)
    self.assertAllClose(z.sum(), z_, check_dtypes=False)
    self.assertAllClose(x_bar, jnp.arange(16).reshape(4, 4).sum(0),
                        check_dtypes=False)

  def test_rewrite_custom_vjp_call_jaxpr(self):
    @jax.custom_vjp
    def foo(x):
      return 2. * x

    def foo_fwd(x):
      return foo(x), None

    def foo_bwd(_, y_bar):
      return 2. * y_bar,

    foo.defvjp(foo_fwd, foo_bwd)

    def foo_scan(x):
      y, _ = jax.lax.scan(lambda x, _: (foo(x), None), x, None, length=1)
      return y

    mesh = jtu.create_global_mesh((4,), ('x',))
    g = shard_map(lambda x: foo_scan(x) * x, mesh,
                  in_specs=(P('x'),), out_specs=P('x'))

    x = jnp.arange(4.)
    y = jax.jit(g)(x)
    self.assertAllClose(y, 2 * x * x, check_dtypes=True)

    y_, x_bar = jax.value_and_grad(lambda x: jax.jit(g)(x).sum())(x)
    self.assertAllClose(y_, (2 * x * x).sum(), check_dtypes=True)
    self.assertAllClose(x_bar, 2 * 2 * x, check_dtypes=True)

  def test_transpose_identity(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P())
    def f(x):
      return x

    jaxpr = jax.make_jaxpr(jax.vjp(f, 1.)[1])(1.)
    e, = jaxpr.jaxpr.eqns
    self.assertEmpty(e.params['jaxpr'].eqns)

    jaxpr = jax.make_jaxpr(jax.vjp(jax.vjp(f, 1.)[1], 1.)[1])((1.,))
    e, = jaxpr.jaxpr.eqns
    self.assertEmpty(e.params['jaxpr'].eqns)

    @partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P())
    def g(x):
      return jax.jit(lambda x: x)(x)

    jaxpr = jax.make_jaxpr(jax.vjp(g, 1.)[1])(1.)
    e, = jaxpr.jaxpr.eqns
    e1, e2 = e.params['jaxpr'].eqns
    self.assertEmpty(e1.outvars)
    self.assertEmpty(e2.params['jaxpr'].eqns)

  def test_fanout_specs_transpose_to_psum(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P('x'))
    def f(x):
      return x

    jaxpr = jax.make_jaxpr(jax.vjp(f, jnp.arange(1.))[1])(jnp.arange(4.))
    e, = jaxpr.jaxpr.eqns
    e2, = e.params['jaxpr'].eqns
    self.assertEqual(str(e2.primitive), 'psum2')
    self.assertEqual(e2.params['axes'], ('x',))

  def test_fanin_psum_transposes_to_fanout(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P())
    def f(x):
      return jax.lax.psum(x, 'x')

    jaxpr = jax.make_jaxpr(jax.vjp(f, jnp.arange(4.))[1])(jnp.array([1.]))
    e, = jaxpr.jaxpr.eqns
    e1, = e.params['jaxpr'].eqns
    self.assertEqual(str(e1.primitive), 'pbroadcast')

  def test_psum_with_implicit_fanout_self_transposes(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f(x):
      return jax.lax.psum(x, 'x')

    jaxpr = jax.make_jaxpr(jax.vjp(f, jnp.arange(4.))[1])(jnp.arange(4.))
    e, = jaxpr.jaxpr.eqns
    e1, e2 = e.params['jaxpr'].eqns
    self.assertEqual(str(e1.primitive), 'psum2')
    self.assertEqual(str(e2.primitive), 'pbroadcast')

  def test_rewrite_binops(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=(P(), P('x')), out_specs=P('x'))
    def f(x, y):
      return x * y

    jaxpr = jax.make_jaxpr(f)(jnp.arange(1.), jnp.arange(4.))
    e, = jaxpr.jaxpr.eqns
    e = e.params['jaxpr'].eqns[0]
    self.assertEqual(e.primitive.name, 'pbroadcast')
    self.assertEqual(e.params['axes'], ('x',))

  def test_rewrite_scan(self):
    mesh = jtu.create_global_mesh((4,), ('x',))

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P('x'))
    def f(x):
      x, _ = jax.lax.scan(lambda x, _: (jax.lax.psum(x, 'x'), None), x, None,
                          length=2)
      return x

    jaxpr = jax.make_jaxpr(f)(jnp.arange(4.))
    e, = jaxpr.jaxpr.eqns
    e, = e.params['jaxpr'].eqns
    e1, e2 = e.params['jaxpr'].eqns
    self.assertEqual(e1.primitive.name, 'psum2')
    self.assertEqual(e2.primitive.name, 'pbroadcast')

  def test_check_rep_false_grads(self):
    # This test is redundant with the systematic tests below, but it serves as a
    # direct regression test for a bug.
    mesh = jtu.create_global_mesh((4,), ('heads',))

    def f(q, k, v):

      def body(q, k, v):
        return q * k[None, :] + v[None, :]

      out = shard_map(body, mesh, check_rep=False,
                      in_specs=(q_spec, kv_spec, kv_spec,),
                      out_specs=q_spec)(q, k, v)
      return out.sum()

    q_spec = P('heads', None)
    kv_spec = P(None)
    q = jax.device_put(jnp.arange(32.).reshape(4, 8), jax.sharding.NamedSharding(mesh, q_spec))
    k = jax.device_put(jnp.arange(8.), jax.sharding.NamedSharding(mesh, kv_spec))
    v = jax.device_put(jnp.arange(8.), jax.sharding.NamedSharding(mesh, kv_spec))

    jtu.check_grads(f, (q, k, v), order=1, modes=['rev'], rtol=1e-2)


class FunSpec(NamedTuple):
  name: str
  num_inputs: int
  fun: Callable
  out_rep: Callable
  valid_types: Optional[Callable] = None

fun_specs = [
    FunSpec('id', 1, lambda x: x, lambda r: r),
    FunSpec('flip', 2, lambda x, y: (y, x), lambda r_x, r_y: (r_y, r_x)),
    FunSpec('transpose', 1, lambda x: x.T, lambda r: r),
    FunSpec('ravel', 1, lambda x: x.ravel(), lambda r: r),
    FunSpec(
        'dot', 2, jnp.dot, lambda r1, r2: r1 & r2,
        lambda x1, x2: (x1.shape and x2.shape and
                        x1.shape[-1] == x2.shape[-2 if x2.ndim > 1 else 0]),
             ),
    FunSpec(
        'sin_dot_sin', 2,
        lambda x1, x2: jnp.sin(jnp.dot(jnp.sin(x1), x2)),
        lambda r1, r2: r1 & r2,
        lambda x1, x2: (x1.shape and x2.shape and
                        x1.shape[-1] == x2.shape[-2 if x2.ndim > 1 else 0])),
]

input_shapes = [
    jax.ShapeDtypeStruct(shape, jnp.dtype('float32'))
    # TODO(mattjj): 0 axis sizes lead to XLA sigfpe, file bug!
    for k in range(1, 4) for shape in it.permutations(range(1, 4), k)
    if not shape or len(set(shape)) > 1  # skip all-equal shapes, boring!
]

mesh_shapes = [
    (1,),
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 4),
    (4, 2),
]

# Reference implementation of shard_map.

ShapeDtypeDuck = Any  # has shape and dtype attributes
Specs = Any  # pytree of PartitionSpec

def shmap_reference(
    body_in_types: Sequence[ShapeDtypeDuck],
    body_out_types: Sequence[ShapeDtypeDuck],
    out_types: Sequence[ShapeDtypeDuck],
    f: Callable, mesh: Mesh, in_specs: Specs, out_specs: Specs
  ) -> Callable:
  def f_shmapped(*args):
    outs = jax.tree_map(lambda y: jnp.zeros(y.shape, y.dtype), out_types)
    getters = [make_indexer(mesh, s, x) for s, x in zip(in_specs, args)]
    putters = jax.tree_map(partial(make_indexer, mesh), out_specs, outs)
    for idx in it.product(*map(range, mesh.shape.values())):
      args_shards = [x[indexer(idx)] for x, indexer in zip(args, getters)]
      assert all(x.shape == r.shape for x, r in zip(args_shards, body_in_types))
      out_shards = f(*args_shards)
      assert jax.tree_util.tree_all(jax.tree_map(lambda y, r: y.shape == r.shape,
                                                 out_shards, body_out_types))
      outs = jax.tree_map(lambda y, out, indexer: out.at[indexer(idx)].set(y),
                          out_shards, outs, putters)
    return outs
  return f_shmapped

def make_indexer(mesh: Mesh, spec: P, x: Any
                 ) -> Callable[[tuple[int, ...]], tuple[slice, ...]]:
  block_shape = [d // math.prod(mesh.shape[ax] for ax in (elt or ()))
                 for d, elt in zip(x.shape, spec)]
  def indexer(idx):
    starts = [0 if el is None else
              idx[list(mesh.shape).index(el)] if type(el) is not tuple else
              sum(idx[list(mesh.shape).index(el[i])]
                  * math.prod(mesh.shape[e] for e in el[i+1:]) for i in range(len(el)))
              for el in spec]
    return tuple(slice(start * size, (start + 1) * size)
                 for start, size in zip(starts, block_shape))
  return indexer


# The code below is similar to named_cases_from_sampler in test_util.py, but it
# uses generators instead of passing a "select" function around.

# To sample test cases efficiently, we construct a generator which yields to the
# caller to choose one of an iterable's options. That is, we can read 'yield' in
# this code as 'choose one'. To call functions which themselves need to make
# choices, we use 'yield from'. That is, we can read 'yield from' in this code
# as 'call this choice-making function'.
Option = Any
CaseSpec = tuple  # first element is a string test name
Chooser = Generator[Iterable[Option], Option, CaseSpec]

def sample_shmap() -> Chooser:
  spec = yield fun_specs
  mesh_shape = yield mesh_shapes
  axis_names = ('i', 'j', 'k', 'l')[:len(mesh_shape)]
  mesh = SimpleNamespace(shape=dict(zip(axis_names, mesh_shape)),
                         axis_names=axis_names)
  in_types = (tys for tys in it.product(input_shapes, repeat=spec.num_inputs)
              if not spec.valid_types or spec.valid_types(*tys))
  body_in_types = yield in_types
  body_out_types = jax.eval_shape(spec.fun, *body_in_types)
  in_types, in_specs = yield from make_in_specs(mesh, body_in_types)
  args = [np.arange(ty.size, dtype=ty.dtype).reshape(ty.shape) / ty.size
          for ty in in_types]
  out_reps = spec.out_rep(*map(partial(unmentioned, mesh), in_specs))
  out_specs = yield from make_out_specs(mesh, body_out_types, out_reps)
  out_types = jax.tree_map(partial(dilate, mesh), out_specs, body_out_types)
  ref = partial(shmap_reference, body_in_types, body_out_types, out_types)
  in_str = '(' + ','.join(jax.core.ShapedArray(t.shape, t.dtype).str_short()
                          for t in in_types) + ')'
  jit = yield [True, False]
  name = f'{spec.name}_{mesh.shape}_jit={jit}_{in_specs}_{out_specs}_{in_str}'
  return name, spec.fun, mesh.shape, jit, in_specs, out_specs, args, ref

def unmentioned(mesh: Mesh, pspec: P) -> set[core.AxisName]:
  return set(mesh.axis_names) - {n for ns in pspec if ns is not None
                                 for n in (ns if type(ns) is tuple else [ns])}


# To drive the sampler, we have `sample` function which just runs a loop.
def sample(num: int, make_gen: Callable[[], Chooser]) -> Iterator[CaseSpec]:
  rng = np.random.RandomState(0)
  seen: set[str] = set()
  while len(seen) < num:
    name, *case = sample_one(rng, make_gen())
    if name not in seen:
      seen.add(name)
      yield name, *case

# To sample one test spec, we run the generator, getting back sequences of
# options from it and sending in our choices from those options until finally a
# test case spec is produced.
def sample_one(rng: np.random.RandomState, gen: Chooser) -> CaseSpec:
  lst = list(next(gen))
  try:
    while True:
      choice = lst[rng.randint(len(lst))]
      lst = list(gen.send(choice))
  except StopIteration as e:
    return e.value

# Next are some choice-making functions for shard_map test specifications.

MeshDuck = Any  # same attributes as a Mesh

def make_in_specs(mesh: MeshDuck, in_types: Sequence[ShapeDtypeDuck]
                  ) -> Chooser:
  pairs = []
  for ty in in_types:
    pair = yield from make_in_spec(mesh, ty)
    pairs.append(pair)
  return tuple(zip(*pairs))

def make_in_spec(mesh: Mesh, in_type_base: ShapeDtypeDuck) -> Chooser:
  assert len(list(powerset(mesh.shape)))
  subset = yield powerset(mesh.shape)
  elts = yield partitions(subset, len(in_type_base.shape))
  partition_spec = P(*(tuple(e) if e else None for e in elts))
  new_type = dilate(mesh, partition_spec, in_type_base)
  return new_type, partition_spec

def dilate(mesh: Mesh, spec: P, shape: ShapeDtypeDuck) -> ShapeDtypeDuck:
  new_shape = tuple(d * math.prod(mesh.shape[ax] for ax in (elt or ()))
                    for d, elt in zip(shape.shape, spec))
  return jax.ShapeDtypeStruct(new_shape, shape.dtype)

def make_out_specs(
    mesh: MeshDuck, out_types: Union[ShapeDtypeDuck, Sequence[ShapeDtypeDuck]],
    out_reps: Union[set[core.AxisName], Sequence[set[core.AxisName]]]
  ) -> Chooser:
  if type(out_types) is not tuple:
    out_spec = yield from make_out_spec(mesh, out_types, out_reps)  # type: ignore
    return out_spec
  else:
    out_specs = []
    for ty, rep in zip(out_types, out_reps):
      out_spec = yield from make_out_spec(mesh, ty, rep)  # type: ignore
      out_specs.append(out_spec)
    return tuple(out_specs)

def make_out_spec(
    mesh: Mesh, out_type: ShapeDtypeDuck, out_rep: set[core.AxisName]
  ) -> Chooser:
  subset = yield (s for s in powerset(mesh.shape)
                  if out_rep | set(s) == set(mesh.shape))
  elts = yield partitions(subset, len(out_type.shape))
  return P(*(tuple(e) if e else None for e in elts))

# Combinatorial helper functions

T = TypeVar('T')
def partitions(s: Sequence[T], k: int) -> Iterator[list[list[T]]]:
  for indices in it.product(range(k), repeat=len(s)):
    outs: list[list[T]] = [[] for _ in range(k)]
    for i, elt in zip(indices, s):
      outs[i].append(elt)
    yield outs

def powerset(s: Iterable[T]) -> Iterator[Sequence[T]]:
  s = list(s)
  return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

# Vmap test helpers

Arr = Any

def sample_shmap_batched(bdim_size: int) -> Chooser:
  name, *shmap_specs, args, ref = yield from sample_shmap()
  bdims = yield all_bdims(*map(op.attrgetter('shape'), args))
  batch_args = map(partial(batchify_arg, bdim_size), bdims, args)
  return name + f'_vmap_{bdims}', bdims, *shmap_specs, batch_args, ref

def all_bdims(*shapes: tuple[int, ...]
              ) -> Iterator[Sequence[Optional[int]]]:
  bdims = ((None, *range(len(shape) + 1)) for shape in shapes)
  return (t for t in it.product(*bdims) if not all(e is None for e in t))

def batchify_arg(size: int, bdim: Optional[int], x: Arr) -> Arr:
  if bdim is None:
    return x
  else:
    iota = np.arange(1, size + 1, dtype=x.dtype).reshape(
        [1 if i != bdim else -1 for i in range(len(x.shape) + 1)])
    return np.expand_dims(x, bdim) * iota

def args_slicer(args: Sequence[Arr], bdims: Sequence[Optional[int]]
                ) -> Callable[[int], Sequence[Arr]]:
  def slicer(x, bdim):
    if bdim is None:
      return lambda _: x
    else:
      return lambda i: x.take(indices=i, axis=bdim)
  slicers = map(slicer, args, bdims)
  return lambda i: [sl(i) for sl in slicers]


class ShardMapSystematicTest(jtu.JaxTestCase):

  @staticmethod
  def make_mesh(mesh_shape):
    return jtu.create_global_mesh(tuple(mesh_shape.values()), tuple(mesh_shape))

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases, sample_shmap))
  def test_eager_against_ref(self, fun, mesh, _, in_specs, out_specs, args, ref):
    mesh = self.make_mesh(mesh)
    args = map(jnp.array, args)
    out = shard_map(fun, mesh, in_specs, out_specs)(*args)
    expected = ref(fun, mesh, in_specs, out_specs)(*args)
    self.assertAllClose(expected, out, check_dtypes=False)

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases, sample_shmap))
  def test_jit_against_ref(self, fun, mesh, _, in_specs, out_specs, args, ref):
    mesh = self.make_mesh(mesh)
    args = map(jnp.array, args)
    out = jax.jit(shard_map(fun, mesh, in_specs, out_specs))(*args)
    expected = ref(fun, mesh, in_specs, out_specs)(*args)
    self.assertAllClose(expected, out, check_dtypes=False)

  @parameterized.named_parameters(
      (name + f'_check_rep={check_rep}', *params, check_rep)
      for (name, *params) in sample(config.FLAGS.jax_num_generated_cases, sample_shmap)
      for check_rep in [True, False]
  )
  @jax.default_matmul_precision("float32")
  def test_grads(self, fun, mesh, jit, in_specs, out_specs, args, _, check_rep):
    mesh = self.make_mesh(mesh)
    args = map(jnp.array, args)
    f = shard_map(fun, mesh, in_specs, out_specs, check_rep=check_rep)
    if jit:
      f = jax.jit(f)
    jtu.check_grads(f, args, order=2, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases, sample_shmap))
  @jax.default_matmul_precision("float32")
  def test_grads_closure(self, fun, mesh, jit, in_specs, out_specs, args, _):
    mesh = self.make_mesh(mesh)
    no_sharding = [all(elt is None for elt in spec) for spec in in_specs]
    args, closed_over_args = partition_list(no_sharding, args)
    in_specs, _ = partition_list(no_sharding, in_specs)
    def f(x, *closed_over_args):
      @partial(shard_map, mesh=mesh, in_specs=(*in_specs,), out_specs=out_specs)
      def g(*args):
        args = [x * arg for arg in args]
        args = merge_lists(no_sharding, args, closed_over_args)
        return fun(*args)
      if jit:
        g = jax.jit(g)
      return g(*args)
    jtu.check_grads(f, (0.2, *closed_over_args), order=2, atol=1e-2, rtol=1e-2)

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases,
             partial(sample_shmap_batched, 5)))
  def test_vmap(self, bdims, fun, mesh, jit, in_specs, out_specs, args, ref):
    mesh = self.make_mesh(mesh)
    args = map(jnp.array, args)

    f = shard_map(fun, mesh, in_specs, out_specs)
    if jit:
      f = jax.jit(f)
    ans = jax.vmap(f, bdims)(*args)

    args_slice = args_slicer(args, bdims)
    expected_slices = [f(*args_slice(i)) for i in range(5)]
    treedef = tree_util.tree_structure(ans)
    if tree_util.treedef_is_strict_leaf(treedef):
      expected = jnp.stack(expected_slices)
    else:
      slices = map(jnp.stack, zip(*expected_slices))
      expected = tree_util.tree_unflatten(treedef, slices)
    tol = 1e-2 if jtu.test_device_matches(['tpu']) else None
    self.assertAllClose(ans, expected, check_dtypes=False, atol=tol, rtol=tol)

  @parameterized.named_parameters(
      sample(config.FLAGS.jax_num_generated_cases,
             partial(sample_shmap_batched, 5)))
  def test_vmap_closure(self, bdims, fun, mesh, jit, in_specs, out_specs, args, _):
    mesh = self.make_mesh(mesh)
    args = map(jnp.array, args)

    no_sharding = [all(elt is None for elt in spec) for spec in in_specs]
    args, closed_over_args = partition_list(no_sharding, args)
    in_specs, _ = partition_list(no_sharding, in_specs)
    explicit_bdims, closed_over_bdims = partition_list(no_sharding, bdims)

    def f(x, *closed_over_args):
      @partial(shard_map, mesh=mesh, in_specs=(*in_specs,), out_specs=out_specs)
      def g(*args):
        args = [x * arg for arg in args]
        args = merge_lists(no_sharding, args, closed_over_args)
        return fun(*args)
      if jit:
        g = jax.jit(g)
      if any(d is not None for d in explicit_bdims):
        return jax.vmap(g, explicit_bdims)(*args)
      else:
        return g(*args)

    xs = jnp.arange(5., dtype='float32')
    ans = jax.vmap(f, (0, *closed_over_bdims))(xs, *closed_over_args)

    args_slice = args_slicer((xs, *closed_over_args), (0, *closed_over_bdims))
    expected_slices = [f(*args_slice(i)) for i in range(5)]
    treedef = tree_util.tree_structure(ans)
    if tree_util.treedef_is_strict_leaf(treedef):
      expected = jnp.stack(expected_slices)
    else:
      slices = map(jnp.stack, zip(*expected_slices))
      expected = tree_util.tree_unflatten(treedef, slices)
    tol = 1e-2 if jtu.test_device_matches(['tpu']) else None
    self.assertAllClose(ans, expected, check_dtypes=False, atol=tol, rtol=tol)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
