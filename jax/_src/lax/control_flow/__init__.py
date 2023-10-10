# Copyright 2022 The JAX Authors.
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
"""Module for the control flow primitives."""
# TODO(mattjj): fix dependent library which expects optimization_barrier_p here
from jax._src.ad_checkpoint import optimization_barrier_p
# Private utilities used elsewhere in JAX
# TODO(sharadmv): lift them into a more common place
from jax._src.lax.control_flow.common import (
    _check_tree_and_avals, _initial_style_jaxpr,
    _initial_style_jaxprs_with_common_consts, _initial_style_open_jaxpr)
from jax._src.lax.control_flow.conditionals import cond, cond_p, switch
from jax._src.lax.control_flow.loops import (
    _scan_impl, associative_scan, cumlogsumexp, cumlogsumexp_p, cummax,
    cummax_p, cummin, cummin_p, cumprod, cumprod_p, cumred_reduce_window_impl,
    cumsum, cumsum_p, fori_loop, map, scan, scan_bind, scan_p, while_loop,
    while_p)
from jax._src.lax.control_flow.solves import (
    _custom_linear_solve_impl, custom_linear_solve, custom_root,
    linear_solve_p)
