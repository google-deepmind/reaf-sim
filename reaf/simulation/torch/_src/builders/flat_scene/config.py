# Copyright 2026 Google LLC
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

"""Environment configuration dataclasses.

Defines the declarative configuration for building REAF environments.
Uses a flat device-list pattern without scene-embodiment attachment logic.
"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Concatenate, Generic, ParamSpec, Protocol, Self, overload

import mujoco
from reaf.core import action_space_adapter as core_action_space_adapter
from reaf.simulation.torch._src.builders.config import Sim

_P = ParamSpec("_P")

RESERVED_PARAM_KEYS = frozenset({
    "bias",
    "noise",
    "scale",
    "offset",
    "clip",
    "history_length",
})


def _feature(features: Mapping[str, Any], *, key: str) -> Any:
  """Extracts a named feature tensor from the features dict."""
  return features[key]


@dataclass
class Term:
  """Universal term: function + params + optional fields.

  For observation terms in ``FunctionalObservationAdapter``, reserved keys in
  ``params`` (not forwarded to ``func``) are applied in order:
  ``bias -> noise -> scale -> offset -> clip -> history_length``
  (where ``bias`` and ``noise`` are only applied when the enclosing
  ``ObsGroup`` has ``enable_noise=True``).
  """

  func: Callable[..., Any]
  params: dict[str, Any] = field(default_factory=dict)

  # optional fields
  weight: float | None = None
  output_shape: tuple[int, ...] | None = None
  dtype: Any = None

  def __init__(
      self,
      func: Callable[..., Any],
      # TODO(b/564469814): Remove legacy `params=` dict once external callers
      # migrate to passing function kwargs directly.
      params: dict[str, Any] | None = None,
      *,
      weight: float | None = None,
      output_shape: tuple[int, ...] | None = None,
      dtype: Any = None,
      **extra_params: Any,
  ):
    self.func = func
    merged: dict[str, Any] = {}
    if params is not None:
      merged.update(params)
    merged.update(extra_params)
    self.params = merged
    self.weight = weight
    self.output_shape = output_shape
    self.dtype = dtype


class Obs(Term):
  """Observation term with explicit post-processing modifiers and kwargs."""

  @overload
  def __init__(self, key: str) -> None:
    ...

  @overload
  def __init__(
      self,
      func: Callable[Concatenate[Any, _P], Any],
      *args: _P.args,
      **kwargs: _P.kwargs,
  ) -> None:
    ...

  def __init__(
      self,
      func_or_key: Callable[..., Any] | str,
      *args: Any,
      **kwargs: Any,
  ):
    if args:
      raise TypeError("Obs() only accepts keyword arguments for `func`.")
    if isinstance(func_or_key, str):
      if kwargs:
        raise TypeError(
            f"Obs({func_or_key!r}) does not accept keyword arguments; "
            "use .configure(...) for observation modifiers."
        )
      super().__init__(_feature, key=func_or_key)
    else:
      super().__init__(func_or_key, params=dict(kwargs))

  def configure(
      self,
      *,
      bias: Any = None,
      noise: Any = None,
      scale: Any = None,
      offset: Any = None,
      clip: dict[str, Any] | None = None,
      history_length: int = 0,
      dtype: Any = None,
      output_shape: tuple[int, ...] | None = None,
  ) -> Self:
    """Configures observation post-processing modifiers and metadata."""
    # TODO(b/564469814): Store post-processing modifiers as explicit fields on
    # `Obs` once legacy `Term(..., params={"noise": ...})` callers migrate.
    if bias is not None:
      self.params["bias"] = bias
    if noise is not None:
      self.params["noise"] = noise
    if scale is not None:
      self.params["scale"] = scale
    if offset is not None:
      self.params["offset"] = offset
    if clip is not None:
      self.params["clip"] = clip
    if history_length > 0:
      self.params["history_length"] = history_length
    if dtype is not None:
      self.dtype = dtype
    if output_shape is not None:
      self.output_shape = output_shape
    return self


class Reward(Term):
  """Reward term with explicit weight and kwargs."""

  def __init__(
      self,
      func: Callable[..., Any],
      params: dict[str, Any] | None = None,
      *,
      weight: float = 1.0,
      **extra_params: Any,
  ):
    super().__init__(func, params=params, weight=weight, **extra_params)


@dataclass
class ObsGroup:
  """Group of observation terms with shared settings (e.g.

  per-episode bias and per-step noise).
  """

  terms: dict[str, Term] = field(default_factory=dict)
  enable_noise: bool = False

  def __init__(
      self,
      terms: Mapping[str, Term] | None = None,
      *,
      enable_noise: bool = False,
      **named_terms: Term,
  ):
    merged: dict[str, Term] = {}
    if terms is not None:
      merged.update(terms)
    merged.update(named_terms)
    self.terms = merged
    self.enable_noise = enable_noise


# TODO(b/564469814): Remove `TermGroup` alias once external callers migrate to
# `ObsGroup`.
TermGroup = ObsGroup


_P_init = ParamSpec("_P_init")
_P_spec = ParamSpec("_P_spec")


class _Configurable(Protocol[_P_spec]):
  """Protocol capturing a device's `configure_spec` parameter specification."""

  def configure_spec(
      self,
      spec: mujoco.MjSpec,
      prefix: str,
      *args: _P_spec.args,
      **kwargs: _P_spec.kwargs,
  ) -> mujoco.MjSpec:
    ...


@dataclass
class Component(Generic[_P_spec]):
  """Unified configuration for REAF devices, processors, etc.."""

  # The class of the component.
  component_type: type[Any]  # e.g., ActuatorDevice, CameraDevice, etc.
  # Parameters sent to configure_spec.
  build_params: dict[str, Any] = field(default_factory=dict)
  # Parameters sent to object constructor.
  runtime_params: dict[str, Any] = field(default_factory=dict)

  # Optional scene spec and attachment info.
  spec_fn: Callable[[], mujoco.MjSpec] | None = None
  attachment_prefix: str = ""
  attachment_site: str | None = None
  attachment_pos: tuple[float, float, float] | None = None
  attachment_quat: tuple[float, float, float, float] | None = None

  @overload
  def __init__(
      self: "Component[_P_spec]",
      component_type: Callable[_P_init, _Configurable[_P_spec]],
      *args: _P_init.args,
      **runtime_kwargs: _P_init.kwargs,
  ) -> None:
    ...

  @overload
  def __init__(
      self: "Component[[]]",
      component_type: Callable[_P_init, Any],
      *args: _P_init.args,
      **runtime_kwargs: _P_init.kwargs,
  ) -> None:
    ...

  # TODO(b/564469814): Legacy overload kept for backwards compatibility with
  # unmigrated external callers passing `build_params`/`runtime_params` dicts
  # or `attachment_*` kwargs directly. Remove once external callers migrate to
  # `Component(Cls, **runtime_kwargs).configure_spec(...).attach(...)`.
  @overload
  def __init__(
      self: "Component[...]",
      component_type: Any,
      build_params: dict[str, Any] | None = None,
      runtime_params: dict[str, Any] | None = None,
      *,
      spec_fn: Callable[[], mujoco.MjSpec] | None = None,
      attachment_prefix: str = "",
      attachment_site: str | None = None,
      attachment_pos: tuple[float, float, float] | None = None,
      attachment_quat: tuple[float, float, float, float] | None = None,
  ) -> None:
    ...

  def __init__(
      self,
      component_type: Any,
      build_params: dict[str, Any] | None = None,
      runtime_params: dict[str, Any] | None = None,
      *,
      spec_fn: Callable[[], mujoco.MjSpec] | None = None,
      attachment_prefix: str = "",
      attachment_site: str | None = None,
      attachment_pos: tuple[float, float, float] | None = None,
      attachment_quat: tuple[float, float, float, float] | None = None,
      **runtime_kwargs: Any,
  ):
    self.component_type = component_type
    self.build_params = dict(build_params) if build_params is not None else {}
    merged_runtime: dict[str, Any] = {}
    if runtime_params is not None:
      merged_runtime.update(runtime_params)
    merged_runtime.update(runtime_kwargs)
    self.runtime_params = merged_runtime
    self.spec_fn = spec_fn
    self.attachment_prefix = attachment_prefix
    self.attachment_site = attachment_site
    self.attachment_pos = attachment_pos
    self.attachment_quat = attachment_quat

  def configure_spec(
      self, *args: _P_spec.args, **build_kwargs: _P_spec.kwargs
  ) -> Self:
    """Configures MjSpec build-time arguments on this component."""
    if args:
      raise TypeError("configure_spec() only accepts keyword arguments.")
    self.build_params.update(build_kwargs)
    return self

  def attach(
      self,
      spec_fn: Callable[[], mujoco.MjSpec] | None = None,
      *,
      prefix: str | None = None,
      site: str | None = None,
      pos: tuple[float, float, float] | None = None,
      quat: tuple[float, float, float, float] | None = None,
  ) -> Self:
    """Configures how this device's MjSpec attaches to the base scene."""
    if spec_fn is not None:
      self.spec_fn = spec_fn
    if prefix is not None:
      self.attachment_prefix = prefix
    if site is not None:
      self.attachment_site = site
    if pos is not None:
      self.attachment_pos = pos
    if quat is not None:
      self.attachment_quat = quat
    return self


@dataclass
class TaskLayer:
  """Task logic configuration."""

  # TODO(b/564469814): Change `rewards` to `dict[str, Reward]` once external
  # callers migrate off `Term`.
  rewards: dict[str, Any] = field(default_factory=dict)
  reward_provider: Component | None = None  # defaults to weighted sum
  # TODO(b/564469814): Change `terminations`, `command_processors`,
  # `feature_producers`, and `command_feature_producers` to `dict[str, ...]`
  # with `default_factory=dict` once external callers migrate off `list`.
  terminations: Any = field(default_factory=list)
  command_processors: Any = field(default_factory=list)
  feature_producers: Any = field(default_factory=list)
  command_feature_producers: Any = field(default_factory=list)


@dataclass
class DeviceLayer:
  """Configuration for simulation devices."""

  devices: dict[str, Component] = field(default_factory=dict)
  # TODO(b/564469814): Change `command_processors` and
  # `substep_measurement_processors` to `dict[str, Component]` with
  # `default_factory=dict` once external callers migrate off `list`.
  command_processors: Any = field(default_factory=list)
  substep_measurement_processors: Any = field(default_factory=list)


@dataclass
class Environment:
  """Top-level environment configuration."""

  # Scene - the arena using our lightweight spec DSL or mjspec.
  base_scene: mujoco.MjSpec

  # Configuration for simulation devices (devices, processors).
  device_layer: DeviceLayer = field(default_factory=DeviceLayer)

  # Task - the logic shared between real and sim.
  task_layer: TaskLayer = field(default_factory=TaskLayer)

  action_space_adapter: core_action_space_adapter.ActionSpaceAdapter | None = (
      None
  )
  # TODO(b/564469814): Change `observation_adapter` to `dict[str, ObsGroup]`
  # once external callers remove `if cfg.observation_adapter is None:` checks.
  observation_adapter: Any = field(default_factory=dict)

  # The all-knowing sim Gods of the experiment.
  resetters: dict[str, Component] = field(default_factory=dict)

  # Simulation parameters.
  sim: Sim = field(default_factory=Sim)

  episode_length_secs: float = 20.0
  nan_termination: bool = False
