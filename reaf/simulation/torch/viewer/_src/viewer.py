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

"""reaf-sim environment viewer."""

import importlib
import logging
from typing import Any, Iterable

import dm_env
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
from mujoco.experimental.studio import launch_web
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol
from reaf.simulation.torch import environment as environment_module
from reaf.simulation.torch import simulator as simulator_module
from reaf.simulation.torch import types
from reaf.simulation.torch.viewer._src import sim_backend
from reaf.simulation.viewer import sim_plugin
from reaf.simulation.viewer import viewer_plugin
import torch
import tree

_module_logger = logging.getLogger(__name__)


class _ZeroActionPolicy(gdmr_policy.Policy[gdmr_types.PolicyDataType]):
  """An agent that always returns a zero action."""

  def __init__(self, action_spec, xpu_device: str = 'cuda:0'):
    self._action = tree.map_structure(
        lambda a: torch.zeros(
            a.shape,
            dtype=types.NUMPY_TO_TORCH_DTYPE[a.dtype],
            device=xpu_device,
        ),
        action_spec,
    )

  def step(self, timestep: dm_env.TimeStep, prev_state=()):
    del timestep, prev_state
    return (self._action, {}), ()

  def initial_state(self):
    return ()

  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec):
    del timestep_spec


def launch(
    environment: gdmr_env.Environment,
    policy: gdmr_policy.Policy[gdmr_types.PolicyDataType] | None = None,
    simulator: simulator_module.Simulator | None = None,
    step_duration: float | None = None,
    reward_plot_keys: Iterable[str] | None = None,
    loggers: Iterable[Any] | None = None,
    main_world_id: int = 0,
    gfx: str = '',
    http_port: int = 0,
    auto_reset: bool = True,
    initial_instruction: str | None = None,
    viewer_plugins: list[Any] | None = None,
    sim_plugins: list[Any] | None = None,
) -> None:
  """Launches a viewer for a reaf-sim environment."""
  if simulator is None:
    if isinstance(environment, environment_module.Environment):
      simulator = environment.device_layer.sim
    else:
      simulator = environment.unwrapped.device_layer.sim  # pyrefly: ignore[missing-attribute]
  if loggers is not None:
    loggers_list = list(loggers)
  else:
    loggers_list = []

  if policy is None:
    policy = _ZeroActionPolicy(
        environment.action_spec(), xpu_device=simulator.xpu_device
    )

  # Compile model/data from the simulator.
  mj_spec = simulator.mj_spec
  model = mj_spec.compile()
  data = mujoco.MjData(model)

  # Create sim and viewer plugins.
  torch_sim_backend = sim_backend.TorchSimBackend(
      environment=environment,
      policy=policy,
      simulator=simulator,
      xpu_device=simulator.xpu_device,
      world_id=main_world_id,
      loggers=loggers_list,
  )
  torch_sim_plugin = sim_plugin.SimPlugin(
      backend=torch_sim_backend,
      step_duration=step_duration,
      auto_reset=auto_reset,
  )
  reaf_viewer_plugin = viewer_plugin.ReafViewerPlugin(
      reward_legends=viewer_plugin.reward_spec_legends(
          torch_sim_backend.reward_spec(), reward_plot_keys
      ),
      instruction=initial_instruction or '',
  )

  # Initial data sync. The first sim step resets the environment.
  torch_sim_backend.sync_to_viewer(model, data)

  # Native viewer is an optional dependency; fall back to web if unavailable.
  run_fn = launch_web.run
  if gfx not in ('web', 'webgl'):
    try:
      run_fn = importlib.import_module(
          'mujoco.experimental.studio.launch_native'
      ).run
    except ImportError:
      pass

  config = viewer_protocol.ViewerConfig(
      title='reaf-sim Environment Viewer',
      gfx=gfx,
      http_port=http_port,
      open_browser=True,
  )

  # Launch the viewer using the studio launch API. Errors from sim plugins,
  # the policy, or the viewer thread propagate to the caller.
  _module_logger.info('Launching viewer...')
  run_fn(
      config,
      model=model,
      data=data,
      viewer_plugins=[viewer_app.ViewerApp(), reaf_viewer_plugin]
      + list(viewer_plugins or []),
      sim_plugins=[torch_sim_plugin] + list(sim_plugins or []),
  )
  _module_logger.info('Exited viewer.')
