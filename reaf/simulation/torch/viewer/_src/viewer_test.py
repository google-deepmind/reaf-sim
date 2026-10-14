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

from unittest import mock

from absl.testing import absltest
import dm_env
from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch import simulator as simulator_module
from reaf.simulation.torch import types
from reaf.simulation.torch.viewer._src import sim_backend
from reaf.simulation.torch.viewer._src import viewer
import torch


class ViewerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.spec = mujoco.MjSpec()
    body = self.spec.worldbody.add_body(name='box')
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0.1, 0.1])
    self.sim = simulator_module.Simulator(
        mj_spec=self.spec, nworld=1, impl=types.Impl.CPP, xpu_device='cpu'
    )

  def test_zero_action_policy(self):
    action_spec = {
        'action': specs.Array(shape=(4,), dtype=np.float32),
    }
    policy = viewer._ZeroActionPolicy(action_spec, xpu_device='cpu')
    self.assertEqual(policy.initial_state(), ())
    assert isinstance(policy._action, dict)
    self.assertIn('action', policy._action)
    self.assertEqual(policy._action['action'].shape, (4,))
    self.assertTrue(torch.all(policy._action['action'] == 0))

    (action, policy_extra), state = policy.step(dm_env.restart({}))
    self.assertIn('action', action)
    self.assertEqual(policy_extra, {})
    self.assertEqual(state, ())

  def test_sync_to_viewer(self):
    model = self.spec.compile()
    data = mujoco.MjData(model)
    action_spec = {'action': specs.Array(shape=(4,), dtype=np.float32)}
    policy = viewer._ZeroActionPolicy(action_spec, xpu_device='cpu')
    backend = sim_backend.TorchSimBackend(
        environment=mock.MagicMock(),
        policy=policy,
        simulator=self.sim,
        xpu_device='cpu',
        world_id=0,
    )
    backend.sync_to_viewer(model, data)
    self.assertEqual(len(model.geom_size), len(self.sim.mj_model.geom_size))


if __name__ == '__main__':
  absltest.main()
