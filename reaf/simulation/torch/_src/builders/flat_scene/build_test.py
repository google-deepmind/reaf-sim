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

"""Tests flat_scene build."""

from absl.testing import absltest
import mujoco
from reaf.simulation.torch._src.builders import config as builders_config
from reaf.simulation.torch._src.builders.flat_scene import build
from reaf.simulation.torch._src.builders.flat_scene import config
from reaf.simulation.torch._src.common import clock_device
from reaf.simulation.torch._src.core import environment


class BuildTest(absltest.TestCase):

  def test_build_env_with_site_attachment(self):
    # 1. Create base scene with a site
    base_spec = mujoco.MjSpec()
    base_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
    )
    base_spec.worldbody.add_site(name="attachment_site", pos=[0, 0, 0.1])

    # 2. Create a device with its own spec
    device_spec = mujoco.MjSpec()
    device_spec.worldbody.add_geom(
        name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
    )

    # 3. Define environment config
    env_config = config.Environment(
        base_scene=base_spec,
        sim=builders_config.Sim(xpu_device="cpu", impl="cpp"),
        device_layer=config.DeviceLayer(
            devices={
                "my_device": config.Component(
                    component_type=clock_device.ClockDevice,
                    spec_fn=lambda: device_spec,
                    attachment_site="attachment_site",
                    runtime_params={
                        "name": "my_device",
                        "time_output_name": "time",
                    },
                )
            },
        ),
    )

    # 4. Build environment
    env = build.build_env(env_config)

    # 5. Verify
    self.assertIsNotNone(env)
    # Check if the device's geom exists in the compiled model
    mj_model = env.device_layer.sim.mj_model
    geom_names = [mj_model.geom(i).name for i in range(mj_model.ngeom)]
    self.assertIn("box", geom_names)

  def test_build_env_with_nvmax_non_warp_raises_error(self):
    base_spec = mujoco.MjSpec()
    base_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
    )
    env_config = config.Environment(
        base_scene=base_spec,
        sim=builders_config.Sim(xpu_device="cpu", impl="cpp", nvmax=10),
    )
    with self.assertRaises(ValueError):
      build.build_env(env_config)

  def test_build_env_applies_sim_solver_options(self):
    base_spec = mujoco.MjSpec()
    base_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
    )
    env_config = config.Environment(
        base_scene=base_spec,
        sim=builders_config.Sim(
            xpu_device="cpu",
            impl="cpp",
            iterations=15,
            ls_iterations=22,
            impratio=2.5,
            solver=mujoco.mjtSolver.mjSOL_CG,
            jacobian=mujoco.mjtJacobian.mjJAC_SPARSE,
            cone=mujoco.mjtCone.mjCONE_PYRAMIDAL,
        ),
    )
    env = build.build_env(env_config)
    mj_model = env.device_layer.sim.mj_model
    self.assertEqual(mj_model.opt.iterations, 15)
    self.assertEqual(mj_model.opt.ls_iterations, 22)
    self.assertAlmostEqual(mj_model.opt.impratio, 2.5)
    self.assertEqual(mj_model.opt.solver, mujoco.mjtSolver.mjSOL_CG)
    self.assertEqual(mj_model.opt.jacobian, mujoco.mjtJacobian.mjJAC_SPARSE)
    self.assertEqual(mj_model.opt.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL)

  def test_build_env_preserves_base_spec_solver_options_when_sim_unspecified(
      self,
  ):
    base_spec = mujoco.MjSpec()
    base_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
    )
    base_spec.option.iterations = 77
    base_spec.option.impratio = 3.3
    base_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    env_config = config.Environment(
        base_scene=base_spec,
        sim=builders_config.Sim(xpu_device="cpu", impl="cpp"),
    )
    env = build.build_env(env_config)
    mj_model = env.device_layer.sim.mj_model
    self.assertEqual(mj_model.opt.iterations, 77)
    self.assertAlmostEqual(mj_model.opt.impratio, 3.3)
    self.assertEqual(mj_model.opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)

  def test_build_device_and_task_layer(self):
    base_spec = mujoco.MjSpec()
    base_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
    )
    env_config = config.Environment(
        base_scene=base_spec,
        sim=builders_config.Sim(xpu_device="cpu", impl="cpp"),
    )

    device_layer = build.build_device_layer(env_config)
    self.assertIsNotNone(device_layer)

    task_layer = build.build_task_layer(env_config)
    self.assertIsNotNone(task_layer)

  def test_build_env_runs_startup_once(self):
    # `Environment.__init__` already calls `set_env` and `on_startup`; a second
    # call from the builder would compound relative startup randomization.
    class CountingResetter(environment.EnvironmentReset):
      startups = 0

      def on_startup(self):
        type(self).startups += 1

    base_spec = mujoco.MjSpec()
    base_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
    )
    env_config = config.Environment(
        base_scene=base_spec,
        sim=builders_config.Sim(xpu_device="cpu", impl="cpp"),
        resetters={
            "counter": config.Component(component_type=CountingResetter)
        },
    )
    build.build_env(env_config)
    self.assertEqual(CountingResetter.startups, 1)


class MakeDeploymentConfigTest(absltest.TestCase):

  def _training_config(self) -> config.Environment:
    base_spec = mujoco.MjSpec()
    base_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 1]
    )
    term = config.Term(func=lambda: None)
    return config.Environment(
        base_scene=base_spec,
        task_layer=config.TaskLayer(
            rewards={"reach": term},
            reward_provider=config.Component(component_type=object),
            terminations=[term],
            feature_producers={
                "policy_producer": config.Component(component_type=object),
                "privileged_producer": (
                    config.Component(component_type=object).mark_privileged()
                ),
            },
        ),
        observation_adapter={
            "policy": config.TermGroup(
                terms={"joints": term}, enable_noise=True
            ),
            "privileged": config.TermGroup(terms={"cube_pos": term}),
        },
        resetters={"randomize": config.Component(component_type=object)},
        sim=builders_config.Sim(xpu_device="cpu", impl="cpp", nworld=4096),
    )

  def test_drops_what_only_the_learner_needs(self):
    deploy_config = build.make_deployment_config(
        self._training_config(), observation_groups=["policy"]
    )

    self.assertEmpty(deploy_config.task_layer.rewards)
    self.assertIsNone(deploy_config.task_layer.reward_provider)
    self.assertEmpty(deploy_config.task_layer.terminations)
    self.assertEqual(
        list(deploy_config.task_layer.feature_producers), ["policy_producer"]
    )
    self.assertEqual(deploy_config.sim.nworld, 1)

    # The resetters stage the scene in simulation and are ignored on hardware,
    # so deployment takes them as they are.
    self.assertEqual(list(deploy_config.resetters), ["randomize"])

    # Only the groups the policy consumes survive, without their noise.
    self.assertIsInstance(deploy_config.observation_adapter, dict)
    self.assertEqual(list(deploy_config.observation_adapter), ["policy"])
    self.assertFalse(deploy_config.observation_adapter["policy"].enable_noise)

  def test_leaves_the_training_config_alone(self):
    env_config = self._training_config()
    env_config.nan_termination = True
    deploy_config = build.make_deployment_config(
        env_config, observation_groups=["policy"]
    )

    self.assertTrue(deploy_config.nan_termination)
    self.assertNotEmpty(env_config.task_layer.rewards)
    self.assertEqual(
        list(env_config.task_layer.feature_producers),
        ["policy_producer", "privileged_producer"],
    )
    self.assertEqual(env_config.sim.nworld, 4096)
    self.assertIsInstance(env_config.observation_adapter, dict)
    self.assertTrue(env_config.observation_adapter["policy"].enable_noise)


class ConfigTest(absltest.TestCase):

  def test_obs_rejects_positional_args_and_string_key_kwargs(self):
    with self.assertRaises(TypeError):
      config.Obs(lambda features: None, 123)
    with self.assertRaises(TypeError):
      config.Obs("joint_pos", extra_kwarg=1)  # pyrefly: ignore[bad-argument-type]

  def test_component_configure_spec_rejects_positional_args(self):
    comp = config.Component(clock_device.ClockDevice)
    with self.assertRaises(TypeError):
      comp.configure_spec(123)  # pyrefly: ignore[bad-argument-count]


if __name__ == "__main__":
  absltest.main()
