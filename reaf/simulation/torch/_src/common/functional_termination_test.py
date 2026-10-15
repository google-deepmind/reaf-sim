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

"""Tests for FunctionalTerminationChecker."""

import dataclasses
from typing import Any
from typing import Callable

from absl.testing import absltest
from reaf.simulation.torch._src.common import functional_termination
from reaf.simulation.torch._src.core import termination_checker
import torch

T = termination_checker.TerminationResult.TERMINATE
R = termination_checker.TerminationResult.TRUNCATE
N = termination_checker.TerminationResult.DO_NOT_TERMINATE


@dataclasses.dataclass
class Term:
  func: Callable[..., torch.Tensor]
  params: dict[str, Any] = dataclasses.field(default_factory=dict)
  weight: float | None = None
  output_shape: tuple[int, ...] | None = None


class FunctionalTerminationCheckerTest(absltest.TestCase):

  def test_terminate_and_truncate(self):
    checker = functional_termination.FunctionalTerminationChecker(
        name="test",
        terms=[
            Term(
                func=lambda m, key: torch.where(m[key], T, N),
                params={"key": "bad_contact"},
            ),
            Term(
                func=lambda m, key: torch.where(m[key], R, N),
                params={"key": "timeout"},
            ),
        ],
        required_keys=["bad_contact", "timeout"],
    )
    checker.initialize(nworld=2, device=torch.device("cpu"))

    def check(bad_contact, timeout):
      m = {
          "bad_contact": torch.tensor(bad_contact, dtype=torch.bool),
          "timeout": torch.tensor(timeout, dtype=torch.bool),
      }
      return checker.check_termination(m).tolist()

    # Neither condition met -> DO_NOT_TERMINATE
    self.assertEqual(check([False, False], [False, False]), [N, N])
    # Terminate takes priority over truncate
    self.assertEqual(check([True, False], [False, False]), [T, N])
    self.assertEqual(check([False, False], [True, False]), [R, N])
    self.assertEqual(check([True, False], [True, False]), [T, N])  # T > R
    # Both worlds
    self.assertEqual(check([True, True], [True, True]), [T, T])


if __name__ == "__main__":
  absltest.main()
