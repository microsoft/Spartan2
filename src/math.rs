// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

pub trait Math {
  fn log_2(self) -> usize;
}

impl Math for usize {
  fn log_2(self) -> usize {
    assert_ne!(self, 0);

    if self.is_power_of_two() {
      (1usize.leading_zeros() - self.leading_zeros()) as usize
    } else {
      (0usize.leading_zeros() - self.leading_zeros()) as usize
    }
  }
}
