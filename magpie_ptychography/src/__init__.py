# Copyright 2024 Borong Zhang
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MAGPIE ptychography.

Public API exports:
- Algorithms (MAGPIE variants, rPIE, L-BFGS)
- Misfits (objective + gradient)
- Synthetic data generation
"""

__version__ = "0.1.0"

# --- Algorithms ---
from magpie_ptychography.algorithms import (
    magpie_recursion,
    magpie_loop,
    rpie,
    lbfgs,
)

# --- Misfits ---
from magpie_ptychography.misfits import feasibility_distance_misfit

# --- Synthetic data ---
from magpie_ptychography.synthetic import (
    generate_intensity_measurements,
    generate_scanning_positions,
)

__all__ = [
    "__version__",
    # algorithms
    "magpie_recursion",
    "magpie_loop",
    "rpie",
    "lbfgs",
    # misfits
    "feasibility_distance_misfit",
    # synthetic
    "generate_intensity_measurements",
    "generate_scanning_positions",
]
