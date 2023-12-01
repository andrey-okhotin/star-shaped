# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""

import torch
import numpy as np

from models.ncsnpp_unet.models.ncsnpp import NCSNpp


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max),
            np.log(config.model.sigma_min),
            config.model.num_scales,
        )
    )

    return sigmas


def create_model(config):
    name2cls = {"ncsnpp": NCSNpp}
    model_name = config.model.name
    return name2cls[model_name](config)
