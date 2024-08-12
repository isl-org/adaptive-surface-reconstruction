#
# Copyright 2024 Intel
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
#
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join('..', '..'))
from datareader import read_data, val_files

import numpy as np
import torch
import tensorflow as tf
from models.v0.net_definitions_torch import UNet5
import argparse


def create_model(cfg):
    """Returns an instance of the network for training and evaluation"""
    model = UNet5(**cfg)
    return model


def main():
    parser = argparse.ArgumentParser(
        description=
        "Script for converting the tensorflow checkpoint to a torch script model"
    )
    parser.add_argument("--cfg",
                        type=str,
                        required=True,
                        help="The path to the config file")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Path to the tensorflow checkpoint")
    args = parser.parse_args()

    reader = tf.train.load_checkpoint(args.checkpoint)
    shape_from_key = reader.get_variable_to_shape_map()

    val_dataset = read_data(
        files=val_files,
        batch_size=1,
        repeat=False,
        random_rotation=False,
        num_points_min=200000,
        num_points_max=200000,
        num_meshes_min=1,
        num_meshes_max=1,
        shuffle_buffer=None,
        num_parallel_calls=1,
        convert_uint64=True,
        octree_levels=UNet5.octree_levels,
    )
    val_data_iter = iter(val_dataset)
    validation_example = next(val_data_iter)[0]
    validation_example['feats'] = np.concatenate([
        validation_example['normals'],
        np.ones_like(validation_example['normals'][:, :1])
    ],
                                                 axis=-1)
    data = {
        k: torch.from_numpy(v)
        for k, v in validation_example.items()
        if isinstance(v, np.ndarray) and v.dtype not in (np.uint32,)
    }

    model = create_model(cfg['model'])

    state_dict = model.state_dict()

    # copy parameters to the state_dict
    for k, v in state_dict.items():
        if k.endswith('offset') and v.count_nonzero().item() == 0:
            print('skipping', k)
            continue
        torch_shape = list(v.shape)

        tf_key = 'model/' + k.replace('.', '/')
        if tf_key.endswith('weight'):
            tf_key = tf_key.replace('weight', 'kernel')
        tf_key += '/.ATTRIBUTES/VARIABLE_VALUE'
        tf_shape = shape_from_key[tf_key]

        print(k, torch_shape)
        arr = reader.get_tensor(tf_key)
        if k.endswith('.weight') and len(
                torch_shape) == 2 and torch_shape[::-1] == tf_shape:
            # transpose kernel for linear layers in torch
            arr = arr.transpose()
        elif torch_shape == tf_shape:
            pass
        else:
            raise Exception('{}: {}, {}'.format(k, torch_shape, tf_shape))
        tensor = torch.from_numpy(arr)
        state_dict[k][...] = tensor

    model.load_state_dict(state_dict)

    # serialize to torch script
    agg = model.aggregate(data)
    code = model.unet(agg, data)
    shift = torch.zeros([code.shape[0], 3])

    script = torch.jit.trace_module(model, {
        'aggregate': data,
        'unet': (agg, data),
        'decode': (shift, code)
    })
    script.save('model.pt')


if __name__ == '__main__':
    sys.exit(main())
