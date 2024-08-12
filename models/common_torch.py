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
from typing import Optional
from open3d.ml.torch import ops
import torch


def window_poly6(r_sqr):
    return torch.clamp((1 - r_sqr)**3, 0, 1)


class SpecialSparseConv(torch.nn.Module):
    """Special Sparse Convolution. This layer computes a convolution which is only
    evaluated at the specified output positions.

    Arguments:
        filters: The number of filters/output channels.

        kernel_size: The kernel size must be 55 for convolutions within the same
            adaptive grid and 9 for transitions between grids.

        activation: The activation function to use. None means no activation.

        use_bias: If True adds an additive bias vector.

        kernel_initializer: Initializer for the kernel weights.

        bias_initializer: Initializer for the bias vector.

        kernel_regularizer: Regularizer for the kernel weights.

        bias_regularizer: Regularizer for the bias vector.

        normalize: If true then the result is normalized by the number of input points.
    """

    def __init__(
            self,
            in_channels,
            filters,
            kernel_size,
            activation=None,
            use_bias=True,
            kernel_initializer=lambda x: torch.nn.init.uniform_(x, -0.05, 0.05),
            bias_initializer=torch.nn.init.zeros_,
            normalize: bool = False,
            **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.kernel_size = kernel_size
        if activation is not None:
            self.activation = activation
        else:

            def identity(x):
                return x

            self.activation = identity

        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.normalize = normalize

        if not kernel_size in (9, 55):
            raise Exception("kernel size must bei 9 or 55.")

        kernel_shape = (self.kernel_size, self.in_channels, self.filters)
        self.kernel = torch.nn.Parameter(data=torch.Tensor(*kernel_shape),
                                         requires_grad=True)
        self.kernel_initializer(self.kernel)

        if self.use_bias:
            self.bias = torch.nn.Parameter(data=torch.Tensor(self.filters),
                                           requires_grad=True)
            self.bias_initializer(self.bias)
        else:
            self.bias = torch.empty((0,), dtype=torch.float32)

    def forward(self,
                inp_features,
                neighbors_index,
                neighbors_kernel_index,
                neighbors_row_splits,
                inp_importance: Optional[torch.Tensor] = None):
        """This function computes the output features.

        Arguments:

          inp_features: A 2D tensor which stores a feature vector for each input
            point.

          neighbors_index: Stores the indices of the face-adjacent voxel neighbors.
            The start and end position for each voxel is defined by the array
            'neighbors_row_splits'.
            
          neighbors_kernel_index: The index for the kernel element to use in the 
            generalized sparse conv. Start and end for each voxel (row) is defined
            by the 'neighbors_row_splits' array.

          neighbors_row_splits: Defines the start and end for each voxel (row) for
            the 'neighbors_index' and 'neighbors_kernel_index' arrays.

          inp_importance: Optional scalar importance value for each input point.

        Returns: A tensor of shape [num output points, filters] with the output
          features.
        """
        if inp_importance is not None:
            neighbors_importance = inp_importance[neighbors_index.to(
                torch.int64)]
            out_importance = ops.reduce_subarrays_sum(neighbors_importance,
                                                      neighbors_row_splits)
        else:
            neighbors_importance = torch.empty((0,), dtype=torch.float32)
            out_importance = torch.empty((0,), dtype=torch.float32)

        out_features = ops.sparse_conv(
            filters=self.kernel,
            inp_features=inp_features,
            inp_importance=torch.empty((0,), dtype=torch.float32),
            neighbors_index=neighbors_index,
            neighbors_kernel_index=neighbors_kernel_index,
            neighbors_importance=neighbors_importance,
            neighbors_row_splits=neighbors_row_splits,
            normalize=self.normalize,
        )

        if self.use_bias:
            out_features += self.bias
        out_features = self.activation(out_features)

        return out_features, out_importance
