#
# Copyright 2022 Intel (Autonomous Agents Lab)
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
from open3d.ml.tf import ops
import tensorflow as tf


def window_poly6(r_sqr):
    return tf.clip_by_value((1 - r_sqr)**3, 0, 1)


class SpecialSparseConv(tf.keras.layers.Layer):
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

    def __init__(self,
                 filters,
                 kernel_size,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 normalize=False,
                 **kwargs):

        from tensorflow.keras import activations, initializers, regularizers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.normalize = normalize

        if not kernel_size in (9, 55):
            raise Exception("kernel size must bei 9 or 55.")

        super().__init__(**kwargs)

    def build(self, inp_features_shape):
        self.in_channels = inp_features_shape[-1]

        kernel_shape = tf.TensorShape(
            (self.kernel_size, self.in_channels, self.filters))
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=self.trainable,
        )

        if self.use_bias:
            bias_shape = tf.TensorShape((self.filters,))
            self.bias = self.add_weight(
                name="bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=self.trainable,
            )
        super().build(inp_features_shape)

    def call(self,
             inp_features,
             neighbors_index=None,
             neighbors_kernel_index=None,
             neighbors_row_splits=None,
             inp_importance=None):
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
        compute_dtype = tf.dtypes.as_dtype(self._dtype_policy.compute_dtype)

        if inp_importance is not None:
            neighbors_importance = tf.gather(inp_importance, neighbors_index)
            out_importance = ops.reduce_subarrays_sum(neighbors_importance,
                                                      neighbors_row_splits)
        else:
            neighbors_importance = tf.ones((0,), dtype=compute_dtype)

        _conv_values = {
            'filters': self.kernel,
            'inp_features': inp_features,
            'inp_importance': tf.ones((0,), dtype=compute_dtype),
            'neighbors_index': neighbors_index,
            'neighbors_kernel_index': neighbors_kernel_index,
            'neighbors_importance': neighbors_importance,
            'neighbors_row_splits': neighbors_row_splits,
            'normalize': self.normalize,
            'output_type': tf.dtypes.as_dtype(self._dtype_policy.compute_dtype),
        }

        out_features = ops.sparse_conv(**_conv_values)

        if self.use_bias:
            out_features += self.bias
        out_features = self.activation(out_features)

        if inp_importance is not None:
            return out_features, out_importance
        else:
            return out_features

    def compute_output_shape(self, inp_features_shape):
        return tf.TensorShape((None, self.filters))
