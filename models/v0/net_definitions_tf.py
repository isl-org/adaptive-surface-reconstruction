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
from models.common_tf import SpecialSparseConv, window_poly6
import open3d.ml.tf as ml3d
import tensorflow as tf
from collections import namedtuple

NNSResult = namedtuple(
    "NNSResult",
    ["neighbors_index", "neighbors_distance", "neighbors_row_splits"])


class CConvAggregationBlock(tf.keras.Model):
    """The aggregation block is a single Continuous Convolution.
    
    In addition to the features the block can return the importance information.
    """

    def __init__(self, name, output_channels, return_importance=False):
        super().__init__(name=name)
        self._convs = []
        self._return_importance = return_importance

        conv_params = {
            'kernel_size': [4, 4, 4],
            'coordinate_mapping': 'ball_to_cube_radial',
            'normalize': True,
        }

        def Conv(name, filters, activation, **kwargs):
            conv = ml3d.layers.ContinuousConv(name=name,
                                              filters=output_channels,
                                              activation=activation,
                                              **kwargs)
            self._convs.append((name, conv))
            setattr(self, name, conv)

        activation = tf.keras.activations.relu
        Conv(name='conv1',
             filters=output_channels,
             activation=activation,
             **conv_params)

    def call(self, feats, inp_points, out_points, out_extents, scale_compat,
             nns):
        """Computes the features and optionally the importance.

        Args:
            feats: The point featurs.
            
            inp_points: The input point positions.
            
            out_points: These are the positions of the voxel centers of the
                finest grid.
            
            out_extents: This is the voxel size.

            scale_compat: The scale compatibility between the input point radii 
                and the voxel sizes.
            
            nns: tuple with 
                - neighbors_index: The indices to neighbor points for each voxel.
                - neighbor_row_splits: Defines the start and end of each voxels
                    neighbors.
                - neighbors_distance: The distance to each neighbor normalized with
                    respect to the voxel size.
        """
        neighbors_importance = scale_compat * window_poly6(
            nns.neighbors_distance)
        feats = self.conv1(
            feats,
            inp_points,
            out_points,
            extents=out_extents,
            user_neighbors_index=nns.neighbors_index,
            user_neighbors_row_splits=nns.neighbors_row_splits,
            user_neighbors_importance=neighbors_importance,
        )
        if self._return_importance:
            return feats, neighbors_importance
        else:
            return feats


class SparseConvBlock(tf.keras.Model):
    """The convolution block for the adaptive grid.

    Args:
        input_channels: Number of input channels.

        output_channels: The number of output channels.

        normalized_channels: The number of channels that will be normalized
            with respect to the importance values.
    """

    def __init__(self, name, output_channels, normalized_channels=0):
        super().__init__(name=name)
        self._convs = []
        self.output_channels = output_channels
        self.normalized_channels = normalized_channels

        conv_params = {
            'kernel_size': 55,
        }

        def Conv(name, filters, activation, **kwargs):
            conv = SpecialSparseConv(name=name,
                                     filters=filters,
                                     activation=activation,
                                     **kwargs)
            self._convs.append((name, conv))
            setattr(self, name, conv)

        activation = tf.keras.activations.relu
        if normalized_channels == 'all':
            Conv(name='conv1',
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv2',
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv3',
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv4',
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
        elif normalized_channels and normalized_channels >= output_channels:
            Conv(name='conv1',
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv2',
                 filters=output_channels,
                 activation=activation,
                 normalize=False,
                 **conv_params)
            Conv(name='conv3',
                 filters=output_channels,
                 activation=activation,
                 normalize=False,
                 **conv_params)
            Conv(name='conv4',
                 filters=output_channels,
                 activation=activation,
                 normalize=False,
                 **conv_params)
        elif normalized_channels and normalized_channels < output_channels:
            Conv(name='conv1a',
                 filters=output_channels - normalized_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv1b',
                 filters=normalized_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv2',
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv3',
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv4',
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
        else:
            Conv(name='conv1',
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv2',
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv3',
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv4',
                 filters=output_channels,
                 activation=activation,
                 **conv_params)

    def call(self, feats, points, neighbors, importance=None):
        """Computes the features and optionally the importance if there are
        normalized channels.

        Args:
            feats: Input features.

            neighbors: dict with the neighbor information.

            importance: The per voxel importance value
        """
        if self.normalized_channels == 'all':
            feats1, out_importance = self.conv1(feats,
                                                inp_importance=importance,
                                                **neighbors)
            feats2, _ = self.conv2(feats1,
                                   inp_importance=importance,
                                   **neighbors)
            feats3, _ = self.conv3(feats2,
                                   inp_importance=importance,
                                   **neighbors)
            feats4, _ = self.conv4(feats3,
                                   inp_importance=importance,
                                   **neighbors)
            return feats4, out_importance
        elif self.normalized_channels and self.normalized_channels < self.output_channels:
            feats1a = self.conv1a(feats, **neighbors)
            feats1b, out_importance = self.conv1b(feats,
                                                  inp_importance=importance,
                                                  **neighbors)
            feats1 = tf.concat([feats1a, feats1b], axis=-1)
            feats2 = self.conv2(feats1, **neighbors)
            feats3 = self.conv3(feats2, **neighbors)
            feats4 = self.conv4(feats3, **neighbors)
            return feats4, out_importance

        elif self.normalized_channels:
            feats1, out_importance = self.conv1(feats,
                                                inp_importance=importance,
                                                **neighbors)
            feats2 = self.conv2(feats1, **neighbors)
            feats3 = self.conv3(feats2, **neighbors)
            feats4 = self.conv4(feats3, **neighbors)
            return feats4, out_importance
        else:
            feats1 = self.conv1(feats, **neighbors)
            feats2 = self.conv2(feats1, **neighbors)
            feats3 = self.conv3(feats2, **neighbors)
            feats4 = self.conv4(feats3, **neighbors)
            return feats4


class SparseConvTransitionBlock(tf.keras.Model):
    """The convolution block for transitions between grids (up- and downconvolutions).

    Args:
        input_channels: Number of input channels.

        output_channels: The number of output channels.

        normalized_channels: The number of channels that will be normalized
            with respect to the importance values.
    """

    def __init__(self, name, output_channels, normalized_channels=0):
        super().__init__(name=name)
        self._convs = []
        self.output_channels = output_channels
        self.normalized_channels = normalized_channels

        conv_params = {
            'kernel_size': 9,
            'activation': tf.keras.activations.relu
        }

        def Conv(name, filters, activation, **kwargs):
            conv = SpecialSparseConv(name=name,
                                     filters=filters,
                                     activation=activation,
                                     **kwargs)
            self._convs.append((name, conv))
            setattr(self, name, conv)

        if normalized_channels == 'all' or normalized_channels >= output_channels:
            Conv(name='conv1',
                 filters=output_channels,
                 normalize=True,
                 **conv_params)
        elif normalized_channels and normalized_channels < output_channels:
            Conv(name='conv1a',
                 filters=output_channels - normalized_channels,
                 **conv_params)
            Conv(name='conv1b',
                 filters=normalized_channels,
                 normalize=True,
                 **conv_params)
        else:
            Conv(name='conv1', filters=output_channels, **conv_params)

    def call(self, feats, inp_points, out_points, neighbors, importance=None):
        """Computes the features and optionally the importance if there are
        normalized channels.

        Args:
            feats: Input features.

            neighbors: dict with the neighbor information.

            importance: The per voxel importance value
        """
        if self.normalized_channels == 'all':
            feats1, out_importance = self.conv1(feats,
                                                inp_importance=importance,
                                                **neighbors)
            return feats1, out_importance
        elif self.normalized_channels and self.normalized_channels < self.output_channels:
            feats1a = self.conv1a(feats, **neighbors)
            feats1b, out_importance = self.conv1b(feats,
                                                  inp_importance=importance,
                                                  **neighbors)
            feats1 = tf.concat([feats1a, feats1b], axis=-1)
            return feats1, out_importance
        elif self.normalized_channels:
            feats1, out_importance = self.conv1(feats,
                                                inp_importance=importance,
                                                **neighbors)
            return feats1, out_importance
        else:
            feats1 = self.conv1(feats, **neighbors)
            return feats1


class UNet5(tf.keras.Model):
    """Unet for adaptive grids predicting the signed and unsigned distance field.

    Args:
        channel_div: Reduces the number of channels for each layer.

        with_importance: Adds channels normalized with the importance values.

        normalized_channels: How many channels should be normalized with the importance.

        residual_skip_connection: If True uses a residual connection for the last skip
            connection. If 'all' uses residual connction for every skip connection.
    """
    octree_levels = 5

    def __init__(self,
                 name=None,
                 channel_div=1,
                 with_importance=False,
                 normalized_channels=0,
                 deeper=0,
                 residual_skip_connection=False):
        super().__init__(name=name, autocast=False)
        if not with_importance in (False, True, 'all'):
            raise Exception('invalid value for "with_importance" {}'.format(
                with_importance))
        self.with_importance = with_importance
        self.residual_skip_connection = residual_skip_connection

        def SparseConvTransition(name,
                                 filters,
                                 normalized_channels=0,
                                 **kwargs):
            return SparseConvTransitionBlock(name, filters, normalized_channels)

        d = channel_div
        self.cconv_block_in = CConvAggregationBlock(
            name="cconv_block_in",
            output_channels=32 // d,
            return_importance=with_importance)

        params = {}
        if with_importance:
            params.update({
                'normalized_channels': normalized_channels,
            })
        self.sparseconv_encblock0 = SparseConvBlock(name="sparseconv_encblock0",
                                                    output_channels=64 // d,
                                                    **params)
        self.sparseconv_down1 = SparseConvTransition(name="sparseconv_down1",
                                                     filters=128 // d,
                                                     **params)
        self.sparseconv_encblock1 = SparseConvBlock(name="sparseconv_encblock1",
                                                    output_channels=128 // d,
                                                    **params)
        self.sparseconv_down2 = SparseConvTransition(name="sparseconv_down2",
                                                     filters=256 // d,
                                                     **params)
        self.sparseconv_encblock2 = SparseConvBlock(name="sparseconv_encblock2",
                                                    output_channels=256 // d,
                                                    **params)
        self.sparseconv_down3 = SparseConvTransition(name="sparseconv_down3",
                                                     filters=256 // d,
                                                     **params)
        self.sparseconv_encblock3 = SparseConvBlock(name="sparseconv_encblock3",
                                                    output_channels=256 // d,
                                                    **params)
        self.sparseconv_down4 = SparseConvTransition(name="sparseconv_down4",
                                                     filters=256 // d,
                                                     **params)
        self.sparseconv_encblock4 = SparseConvBlock(name="sparseconv_encblock4",
                                                    output_channels=256 // d,
                                                    **params)

        params = {}
        self.sparseconv_up3 = SparseConvTransition(name="sparseconv_up3",
                                                   filters=256 // d)
        self.sparseconv_decblock3 = SparseConvBlock(name="sparseconv_decblock3",
                                                    output_channels=256 // d,
                                                    **params)
        self.sparseconv_up2 = SparseConvTransition(name="sparseconv_up2",
                                                   filters=256 // d)
        self.sparseconv_decblock2 = SparseConvBlock(name="sparseconv_decblock2",
                                                    output_channels=256 // d,
                                                    **params)
        if self.residual_skip_connection == 'all':
            self.sparseconv_up1 = SparseConvTransition(name="sparseconv_up1",
                                                       filters=128 // d)
        else:
            self.sparseconv_up1 = SparseConvTransition(name="sparseconv_up1",
                                                       filters=256 // d)
        self.sparseconv_decblock1 = SparseConvBlock(name="sparseconv_decblock1",
                                                    output_channels=128 // d,
                                                    **params)
        self.sparseconv_up0 = SparseConvTransition(name="sparseconv_up0",
                                                   filters=64 // d)
        self.sparseconv_decblock0 = SparseConvBlock(name="sparseconv_decblock0",
                                                    output_channels=32 // d,
                                                    **params)

        activation = tf.keras.activations.relu
        self.dense_decoder1 = tf.keras.layers.Dense(32 // d,
                                                    name='dense_decoder1',
                                                    activation=activation,
                                                    use_bias=True)
        self.dense_decoder2 = tf.keras.layers.Dense(32 // d,
                                                    name='dense_decoder2',
                                                    activation=activation,
                                                    use_bias=True)
        self.dense_decoder3 = tf.keras.layers.Dense(2,
                                                    name='dense_decoder3',
                                                    activation=None,
                                                    use_bias=False)

        self._all_layers = []
        self._collect_layers(self.layers)

    def _collect_layers(self, layers):
        for x in layers:
            if hasattr(x, 'layers'):
                self._collect_layers(x.layers)
            else:
                self._all_layers.append(x)

    @tf.function
    def call(self, input_dict):
        """Does a forward pass with aggregation and decode suited for training.
        """

        feats = input_dict['feats']

        feats1 = self.aggregate(input_dict)

        code = self.unet(feats1, input_dict)

        value, shifts_grad = self.decode(input_dict['voxel_shifts0'], code)

        result = {
            'value': value,
            'shifts_grad': shifts_grad,
            # 'code': code,
        }

        debug_info = {}

        return result, debug_info

    @tf.function
    def unet(self, feats1, input_dict, keep_threshold=1):
        """Forward pass through the unet. Excludes aggregation and decode."""
        neighbors = []
        for i in range(5):
            neighbors.append({
                'neighbors_index':
                    input_dict['neighbors_index{}'.format(i)],
                'neighbors_kernel_index':
                    input_dict['neighbors_kernel_index{}'.format(i)],
                'neighbors_row_splits':
                    input_dict['neighbors_row_splits{}'.format(i)],
            })

        neighbors_down = []
        for i in range(4):
            num_points = tf.shape(input_dict['voxel_centers{}'.format(i + 1)],
                                  out_type=tf.int64)[0]
            ans = ml3d.ops.invert_neighbors_list(
                num_points, input_dict['up_neighbors_index{}'.format(i)],
                input_dict['up_neighbors_row_splits{}'.format(i)],
                tf.cast(input_dict['up_neighbors_kernel_index{}'.format(i)],
                        dtype=tf.int32))
            neighbors_down.append({
                'neighbors_index':
                    ans.neighbors_index,
                'neighbors_kernel_index':
                    tf.cast(ans.neighbors_attributes, dtype=tf.int16),
                'neighbors_row_splits':
                    ans.neighbors_row_splits,
            })

        neighbors_up = []
        for i in range(4):
            neighbors_up.append({
                'neighbors_index':
                    input_dict['up_neighbors_index{}'.format(i)],
                'neighbors_kernel_index':
                    input_dict['up_neighbors_kernel_index{}'.format(i)],
                'neighbors_row_splits':
                    input_dict['up_neighbors_row_splits{}'.format(i)],
            })

        if self.with_importance and keep_threshold < 1:
            feats1, importance = feats1
            nonzero_count = tf.math.count_nonzero(importance > 1e-3)
            threshold_idx = tf.cast(tf.cast(nonzero_count, dtype=tf.float32) *
                                    keep_threshold,
                                    dtype=tf.int32)
            threshold_value = tf.sort(importance,
                                      direction='DESCENDING')[threshold_idx]
            feats1 = tf.where((importance < threshold_value)[:, None],
                              tf.zeros_like(feats1), feats1)
            importance = tf.where(importance < threshold_value,
                                  tf.zeros_like(importance), importance)

            feats1 = (feats1, importance)

        if self.with_importance:
            feats1, importance = feats1
        else:
            importance = None
        feats2 = self.sparseconv_encblock0(feats1,
                                           input_dict['voxel_centers0'],
                                           neighbors[0],
                                           importance=importance)
        if self.with_importance == 'all':
            feats2, importance = feats2
            feats3, importance = self.sparseconv_down1(
                feats2,
                input_dict['voxel_centers0'],
                input_dict['voxel_centers1'],
                neighbors_down[0],
                importance=importance)
            feats4, importance = self.sparseconv_encblock1(
                feats3,
                input_dict['voxel_centers1'],
                neighbors[1],
                importance=importance)
            feats5, importance = self.sparseconv_down2(
                feats4,
                input_dict['voxel_centers1'],
                input_dict['voxel_centers2'],
                neighbors_down[1],
                importance=importance)
            feats6, importance = self.sparseconv_encblock2(
                feats5,
                input_dict['voxel_centers2'],
                neighbors[2],
                importance=importance)
            feats7, importance = self.sparseconv_down3(
                feats6,
                input_dict['voxel_centers2'],
                input_dict['voxel_centers3'],
                neighbors_down[2],
                importance=importance)
            feats8, importance = self.sparseconv_encblock3(
                feats7,
                input_dict['voxel_centers3'],
                neighbors[3],
                importance=importance)
            feats9, importance = self.sparseconv_down3(
                feats8,
                input_dict['voxel_centers3'],
                input_dict['voxel_centers4'],
                neighbors_down[3],
                importance=importance)
            feats10, importance = self.sparseconv_encblock4(
                feats9,
                input_dict['voxel_centers4'],
                neighbors[4],
                importance=importance)
        else:
            if self.with_importance:
                feats2, _ = feats2
            feats3 = self.sparseconv_down1(feats2, input_dict['voxel_centers0'],
                                           input_dict['voxel_centers1'],
                                           neighbors_down[0])
            feats4 = self.sparseconv_encblock1(feats3,
                                               input_dict['voxel_centers1'],
                                               neighbors[1])
            feats5 = self.sparseconv_down2(feats4, input_dict['voxel_centers1'],
                                           input_dict['voxel_centers2'],
                                           neighbors_down[1])
            feats6 = self.sparseconv_encblock2(feats5,
                                               input_dict['voxel_centers2'],
                                               neighbors[2])
            feats7 = self.sparseconv_down3(feats6, input_dict['voxel_centers2'],
                                           input_dict['voxel_centers3'],
                                           neighbors_down[2])
            feats8 = self.sparseconv_encblock3(feats7,
                                               input_dict['voxel_centers3'],
                                               neighbors[3])
            feats9 = self.sparseconv_down3(feats8, input_dict['voxel_centers3'],
                                           input_dict['voxel_centers4'],
                                           neighbors_down[3])
            feats10 = self.sparseconv_encblock4(feats9,
                                                input_dict['voxel_centers4'],
                                                neighbors[4])

        feats11 = self.sparseconv_up3(feats10, input_dict['voxel_centers4'],
                                      input_dict['voxel_centers3'],
                                      neighbors_up[3])
        if self.residual_skip_connection == 'all':
            feats12 = feats11 + feats8
        else:
            feats12 = tf.concat([feats11, feats8], axis=-1)
        feats13 = self.sparseconv_decblock3(feats12,
                                            input_dict['voxel_centers3'],
                                            neighbors[3])
        feats14 = self.sparseconv_up2(feats13, input_dict['voxel_centers3'],
                                      input_dict['voxel_centers2'],
                                      neighbors_up[2])
        if self.residual_skip_connection == 'all':
            feats15 = feats14 + feats6
        else:
            feats15 = tf.concat([feats14, feats6], axis=-1)
        feats16 = self.sparseconv_decblock2(feats15,
                                            input_dict['voxel_centers2'],
                                            neighbors[2])
        feats17 = self.sparseconv_up1(feats16, input_dict['voxel_centers2'],
                                      input_dict['voxel_centers1'],
                                      neighbors_up[1])
        if self.residual_skip_connection == 'all':
            feats18 = feats17 + feats4
        else:
            feats18 = tf.concat([feats17, feats4], axis=-1)
        feats19 = self.sparseconv_decblock1(feats18,
                                            input_dict['voxel_centers1'],
                                            neighbors[1])
        feats20 = self.sparseconv_up0(feats19, input_dict['voxel_centers1'],
                                      input_dict['voxel_centers0'],
                                      neighbors_up[0])
        if self.residual_skip_connection:
            feats21 = feats20 + feats2
        else:
            feats21 = tf.concat([feats20, feats2], axis=-1)
        code = self.sparseconv_decblock0(feats21, input_dict['voxel_centers0'],
                                         neighbors[0])

        return code

    @tf.function
    def aggregate(self, input_dict):
        """Aggregation step."""
        feats = input_dict['feats']
        nns = NNSResult(input_dict['aggregation_neighbors_index'],
                        input_dict['aggregation_neighbors_dist'],
                        input_dict['aggregation_row_splits'])
        feats1 = self.cconv_block_in(
            feats,
            input_dict['points'],
            input_dict['voxel_centers0'],
            input_dict['voxel_sizes0'],
            scale_compat=input_dict['aggregation_scale_compat'],
            nns=nns)
        return feats1

    @tf.function
    def decode(self, shifts, code):
        """Decode step and returns the gradient with respect to the shift.
        Args:
            shifts: Positions inside the voxels.
            code: Output features of the unet for each voxel.
        """
        new_code_shape = tf.concat(
            [tf.shape(shifts)[:-1], tf.shape(code)[-1:]], axis=0)
        code = tf.broadcast_to(code, new_code_shape)

        decoder_input = tf.concat([shifts, code], axis=-1)
        feats1 = self.dense_decoder1(decoder_input)
        feats2 = self.dense_decoder2(feats1)
        value = self.dense_decoder3(feats2)
        shifts_grad = tf.gradients(value[..., 0], shifts)[0]
        return tf.dtypes.cast(value, dtype=tf.float32), tf.dtypes.cast(
            shifts_grad, dtype=tf.float32)
