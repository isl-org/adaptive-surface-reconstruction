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
from models.common_torch import SpecialSparseConv, window_poly6
import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d


@torch.jit.script
def invert_neighbors_list_script(num_points_tensor, neighbors_index,
                                 neighbors_row_splits, neighbors_kernel_index):
    """Converts the neighbor information.

    This function is used to convert the connectivity info for upconvolutions
    to connectivity info for downconvolutions.
    """
    ans = ml3d.ops.invert_neighbors_list(
        num_points_tensor.shape[0],
        neighbors_index,
        neighbors_row_splits,
        neighbors_kernel_index,
    )
    return ans


class CConvAggregationBlock(torch.nn.Module):
    """The aggregation block is a single Continuous Convolution.
    
    In addition to the features the block can return the importance information.
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 return_importance=False):
        super().__init__()
        self._return_importance = return_importance
        self.output_channels = output_channels

        conv_params = {
            'kernel_size': [4, 4, 4],
            'coordinate_mapping': 'ball_to_cube_radial',
            'normalize': True,
        }

        def Conv(name, filters, activation, **kwargs):
            conv = ml3d.layers.ContinuousConv(in_channels=input_channels,
                                              filters=output_channels,
                                              activation=activation,
                                              **kwargs)
            setattr(self, name, conv)

        activation = F.relu
        Conv(name='conv1',
             filters=output_channels,
             activation=activation,
             **conv_params)

    def forward(
        self,
        feats,
        inp_points,
        out_points,
        out_extents,
        scale_compat,
        neighbors_index,
        neighbors_row_splits,
        neighbors_distance,
    ):
        """Computes the features and optionally the importance.

        Args:
            feats: The point featurs.
            
            inp_points: The input point positions.
            
            out_points: These are the positions of the voxel centers of the
                finest grid.
            
            out_extents: This is the voxel size.

            scale_compat: The scale compatibility between the input point radii 
                and the voxel sizes.
            
            neighbors_index: The indices to neighbor points for each voxel.

            neighbor_row_splits: Defines the start and end of each voxels
                neighbors.

            neighbors_distance: The distance to each neighbor normalized with
                respect to the voxel size.
        """

        neighbors_importance = scale_compat * window_poly6(neighbors_distance)
        feats = self.conv1(
            feats,
            inp_points,
            out_points,
            extents=out_extents,
            user_neighbors_index=neighbors_index,
            user_neighbors_row_splits=neighbors_row_splits,
            user_neighbors_importance=neighbors_importance,
        )
        if self._return_importance:
            return feats, neighbors_importance
        else:
            return feats


class SparseConvBlock(torch.nn.Module):
    """The convolution block for the adaptive grid.

    Args:
        input_channels: Number of input channels.

        output_channels: The number of output channels.

        normalized_channels: The number of channels that will be normalized
            with respect to the importance values.
    """

    def __init__(self, input_channels, output_channels, normalized_channels=0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.normalized_channels = normalized_channels

        conv_params = {
            'kernel_size': 55,
        }

        def Conv(name, in_channels, filters, activation, **kwargs):
            conv = SpecialSparseConv(name=name,
                                     in_channels=in_channels,
                                     filters=filters,
                                     activation=activation,
                                     **kwargs)
            setattr(self, name, conv)

        activation = F.relu
        if normalized_channels == 'all':
            Conv(name='conv1',
                 in_channels=input_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv2',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv3',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv4',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
        elif normalized_channels and normalized_channels >= output_channels:
            Conv(name='conv1',
                 in_channels=input_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv2',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=False,
                 **conv_params)
            Conv(name='conv3',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=False,
                 **conv_params)
            Conv(name='conv4',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 normalize=False,
                 **conv_params)
        elif normalized_channels and normalized_channels < output_channels:
            Conv(name='conv1a',
                 in_channels=input_channels,
                 filters=output_channels - normalized_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv1b',
                 in_channels=input_channels,
                 filters=normalized_channels,
                 activation=activation,
                 normalize=True,
                 **conv_params)
            Conv(name='conv2',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv3',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv4',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
        else:
            Conv(name='conv1',
                 in_channels=input_channels,
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv2',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv3',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 **conv_params)
            Conv(name='conv4',
                 in_channels=output_channels,
                 filters=output_channels,
                 activation=activation,
                 **conv_params)

    def forward(self, feats, neighbors, importance=None):
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
            feats1a, _ = self.conv1a(feats, **neighbors)
            feats1b, out_importance = self.conv1b(feats,
                                                  inp_importance=importance,
                                                  **neighbors)
            feats1 = torch.cat([feats1a, feats1b], axis=-1)
            feats2, _ = self.conv2(feats1, **neighbors)
            feats3, _ = self.conv3(feats2, **neighbors)
            feats4, _ = self.conv4(feats3, **neighbors)
            return feats4, out_importance

        elif self.normalized_channels:
            feats1, out_importance = self.conv1(feats,
                                                inp_importance=importance,
                                                **neighbors)
            feats2, _ = self.conv2(feats1, **neighbors)
            feats3, _ = self.conv3(feats2, **neighbors)
            feats4, _ = self.conv4(feats3, **neighbors)
            return feats4, out_importance
        else:
            feats1, _ = self.conv1(feats, **neighbors)
            feats2, _ = self.conv2(feats1, **neighbors)
            feats3, _ = self.conv3(feats2, **neighbors)
            feats4, _ = self.conv4(feats3, **neighbors)
            return feats4


class SparseConvTransitionBlock(torch.nn.Module):
    """The convolution block for transitions between grids (up- and downconvolutions).

    Args:
        input_channels: Number of input channels.

        output_channels: The number of output channels.

        normalized_channels: The number of channels that will be normalized
            with respect to the importance values.
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 normalized_channels=0,
                 **kwargs):
        super().__init__()
        self.output_channels = output_channels
        self.normalized_channels = normalized_channels

        conv_params = {'kernel_size': 9, 'activation': F.relu}

        def Conv(name, in_channels, filters, activation, **kwargs):
            conv = SpecialSparseConv(in_channels=in_channels,
                                     filters=filters,
                                     activation=activation,
                                     **kwargs)
            setattr(self, name, conv)

        if normalized_channels == 'all' or normalized_channels >= output_channels:
            Conv(name='conv1',
                 in_channels=input_channels,
                 filters=output_channels,
                 normalize=True,
                 **conv_params)
        elif normalized_channels and normalized_channels < output_channels:
            Conv(name='conv1a',
                 in_channels=input_channels,
                 filters=output_channels - normalized_channels,
                 **conv_params)
            Conv(name='conv1b',
                 in_channels=input_channels,
                 filters=normalized_channels,
                 normalize=True,
                 **conv_params)
        else:
            Conv(name='conv1',
                 in_channels=input_channels,
                 filters=output_channels,
                 **conv_params)

    def forward(self, feats, neighbors, importance=None):
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
            feats1a, _ = self.conv1a(feats, **neighbors)
            feats1b, out_importance = self.conv1b(feats,
                                                  inp_importance=importance,
                                                  **neighbors)
            feats1 = torch.cat([feats1a, feats1b], axis=-1)
            return feats1, out_importance
        elif self.normalized_channels:
            feats1, out_importance = self.conv1(feats,
                                                inp_importance=importance,
                                                **neighbors)
            return feats1, out_importance
        else:
            feats1, _ = self.conv1(feats, **neighbors)
            return feats1


class UNet5(torch.nn.Module):
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
                 channel_div=1,
                 with_importance=False,
                 normalized_channels=0,
                 residual_skip_connection=False):
        super().__init__()
        if not with_importance in (False, True, 'all'):
            raise Exception('invalid value for "with_importance" {}'.format(
                with_importance))
        self.with_importance = with_importance
        self.residual_skip_connection = residual_skip_connection

        d = channel_div
        self.cconv_block_in = CConvAggregationBlock(
            input_channels=4,
            output_channels=32 // d,
            return_importance=with_importance)

        params = {}
        if with_importance:
            params.update({
                'normalized_channels': normalized_channels,
            })
        self.sparseconv_encblock0 = SparseConvBlock(
            input_channels=self.cconv_block_in.output_channels,
            output_channels=64 // d,
            **params)
        self.sparseconv_down1 = SparseConvTransitionBlock(
            input_channels=self.sparseconv_encblock0.output_channels,
            output_channels=128 // d,
            **params)
        self.sparseconv_encblock1 = SparseConvBlock(
            input_channels=self.sparseconv_down1.output_channels,
            output_channels=128 // d,
            **params)
        self.sparseconv_down2 = SparseConvTransitionBlock(
            input_channels=self.sparseconv_encblock1.output_channels,
            output_channels=256 // d,
            **params)
        self.sparseconv_encblock2 = SparseConvBlock(
            input_channels=self.sparseconv_down2.output_channels,
            output_channels=256 // d,
            **params)
        self.sparseconv_down3 = SparseConvTransitionBlock(
            input_channels=self.sparseconv_encblock2.output_channels,
            output_channels=256 // d,
            **params)
        self.sparseconv_encblock3 = SparseConvBlock(
            input_channels=self.sparseconv_down3.output_channels,
            output_channels=256 // d,
            **params)
        # self.sparseconv_down4 = SparseConvTransitionBlock(input_channels=self.sparseconv_encblock3.output_channels, output_channels=256//d, **params)
        self.sparseconv_encblock4 = SparseConvBlock(
            input_channels=self.sparseconv_down3.output_channels,
            output_channels=256 // d,
            **params)

        params = {}
        self.sparseconv_up3 = SparseConvTransitionBlock(
            input_channels=self.sparseconv_encblock4.output_channels,
            output_channels=256 // d)
        if self.residual_skip_connection == 'all':
            input_channels = self.sparseconv_up3.output_channels
        else:
            input_channels = self.sparseconv_up3.output_channels + self.sparseconv_encblock3.output_channels
        self.sparseconv_decblock3 = SparseConvBlock(
            input_channels=input_channels, output_channels=256 // d, **params)
        self.sparseconv_up2 = SparseConvTransitionBlock(
            input_channels=self.sparseconv_decblock3.output_channels,
            output_channels=256 // d)
        if self.residual_skip_connection == 'all':
            input_channels = self.sparseconv_up2.output_channels
        else:
            input_channels = self.sparseconv_up2.output_channels + self.sparseconv_encblock2.output_channels
        self.sparseconv_decblock2 = SparseConvBlock(
            input_channels=input_channels, output_channels=256 // d, **params)
        if self.residual_skip_connection == 'all':
            self.sparseconv_up1 = SparseConvTransitionBlock(
                input_channels=self.sparseconv_decblock2.output_channels,
                output_channels=128 // d)
        else:
            self.sparseconv_up1 = SparseConvTransitionBlock(
                input_channels=self.sparseconv_decblock2.output_channels,
                output_channels=256 // d)
        if self.residual_skip_connection == 'all':
            input_channels = self.sparseconv_up1.output_channels
        else:
            input_channels = self.sparseconv_up1.output_channels + self.sparseconv_encblock1.output_channels
        self.sparseconv_decblock1 = SparseConvBlock(
            input_channels=input_channels, output_channels=128 // d, **params)
        self.sparseconv_up0 = SparseConvTransitionBlock(
            input_channels=self.sparseconv_decblock1.output_channels,
            output_channels=64 // d)
        self.sparseconv_decblock0 = SparseConvBlock(
            input_channels=self.sparseconv_up0.output_channels,
            output_channels=32 // d,
            **params)

        self.dense_decoder1 = torch.nn.Linear(in_features=32 + 3,
                                              out_features=32 // d,
                                              bias=True)
        self.dense_decoder2 = torch.nn.Linear(in_features=32,
                                              out_features=32 // d,
                                              bias=True)
        self.dense_decoder3 = torch.nn.Linear(in_features=32,
                                              out_features=2,
                                              bias=False)

    def forward(self, input_dict):
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

    def unet(self, feats1, input_dict):
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
            ans = invert_neighbors_list_script(
                input_dict['voxel_centers{}'.format(i + 1)],
                input_dict['up_neighbors_index{}'.format(i)],
                input_dict['up_neighbors_row_splits{}'.format(i)],
                input_dict['up_neighbors_kernel_index{}'.format(i)])
            neighbors_down.append({
                'neighbors_index': ans.neighbors_index,
                'neighbors_kernel_index': ans.neighbors_attributes,
                'neighbors_row_splits': ans.neighbors_row_splits,
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

        if self.with_importance:
            feats1, importance = feats1
        else:
            importance = None
        feats2 = self.sparseconv_encblock0(feats1,
                                           neighbors[0],
                                           importance=importance)
        if self.with_importance == 'all':
            feats2, importance = feats2
            feats3, importance = self.sparseconv_down1(feats2,
                                                       neighbors_down[0],
                                                       importance=importance)
            feats4, importance = self.sparseconv_encblock1(
                feats3, neighbors[1], importance=importance)
            feats5, importance = self.sparseconv_down2(feats4,
                                                       neighbors_down[1],
                                                       importance=importance)
            feats6, importance = self.sparseconv_encblock2(
                feats5, neighbors[2], importance=importance)
            feats7, importance = self.sparseconv_down3(feats6,
                                                       neighbors_down[2],
                                                       importance=importance)
            feats8, importance = self.sparseconv_encblock3(
                feats7, neighbors[3], importance=importance)
            feats9, importance = self.sparseconv_down3(feats8,
                                                       neighbors_down[3],
                                                       importance=importance)
            feats10, importance = self.sparseconv_encblock4(
                feats9, neighbors[4], importance=importance)
        else:
            if self.with_importance:
                feats2, _ = feats2
            feats3 = self.sparseconv_down1(feats2, neighbors_down[0])
            feats4 = self.sparseconv_encblock1(feats3, neighbors[1])
            feats5 = self.sparseconv_down2(feats4, neighbors_down[1])
            feats6 = self.sparseconv_encblock2(feats5, neighbors[2])
            feats7 = self.sparseconv_down3(feats6, neighbors_down[2])
            feats8 = self.sparseconv_encblock3(feats7, neighbors[3])
            feats9 = self.sparseconv_down3(feats8, neighbors_down[3])
            feats10 = self.sparseconv_encblock4(feats9, neighbors[4])

        feats11 = self.sparseconv_up3(feats10, neighbors_up[3])
        if self.residual_skip_connection == 'all':
            feats12 = feats11 + feats8
        else:
            feats12 = torch.cat([feats11, feats8], axis=-1)
        feats13 = self.sparseconv_decblock3(feats12, neighbors[3])
        feats14 = self.sparseconv_up2(feats13, neighbors_up[2])
        if self.residual_skip_connection == 'all':
            feats15 = feats14 + feats6
        else:
            feats15 = torch.cat([feats14, feats6], axis=-1)
        feats16 = self.sparseconv_decblock2(feats15, neighbors[2])
        feats17 = self.sparseconv_up1(feats16, neighbors_up[1])
        if self.residual_skip_connection == 'all':
            feats18 = feats17 + feats4
        else:
            feats18 = torch.cat([feats17, feats4], axis=-1)
        feats19 = self.sparseconv_decblock1(feats18, neighbors[1])
        feats20 = self.sparseconv_up0(feats19, neighbors_up[0])
        if self.residual_skip_connection:
            feats21 = feats20 + feats2
        else:
            feats21 = torch.cat([feats20, feats2], axis=-1)
        code = self.sparseconv_decblock0(feats21, neighbors[0])

        return code

    def aggregate(self, input_dict):
        """Aggregation step."""
        feats = input_dict['feats']
        feats1 = self.cconv_block_in(
            feats,
            input_dict['points'],
            input_dict['voxel_centers0'],
            input_dict['voxel_sizes0'],
            scale_compat=input_dict['aggregation_scale_compat'],
            neighbors_index=input_dict['aggregation_neighbors_index'],
            neighbors_row_splits=input_dict['aggregation_row_splits'],
            neighbors_distance=input_dict['aggregation_neighbors_dist'],
        )
        return feats1

    def decode(self, shifts, code):
        """Decode step.
        Args:
            shifts: Positions inside the voxels.
            code: Output features of the unet for each voxel.
        """
        decoder_input = torch.cat([shifts, code], axis=-1)
        feats1 = F.relu(self.dense_decoder1(decoder_input))
        feats2 = F.relu(self.dense_decoder2(feats1))
        value = self.dense_decoder3(feats2)

        return value

    def decode_with_gradient(self, shifts, code):
        """Decode step and returns the gradient with respect to the shift.
        Args:
            shifts: Positions inside the voxels.
            code: Output features of the unet for each voxel.
        """
        decoder_input = torch.cat([shifts, code], axis=-1)
        feats1 = F.relu(self.dense_decoder1(decoder_input))
        feats2 = F.relu(self.dense_decoder2(feats1))
        value = self.dense_decoder3(feats2)

        # compute gradient with respect to shifts
        z3 = torch.ones(
            (shifts.shape[0], 1)) * self.dense_decoder3.weight[:1, :]
        z3[feats2 <= 0] = 0
        z2 = torch.matmul(z3, self.dense_decoder2.weight)
        z2[feats1 <= 0] = 0
        z1 = torch.matmul(z2, self.dense_decoder1.weight)
        return value, z1[:, :3]
