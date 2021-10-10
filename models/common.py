#
# Copyright 2021 Intel (Autonomous Agents Lab)
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
import numpy as np


def compute_scale_compatibility(query_scale,
                                point_scale,
                                neighbors_index,
                                neighbors_row_splits,
                                gamma=2):
    """Computes the scale compatibility.
    
    Args:
        query_scale: 1D array with the scales for each query point.
        point_scale: 1D array with the scales for each point.
        neighbors_index: 1D array with the indices into point_scale.
        neighbors_row_splits: 1D array defining the start and end of each row of neighbors_index.
        gamma: exponent.
    
    Returns:
        Returns min(s_i,s_j)^gamma/max(s_i,s_j) where s_i is the scale of the query point 
        and s_j are the scales of the neighbor points for each pair. The returned array has
        the same length and structure as neighbors_index.
    """
    row_lengths = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
    query_index = np.repeat(np.arange(len(neighbors_row_splits) - 1),
                            row_lengths)
    a = query_scale[query_index]
    b = point_scale[neighbors_index]
    ans = (np.minimum(a, b) / np.maximum(a, b))**gamma
    return ans
