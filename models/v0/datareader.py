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
import os
import open3d as o3d
import numpy as np
from glob import glob
import dataflow
import numpy as np
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
from models.common import compute_scale_compatibility
import adaptivesurfacereconstruction as asr

_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '../../datasets/t10k')
_dataset_path_synthetic_colmap = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../datasets/synthetic_colmap_data')

val_files = sorted(glob(os.path.join(_dataset_path, 'valid', '*.msgpack.zst')))
train_files = [
    x
    for x in sorted(glob(os.path.join(_dataset_path, 'train', '*.msgpack.zst')))
    if not x in val_files
]

val_files_synthetic_colmap = sorted(
    glob(os.path.join(_dataset_path_synthetic_colmap, 'valid',
                      '*.msgpack.zst')))
train_files_synthetic_colmap = [
    x for x in sorted(
        glob(
            os.path.join(_dataset_path_synthetic_colmap, 'train',
                         '*.msgpack.zst')))
    if not x in val_files_synthetic_colmap
]


def create_camera_extrinsics(position, lookat, up=None):
    """Converts to R,t camera pose

    position: np.array
        The camera position

    lookat: np.array
        The point the camera is looking at

    up: np.array
        The up vector, usually [0,1,0], which is the default value
    """
    if up is None:
        up = np.array([0, 1, 0]).astype(np.float64)
    R = np.eye(3).astype(np.float64)
    R[1] = up / np.linalg.norm(up)
    R[2] = lookat - position
    R[2] /= np.linalg.norm(R[2, :])
    R[0] = np.cross(R[1], R[2])
    R[0] /= np.linalg.norm(R[0, :])
    R[1] = np.cross(R[2], R[0])
    t = -R.dot(position).astype(np.float64)
    return {'R': R, 't': t}


def create_camera_intrinsics(field_of_view,
                             width,
                             height,
                             normalized_principal_point=None):
    """Create the camera matrix K with the intrinsic parameters
    
    field_of_view: scalar or 2-tuple
        Horizontal field of view in degree
        
    width: scalar
        Image width
    height: scalar
        Image height
    
    normalized_principal_point: scalar or 2-tuple
        The default value is the image center
    """
    if normalized_principal_point is None:
        normalized_principal_point = 0.5

    focal_length = 0.5 * np.array([width]) / np.tan(
        np.deg2rad(0.5 * field_of_view))
    focal_length = np.broadcast_to(focal_length, [2])
    cxy = np.broadcast_to(normalized_principal_point, [2]) * np.array(
        [width, height])
    K = np.array([[focal_length[0], 0, cxy[0]], [0, focal_length[1], cxy[1]],
                  [0, 0, 1]])
    return {'K': K, 'width': width, 'height': height}


def create_rays_for_camera(intrinsics, extrinsics):
    """Create rays the specified intrinsics and extrinsics.
    
    The rays are not normalized and can be used to compute a depth map
    with depth values stored as the camera z value (perpendicular distance).
    
    intrinsics: dict as returned by create_camera_intrinsics 
                with keys 'K', 'width' and 'height'.
    
    extrinsics: dict as returned by create_camera_extrinsics
                with keys 'R' and 't'.
    """
    w, h = intrinsics['width'], intrinsics['height']
    inv_K = np.linalg.inv(intrinsics['K'])
    R = extrinsics['R']
    t = extrinsics['t']
    C = -R.T.dot(t)

    # create the homogeneous coordinates for each pixel
    X, Y = np.meshgrid(np.arange(0, w) + 0.5, np.arange(0, h) + 0.5)
    im_coords = np.stack([X, Y, np.ones_like(X)], axis=-1)
    ray_dirs = inv_K @ im_coords.reshape((-1, 3, 1))
    ray_dirs = (R.T @ ray_dirs).reshape(im_coords.shape)
    origins = np.broadcast_to(C, ray_dirs.shape)
    rays = np.concatenate([origins, ray_dirs], axis=-1)
    return rays


class BBox:
    """Simple bounding box class to make sure models don't overlap."""

    def __init__(self, bbmin, bbmax):
        self.bbmin_original = bbmin
        self.translation = np.zeros_like(bbmin)
        self.size = bbmax - bbmin

    @property
    def bbmin(self):
        return self.bbmin_original + self.translation

    @property
    def bbmax(self):
        return self.bbmin + self.size

    def is_point_inside(self, points):
        return ((points < self.bbmax) & (points > self.bbmin)).all(axis=1)

    def intersects(self, bbox):
        dim = self.bbmin.shape[0]
        for d in range(dim):
            if self.bbmax[d] < bbox.bbmin[d] or self.bbmin[d] > bbox.bbmax[d]:
                return False
        return True

    @staticmethod
    def join(bboxes):
        bbmin = np.min([bb.bbmin for bb in bboxes], axis=0)
        bbmax = np.max([bb.bbmax for bb in bboxes], axis=0)
        return BBox(bbmin, bbmax)

    def resolve_intersection_with(self, bbox, scale_shift=1.02):
        if not self.intersects(bbox):
            return
        dim = self.bbmin.shape[0]
        shift_vecs = np.zeros(shape=(2 * dim, dim))
        for i in range(dim):
            shift_vecs[2 * i + 0, i] = bbox.bbmax[i] - self.bbmin[i]
            shift_vecs[2 * i + 1, i] = bbox.bbmin[i] - self.bbmax[i]
        idx = np.argmin(np.abs(np.sum(shift_vecs, axis=1)))
        shift_vec = shift_vecs[idx]
        self.translation += shift_vec * scale_shift


def _compute_inlier_and_radii(points, knn=24):
    idx = asr.KDTree(points)
    point_radii = idx.compute_k_radius(knn)
    inlier = idx.compute_inlier(point_radii, k=knn)
    return inlier, point_radii


def _compute_inlier_from_density(points, radii, density_percentile_threshold):
    idx = asr.KDTree(points)
    rn = idx.compute_radius_neighbors(radii)
    inlier = rn > np.percentile(rn, density_percentile_threshold)
    return inlier


def preprocess_input_points(points,
                            point_attributes=None,
                            num_levels=3,
                            max_depth=21,
                            radius_knn=24,
                            point_scale_factor=1,
                            density_percentile_threshold=10,
                            recompute_k_radius=0):
    input_data = {'points': points}
    if not point_attributes is None:
        input_data.update(point_attributes)

    if 'value' in point_attributes:
        input_data['radii'] = input_data['value'] * point_scale_factor
        del input_data['value']
        inlier = _compute_inlier_from_density(points, input_data['radii'],
                                              density_percentile_threshold)
    else:
        inlier, point_radii = _compute_inlier_and_radii(points, radius_knn)
        input_data['radii'] = point_radii

    for k in input_data:
        input_data[k] = input_data[k][inlier]

    if recompute_k_radius:
        idx = asr.KDTree(input_data['points'])
        input_data['radii'] = idx.compute_k_radius(recompute_k_radius)

    # generate octree
    margin = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    bb_min = np.min(input_data['points'], axis=0) - margin
    bb_max = np.max(input_data['points'], axis=0) + margin
    octree = asr.create_octree(input_data['points'],
                               input_data['radii'],
                               bb_min,
                               bb_max,
                               radius_scale=1,
                               grow_steps=0,
                               max_depth=max_depth)

    octree_levels = asr.create_grids_from_octree(octree,
                                                 num_levels=num_levels,
                                                 voxel_info_all_levels=True)
    for i, ol in enumerate(octree_levels):
        for k, v in ol.items():
            input_data[k + str(i)] = v

    return input_data, octree


class PointCloudReconstructionDataFlow(dataflow.RNGDataFlow):
    """Class for generating pairs of point clouds and distance fields for training.

    Args:
        files: A list of data files
        num_points_min: The minimum number of points to return for each batch item
        num_points_max: The maximum number of points to return for each batch item
        num_meshes_min: The minimum number of meshes to use
        num_meshes_max: The maximum number of meshes to use
        random_rotation: If True randomly rotate the data
        shuffle: If true shuffles meshes within one data file
        convert_uint64: If True convert uint64 to int64
        single_example: Generate a single example for debugging
        random_sign: Can be a boolean value or a list like [1,1,1,-1] from which the sign is randomly selected
        random_crop: Probability for cropping a mesh
        octree_levels: The number grid levels to generate. This should match the network architecture
        noise_strength: Use 'auto'
        exclude_boundary: Exclude the boundary from the loss by reducing the weight
        shift_range: The shift range to use for the distance decode
        num_samples_per_voxel: How many samples per voxel to generate for the decoding
    """

    def __init__(
        self,
        files,
        num_points_min=10000,
        num_points_max=20000,
        num_meshes_min=1,
        num_meshes_max=1,
        random_rotation=False,
        shuffle=False,
        convert_uint64=False,
        single_example=False,
        random_sign=False,
        random_crop=0,
        octree_levels=3,
        noise_strength='auto',
        exclude_boundary=False,
        shift_range=1.5,
        num_samples_per_voxel=1,
    ):
        assert len(files)
        self.files = files
        self.num_points_min = num_points_min
        self.num_points_max = num_points_max
        self.num_meshes_min = num_meshes_min
        self.num_meshes_max = num_meshes_max
        self.random_rotation = random_rotation
        self.shuffle = shuffle
        self.convert_uint64 = convert_uint64
        self.single_example = single_example
        self.single_example_data = None
        self.random_sign = random_sign
        self.random_crop = random_crop
        self.octree_levels = octree_levels
        self.noise_strength = noise_strength
        self.exclude_boundary = exclude_boundary
        self.shift_range = shift_range
        self.num_samples_per_voxel = num_samples_per_voxel
        self.raycast_nthreads = 1

    def _random_rotation_matrix(self, dtype=None):
        """Creates a random rotation matrix with specified type

        dtype: output dtype. Default is np.float32
        """
        from scipy.spatial.transform import Rotation

        if dtype is None:
            dtype = np.float32

        r = Rotation.random(random_state=self.rng)
        rand_R = r.as_matrix()
        return rand_R.astype(dtype)

    def _create_holes(self, points, hole_sizes):
        from scipy.spatial import cKDTree
        param_knn = 32
        remove_weight = 8  # higher means less points

        tree = cKDTree(points)

        knn = tree.query(points, k=param_knn)
        point_radii = np.max(knn[0], axis=-1)

        holes = np.full(point_radii.shape, False)
        for hole_size in hole_sizes:

            center_index = self.rng.randint(0, points.shape[0])
            center = points[center_index]
            r = point_radii[center_index] * hole_size
            nn_idx = np.array(tree.query_ball_point(center, r, p=1))
            nn_dist = np.linalg.norm(points[nn_idx] - center, ord=1, axis=-1)

            tmp = self.rng.rand(nn_idx.shape[0]) * r
            remove = nn_dist < remove_weight * tmp
            holes[nn_idx[remove]] = True
        return holes

    def _combine_items(self, items):
        if len(items) == 1:
            return items[0]

        bboxes = []
        for item in items:
            verts = item['vertices']
            if self.random_rotation:
                rand_R = self._random_rotation_matrix()
                verts = np.matmul(item['vertices'], rand_R)
                item['vertices'] = verts
            bbox = BBox(np.min(verts, axis=0), np.max(verts, axis=0))
            bbox.translation = self.rng.rand(3) - 0.5
            bbox.translation /= np.linalg.norm(bbox.translation)
            bbox.translation *= 0.6 * np.linalg.norm(bbox.size)
            bboxes.append(bbox)

        combined_bbox = BBox(bboxes[0].bbmin, bboxes[0].bbmax)
        for i in range(1, len(bboxes)):
            bboxes[i].resolve_intersection_with(combined_bbox)
            combined_bbox = BBox.join([combined_bbox, bboxes[i]])

        center_shift = -(combined_bbox.bbmin + 0.5 * combined_bbox.size)
        combined_bbox.translation += center_shift
        for bbox in bboxes:
            bbox.translation += center_shift

        for item, bbox in zip(items, bboxes):
            item['vertices'] = item['vertices'] + bbox.translation

        # now combine the meshes
        result = {
            'vertices': np.concatenate([x['vertices'] for x in items], axis=0),
            'triangles': []
        }
        offset = 0
        for item in items:
            result['triangles'].append(item['triangles'] + offset)
            offset += item['vertices'].shape[0]
        result['triangles'] = np.concatenate(result['triangles'], axis=0)

        return result

    def _clip_mesh(self, vertices, triangles, mesh_id=None):
        bb_min = vertices.min(axis=0)
        bb_max = vertices.max(axis=0)

        a = self.rng.uniform(0.4, 0.6, size=(3,))
        clip_center = a * bb_min + (1 - a) * bb_max
        clip_normal = self.rng.uniform(-1, 1, size=(3,))
        clip_normal /= np.linalg.norm(clip_normal)

        keep_verts = np.sum(
            clip_normal[np.newaxis, :] * (vertices - clip_center), axis=-1) > 0
        keep_tris = keep_verts[triangles].all(axis=-1)
        if np.count_nonzero(keep_tris) < 0.2 * triangles.shape[0]:
            return vertices, triangles

        new_tris = triangles[keep_tris]
        vidx_map = np.cumsum(keep_verts) - 1
        new_tris = vidx_map[new_tris]
        new_verts = vertices[keep_verts]

        vertices = new_verts
        triangles = new_tris
        return vertices, triangles

    def _generate_sample_point_cloud(self,
                                     vertices,
                                     triangles,
                                     noise=None,
                                     clipped_mesh=False,
                                     pointcloud=None,
                                     **kwargs):
        # parameters
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.uint32)

        if self.random_rotation:
            rand_R = self._random_rotation_matrix()
            vertices = np.matmul(vertices, rand_R)

        max_num_points = 500000

        if pointcloud is None:
            num_cams = self.rng.randint(10, 50)
            num_cams = 50
            img_shape = self.rng.randint(10, 14) * np.array([15, 20])
            height_px, width_px = img_shape

            if noise is None:
                noise = self.rng.uniform(0, 1.5)
            elif isinstance(noise, (tuple, list)):
                noise = self.rng.uniform(*noise)
            else:
                pass
            depth_noise_strength = 0.001 * noise
            normal_noise_strength = 0.1 * noise
            origin_noise_strength = 1 * noise
            dir_noise_strength = 0.4 * noise
            normal_outlier_prob = 0.02 * noise
            outlier_ratio = self.rng.uniform(0, 1) * 0.05 * noise
            dead_pixel_prob = 0.05
            misalign_prob = 0.05
            misalign_strength = self.rng.uniform(0, 0.1)

            result_data = []

            # create cameras
            max_vertex_radius = np.linalg.norm(vertices, axis=-1).max()
            camera_positions = []
            min_radius = 1.2 * max_vertex_radius
            max_radius = 5

            cam_z_loc = self.rng.uniform(-0.5 * min_radius, 0.5 * min_radius)
            cam_z_scale = self.rng.uniform(0.2, 2)
            while len(camera_positions) < num_cams:
                p = self.rng.uniform(-max_radius, max_radius, size=(3,))
                p[2] = self.rng.normal(loc=cam_z_loc, scale=cam_z_scale)
                r = np.linalg.norm(p)
                if r > min_radius and r < max_radius:
                    camera_positions.append(p)

            rand_R_cam = self._random_rotation_matrix()
            camera_positions = [
                np.matmul(p[np.newaxis, :], rand_R_cam)[0]
                for p in camera_positions
            ]

            camera_lookats = []
            while len(camera_lookats) < num_cams:
                p = self.rng.uniform(-min_radius, min_radius, size=(3,))
                r = np.linalg.norm(p)
                if r > min_radius and r < max_radius:
                    camera_lookats.append(p)

            camera_upvectors = self.rng.rand(num_cams, 3) - 0.5
            camera_upvectors /= np.linalg.norm(camera_upvectors,
                                               axis=-1,
                                               keepdims=True)

            cam_extrinsics = []
            for i in range(num_cams):
                cam_ex = create_camera_extrinsics(camera_positions[i],
                                                  camera_lookats[i],
                                                  camera_upvectors[i])
                cam_extrinsics.append(cam_ex)

            cam_intrinsics = [
                create_camera_intrinsics(self.rng.uniform(30, 90), width_px,
                                         height_px) for i in range(num_cams)
            ]

            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(vertices, triangles)

            output_data = {
                'points': [],
                'normals': [],
                'campos': [],
            }

            for cam_i in range(len(cam_intrinsics)):
                cam_in = cam_intrinsics[cam_i]
                cam_ex = cam_extrinsics[cam_i]
                K = cam_in['K']

                rays = create_rays_for_camera(cam_in, cam_ex).astype(np.float32)
                # keep a copy
                rays_original = rays.copy()

                origin_noise_scale = origin_noise_strength / K[0, 0]
                dir_noise_scale = dir_noise_strength / K[0, 0]

                ray_noise_origin = self.rng.laplace(scale=origin_noise_scale,
                                                    size=rays.shape[:-1] + (3,))
                ray_noise_dir = self.rng.laplace(scale=dir_noise_scale,
                                                 size=rays.shape[:-1] + (3,))

                ray_noise = np.concatenate([ray_noise_origin, ray_noise_dir],
                                           axis=-1)
                rays += ray_noise

                hits = {
                    k: v.numpy() for k, v in scene.cast_rays(
                        rays, nthreads=self.raycast_nthreads).items()
                }
                hits_clean = {
                    k: v.numpy() for k, v in scene.cast_rays(
                        rays_original, nthreads=self.raycast_nthreads).items()
                }

                depth_im = hits['t_hit']
                depth_im_clean = hits_clean['t_hit']
                normal_im = hits['primitive_normals']

                depth_im[depth_im == np.inf] = 0
                depth_im_clean[depth_im_clean == np.inf] = 0
                points_clean = rays_original[:, :, :
                                             3] + rays_original[:, :,
                                                                3:] * depth_im_clean[:, :,
                                                                                     np
                                                                                     .
                                                                                     newaxis]
                camvec = -rays_original[:, :, 3:].copy()
                camvec /= np.linalg.norm(camvec, axis=-1, keepdims=True)

                surf_angle_noise_factor = 0.5 + 7.5 * (
                    1 - np.clip(np.sum(normal_im * camvec, axis=-1), 0, 1))**4

                depth_im = self.rng.laplace(loc=depth_im,
                                            scale=depth_noise_strength *
                                            depth_im * surf_angle_noise_factor)

                normal_noise = self.rng.rand(rays.shape[0], rays.shape[1], 3)
                normal_im += normal_noise_strength * normal_noise
                normal_outlier_mask = self.rng.rand(
                    *normal_im.shape[:-1]) < normal_outlier_prob
                normal_im[normal_outlier_mask] = (
                    self.rng.rand(*normal_im[normal_outlier_mask].shape) - 0.5)
                normal_im /= np.linalg.norm(normal_im, axis=-1, keepdims=True)

                # compute the points using the noisy depth_im
                points = rays_original[:, :, :
                                       3] + rays_original[:, :,
                                                          3:] * depth_im[:, :,
                                                                         np.
                                                                         newaxis]

                dead_pixel_mask = self.rng.rand(
                    *depth_im.shape) > dead_pixel_prob
                valid_mask = np.logical_and(depth_im > 0, dead_pixel_mask)

                # results
                output_data['points'].append(points[valid_mask])
                output_data['campos'].append(rays_original[valid_mask, :3])
                output_data['normals'].append(normal_im[valid_mask])

                if self.rng.rand() < misalign_prob:
                    output_data['points'][-1] += np.random.uniform(
                        -misalign_strength * min_radius,
                        misalign_strength * min_radius,
                        size=(1, 3))

            # concatenate each point attribute list to a large vector
            for k, v in output_data.items():
                output_data[k] = np.concatenate(v, axis=0)

            num_points = output_data['points'].shape[0]

            # limit the number of points
            if num_points > max_num_points:
                for k, v in output_data.items():
                    output_data[k] = v[0:max_num_points]

            # generate outliers
            num_outliers = np.floor(outlier_ratio * num_points /
                                    (1 - outlier_ratio)).astype(np.int32)
            if num_outliers > 0:
                outlier = {}
                outlier['points'] = (
                    output_data['points'][:(num_points // num_outliers) *
                                          num_outliers:num_points //
                                          num_outliers]).copy()
                outlier['points'] += 0.4 * min_radius * (
                    self.rng.rand(*outlier['points'].shape) - 0.5)
                outlier['campos'] = max_radius * (
                    2 * self.rng.rand(*outlier['points'].shape) - 1)
                outlier['normals'] = (self.rng.rand(*outlier['points'].shape) -
                                      0.5)
                outlier['normals'] /= np.linalg.norm(outlier['normals'],
                                                     axis=-1,
                                                     keepdims=True)
                # TODO add sparse outliers in the bounding box

                for k, v in output_data.items():
                    output_data[k] = np.concatenate([v, outlier[k]], axis=0)

            sample = {}
            sample['points'] = output_data['points'].astype(np.float32)
            sample['normals'] = output_data['normals'].astype(np.float32)
            sample['campos'] = output_data['campos']

            # create holes
            holes = self._create_holes(sample['points'], hole_sizes=[8, 12])
            for k in ('points', 'campos', 'normals'):
                sample[k] = sample[k][np.logical_not(holes)]

        else:  # use precomputed point cloud

            # reduce number of points
            if pointcloud['points'].shape[0] > max_num_points:
                mask_points = np.full(shape=[pointcloud['points'].shape[0]],
                                      fill_value=False)
                mask_points[:max_num_points] = True
                self.rng.shuffle(mask_points)
                pointcloud['points'] = pointcloud['points'][mask_points]
                pointcloud['normals'] = pointcloud['normals'][mask_points]

            sample = {
                'points': np.matmul(pointcloud['points'], rand_R),
                'normals': np.matmul(pointcloud['normals'], rand_R),
                'campos': np.zeros_like(pointcloud['points']),  # TODO rm
            }

            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(vertices, triangles)

        # some preprocessing
        sample, octree = preprocess_input_points(sample['points'],
                                                 sample,
                                                 self.octree_levels,
                                                 radius_knn=self.rng.randint(
                                                     8, 24))

        def compute_signed_distance_and_closest_points(points):
            original_shape = points.shape
            points = points.reshape(-1, 3)
            closest_points = scene.compute_closest_points(
                points, nthreads=self.raycast_nthreads)['points'].numpy()
            distances = np.linalg.norm(points - closest_points, axis=-1)
            rays = np.empty([points.shape[0], 6], dtype=np.float32)
            rays[:, 0:3] = points
            rays[:, 3] = 1
            rays[:, 4:] = 0
            odd_intersections = (scene.count_intersections(
                rays, nthreads=self.raycast_nthreads).numpy() % 2) == 1
            distances[odd_intersections] *= -1
            points = points.reshape(*original_shape)
            return distances.reshape(original_shape[0],
                                     -1), closest_points.reshape(
                                         original_shape[0], -1, 3)

        def compute_signed_distance_and_closest_points_clipped(points):
            original_shape = points.shape
            points = point.reshape(-1, 3)
            ans = scene.compute_closest_points(points,
                                               nthreads=self.raycast_nthreads)
            closest_points = ans['points'].numpy()
            distances = np.linalg.norm(points - closest_points, axis=-1)

            # compute surface normals for the closest points
            tris_verts = vertices[triangles]
            tris_e1 = tris_verts[:, 1, :] - tris_verts[:, 0, :]
            tris_e2 = tris_verts[:, 2, :] - tris_verts[:, 0, :]
            tris_normals = np.cross(tris_e1, tris_e2)
            with np.errstate(divide='ignore', invalid='ignore'):
                tris_normals = np.nan_to_num(
                    tris_normals /
                    np.linalg.norm(tris_normals, axis=-1, keepdims=True))
            surf_normals = tris_normals[ans['primitive_ids'].numpy()]

            dot = np.sum((closest_points - points) * surf_normals, axis=-1)
            distances[dot > 0] *= -1
            points = points.reshape(*original_shape)
            return distances.reshape(original_shape[0],
                                     -1), closest_points.reshape(
                                         original_shape[0], -1,
                                         3), surf_normals.reshape(
                                             original_shape[0], -1, 3)

        sample['voxel_shifts0'] = self.rng.uniform(
            -self.shift_range,
            self.shift_range,
            size=(self.num_samples_per_voxel,) +
            sample['voxel_centers0'].shape).astype(np.float32)

        query_points = sample['voxel_centers0'][np.newaxis, :, :] + sample[
            'voxel_shifts0'] * sample['voxel_sizes0'][np.newaxis, :, np.newaxis]
        # the gt computation is different for clipped meshes because inside/outside is not defined
        if clipped_mesh:
            distances, closest_points, surf_normals = compute_signed_distance_and_closest_points_clipped(
                query_points)
            sample['voxel_shifts_closest_points0'] = closest_points
            sample['voxel_shifts_signed_distances0'] = distances
            sample['voxel_shifts_normals0'] = surf_normals
            sample['voxel_shifts_valid_signed_distances0'] = (
                (np.abs(distances) / sample['voxel_sizes0']) < 0.8).astype(
                    np.float32)
        else:
            distances, closest_points = compute_signed_distance_and_closest_points(
                query_points)
            sample['voxel_shifts_closest_points0'] = closest_points
            sample['voxel_shifts_signed_distances0'] = distances
            with np.errstate(divide='ignore', invalid='ignore'):
                sample['voxel_shifts_normals0'] = np.nan_to_num(
                    (query_points - closest_points) /
                    distances[..., np.newaxis])
            sample['voxel_shifts_valid_signed_distances0'] = np.ones_like(
                distances)

        if self.exclude_boundary:
            bbox_min = sample['voxel_centers0'].min(axis=0)
            bbox_max = sample['voxel_centers0'].max(axis=0)
            bbox_center = 0.5 * (bbox_min + bbox_max)
            bbox_size = 0.8 * (bbox_max - bbox_min)
            new_bbox_min = bbox_center - 0.5 * bbox_size
            new_bbox_max = bbox_center + 0.5 * bbox_size
            inside_bb = np.logical_and(
                np.all(sample['voxel_centers0'] < new_bbox_max[None, :],
                       axis=-1),
                np.all(sample['voxel_centers0'] > new_bbox_min[None, :],
                       axis=-1))
            sample['voxel_shifts_valid_signed_distances0'] *= inside_bb.astype(
                np.float32)

        if self.random_sign and not clipped_mesh:
            if isinstance(self.random_sign, (list, tuple)):
                sign = self.rng.choice(self.random_sign)
            else:
                sign = self.rng.choice([1, -1])
            if sign == -1:
                sample['normals'] = -sample['normals']
                sample['voxel_shifts_signed_distances0'] = -sample[
                    'voxel_shifts_signed_distances0']
                sample[
                    'voxel_shifts_normals0'] = -sample['voxel_shifts_normals0']

        num_points_min = min(sample['points'].shape[0], self.num_points_min)
        num_points_max = min(sample['points'].shape[0], self.num_points_max)

        num_points = self.rng.randint(num_points_min, num_points_max + 1)
        mask_points = np.full(shape=[sample['points'].shape[0]],
                              fill_value=False)
        mask_points[:(len(mask_points) // num_points) *
                    num_points:len(mask_points) // num_points] = True

        for k in ('points', 'campos', 'radii', 'normals'):
            sample[k] = sample[k][mask_points].astype(np.float32)

        # compute neighbors and scale compatibility
        def search_neighbors(points, query_points, radii):
            nns = o3d.core.nns.NearestNeighborSearch(
                o3d.core.Tensor.from_numpy(points))
            nns.multi_radius_index()
            ans = nns.multi_radius_search(
                o3d.core.Tensor.from_numpy(query_points),
                o3d.core.Tensor.from_numpy(radii))
            del nns
            return ans[0].numpy().astype(
                np.int32), ans[1].numpy(), ans[2].numpy()

        sample['aggregation_neighbors_index'], sample[
            'aggregation_neighbors_dist'], sample[
                'aggregation_row_splits'] = search_neighbors(
                    sample['points'], sample['voxel_centers0'],
                    sample['voxel_sizes0'])
        sample['aggregation_scale_compat'] = compute_scale_compatibility(
            sample['voxel_sizes0'], 2 * sample['radii'],
            sample['aggregation_neighbors_index'],
            sample['aggregation_row_splits'])

        # return sample

        sample['vertices'] = vertices
        sample['triangles'] = triangles.astype(np.int32)

        sample['dual_vertex_indices'] = asr.create_dual_vertex_indices(octree)

        return sample

    @staticmethod
    def item_types(octree_levels=3):
        import numpy
        types = {
            'points': numpy.float32,
            'normals': numpy.float32,
            'campos': numpy.float32,
            'radii': numpy.float32,
            'aggregation_neighbors_index': numpy.int32,
            'aggregation_neighbors_dist': numpy.float32,
            'aggregation_row_splits': numpy.int64,
            'aggregation_scale_compat': numpy.float32,
            'voxel_shifts0': numpy.float32,
            'voxel_shifts_closest_points0': numpy.float32,
            'voxel_shifts_signed_distances0': numpy.float32,
            'voxel_shifts_normals0': numpy.float32,
            'voxel_shifts_valid_signed_distances0': numpy.float32,
            'vertices': numpy.float32,
            'triangles': numpy.int32,
            'dual_vertex_indices': numpy.int64,
            'mesh_id': 'string'
        }
        for i in range(octree_levels):
            types.update({
                'voxel_centers{}'.format(i):
                    numpy.float32,
                'voxel_keys{}'.format(i):
                    numpy.int64,
                # 'voxel_levels{}'.format(i): numpy.int32,
                'voxel_sizes{}'.format(i):
                    numpy.float32,
                'neighbors_index{}'.format(i):
                    numpy.int32,
                'neighbors_kernel_index{}'.format(i):
                    numpy.uint8,
                'neighbors_row_splits{}'.format(i):
                    numpy.longlong,
            })
        for i in range(octree_levels - 1):
            types.update({
                'up_neighbors_index{}'.format(i): numpy.int32,
                'up_neighbors_kernel_index{}'.format(i): numpy.uint8,
                'up_neighbors_row_splits{}'.format(i): numpy.longlong,
            })
        return types

    @staticmethod
    def item_shapes(octree_levels=3):
        shapes = {
            'points': (None, 3),
            'normals': (None, 3),
            'campos': (None, 3),
            'radii': (None,),
            'aggregation_neighbors_index': (None,),
            'aggregation_neighbors_dist': (None,),
            'aggregation_row_splits': (None,),
            'aggregation_scale_compat': (None,),
            'voxel_shifts0': (None, None, 3),
            'voxel_shifts_closest_points0': (None, None, 3),
            'voxel_shifts_signed_distances0': (
                None,
                None,
            ),
            'voxel_shifts_normals0': (None, None, 3),
            'voxel_shifts_valid_signed_distances0': (
                None,
                None,
            ),
            'vertices': (None, 3),
            'triangles': (None, 3),
            'dual_vertex_indices': (None, 8),
            'mesh_id': ()
        }
        for i in range(octree_levels):
            shapes.update({
                'voxel_centers{}'.format(i): (None, 3),
                'voxel_keys{}'.format(i): (None,),
                # 'voxel_levels{}'.format(i): (None,),
                'voxel_sizes{}'.format(i): (None,),
                'neighbors_index{}'.format(i): (None,),
                'neighbors_kernel_index{}'.format(i): (None,),
                'neighbors_row_splits{}'.format(i): (None,),
            })
        for i in range(octree_levels - 1):
            shapes.update({
                'up_neighbors_index{}'.format(i): (None,),
                'up_neighbors_kernel_index{}'.format(i): (None,),
                'up_neighbors_row_splits{}'.format(i): (None,),
            })
        return shapes

    @staticmethod
    def _create_item_types_and_shapes_dicts(data):

        def get_shape(v):
            if isinstance(v, str):
                return tuple()
            elif isinstance(v, np.ndarray):
                return (None,) + v.shape[1:]
            else:
                raise Exception("unknown type")

        def get_type(v):
            if isinstance(v, str):
                return 'string'
            elif isinstance(v, np.ndarray):
                return v.dtype.type
            else:
                raise Exception("unknown type")

        shapes = {k: get_shape(v) for k, v in data.items()}
        types = {k: get_type(v) for k, v in data.items()}

        return {'shapes': shapes, 'types': types}

    def __iter__(self):
        decompressor = zstd.ZstdDecompressor()
        files_idxs = np.arange(len(self.files))

        if not self.shuffle and not self.random_rotation:
            self.rng.seed(0)

        if self.shuffle:
            self.rng.shuffle(files_idxs)

        for file_i in files_idxs:
            # read all data from file
            try:
                with open(self.files[file_i], 'rb') as f:
                    data = msgpack.unpackb(decompressor.decompress(f.read()),
                                           raw=False)
            except Exception as e:
                print(e, file_i, self.files[file_i])
                continue

            # do not combine meshes if we have a precomputed point cloud
            if 'pointcloud' in data[0]:
                num_meshes_min = 1
                num_meshes_max = 1
            else:
                num_meshes_min = self.num_meshes_min
                num_meshes_max = self.num_meshes_max

            data_idxs = np.arange(len(data))
            if self.shuffle:
                self.rng.shuffle(data_idxs)
            # create splits containing up to num_meshes_max meshes
            splits = [0]
            while splits[-1] < len(data_idxs):
                splits.append(splits[-1] +
                              self.rng.randint(num_meshes_min, num_meshes_max +
                                               1))
            splits[-1] = len(data_idxs)

            # subsample and augment one item to generate a sample
            for splits_i in range(len(splits) - 1):
                if self.single_example and self.single_example_data is not None:
                    yield self.single_example_data

                items = [
                    data[x]
                    for x in data_idxs[splits[splits_i]:splits[splits_i + 1]]
                ]

                if self.random_crop and len(
                        items) == 1 and self.random_crop > self.rng.rand():
                    verts, tris = self._clip_mesh(**items[0])
                    items = [{
                        'vertices': verts,
                        'triangles': tris,
                        'mesh_id': items[0]['mesh_id']
                    }]
                    clipped_mesh = True
                else:
                    clipped_mesh = False

                combined_item = self._combine_items(items)

                sample = self._generate_sample_point_cloud(
                    **combined_item,
                    clipped_mesh=clipped_mesh,
                    noise=self.noise_strength
                    if self.noise_strength != 'auto' else None)
                if all(['mesh_id' in x for x in items]):
                    sample['mesh_id'] = ','.join([x['mesh_id'] for x in items])
                else:
                    sample['mesh_id'] = 'unknown'

                # tensorflow does not support uint64
                if self.convert_uint64:
                    for k in sample:
                        if isinstance(
                                sample[k],
                                np.ndarray) and sample[k].dtype == np.uint64:
                            sample[k] = sample[k].astype(np.int64)

                if self.single_example and self.single_example_data is None:
                    self.single_example_data = sample
                yield sample


def read_data(files=None,
              batch_size=1,
              num_points_min=10000,
              num_points_max=20000,
              num_meshes_min=1,
              num_meshes_max=1,
              random_rotation=False,
              repeat=False,
              shuffle_buffer=None,
              num_parallel_calls=1,
              octree_levels=3,
              **kwargs):
    """Creates a dataflow.

    Args:
        files: A list of data files or a dictionary with multiple lists.
        batch_size: The batch size
        num_points_min: The minimum number of points to return for each batch item
        num_points_max: The maximum number of points to return for each batch item
        num_meshes_min: The minimum number of meshes to use
        num_meshes_max: The maximum number of meshes to use
        random_rotation: If True randomly rotate the data
        repeat: If True repeat forever
        shuffle_buffer: Shuffle buffer used for randomizing the order
        num_parallel_calls: The number of workers to use
        octree_levels: The number grid levels to generate. This should match the network architecture
    """

    if not isinstance(files, dict):
        print(files[:20])

    if isinstance(files, dict):
        print('files contains the parts', list(files.keys()))
        df_list = []
        for name, value in files.items():
            df = PointCloudReconstructionDataFlow(
                files=value,
                random_rotation=random_rotation,
                shuffle=True if shuffle_buffer else False,
                num_points_min=num_points_min,
                num_points_max=num_points_max,
                num_meshes_min=num_meshes_min,
                num_meshes_max=num_meshes_max,
                octree_levels=octree_levels,
                **kwargs)
            df.set_name = name
            df_list.append(df)
        df = dataflow.RandomChooseData(df_list)

    else:
        df = PointCloudReconstructionDataFlow(
            files=files,
            random_rotation=random_rotation,
            shuffle=True if shuffle_buffer else False,
            num_points_min=num_points_min,
            num_points_max=num_points_max,
            num_meshes_min=num_meshes_min,
            num_meshes_max=num_meshes_max,
            octree_levels=octree_levels,
            **kwargs)

    if repeat:
        df = dataflow.RepeatedData(df, -1)

    if shuffle_buffer:
        df = dataflow.LocallyShuffleData(df, shuffle_buffer)

    if num_parallel_calls > 1:
        df = dataflow.MultiProcessRunnerZMQ(df, num_proc=num_parallel_calls)

    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)

    def dict_of_lists_to_tuple_of_dicts(dic):
        batch_size = len(next(iter(dic.values())))
        result = []
        for batch_i in range(batch_size):
            result.append({k: v[batch_i] for k, v in dic.items()})
        return tuple(result)

    def cast_uint64_to_int64(batch):
        result = []

        def cast(v):
            if isinstance(v, np.ndarray) and np.dtype(v.dtype).str == np.dtype(
                    np.uint64).str:
                return v.astype(np.int64)
            else:
                return v

        for item in batch:
            result.append({k: cast(v) for k, v in item.items()})
        return tuple(result)

    df = dataflow.MapData(df, dict_of_lists_to_tuple_of_dicts)
    df = dataflow.MapData(df, cast_uint64_to_int64)

    df.reset_state()
    return df
