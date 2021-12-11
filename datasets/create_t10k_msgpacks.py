#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import open3d as o3d
import numpy as np
from glob import glob
import argparse
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from multiprocessing import Pool

def write_compressed_msgpack(data, path, level=22, threads=0):
    compressor = zstd.ZstdCompressor(level=level, threads=threads)
    with open(path, 'wb') as f:
        print('writing', path)
        f.write(compressor.compress(msgpack.packb(data, use_bin_type=True)))


def read_compressed_msgpack(path, decompressor=None):
    if decompressor is None:
        decompressor = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
    return data


def select_good_meshes(info_dict, data_dir):
    # select only good meshes
    raw_meshes_dir = os.path.join(data_dir,'raw_meshes')
    selected_meshes = []
    attribution = []
    selection = {
        'Closed': 'TRUE',
        'Single Component': 'TRUE',
        'No duplicated faces': 'TRUE',
        'No degenerate faces': 'TRUE',
        'Vertex manifold': 'TRUE',
        'oriented': '1',
        'solid': '1',
    }
    licenses = (
        'Creative Commons - Attribution - Share Alike',
        'Creative Commons - Attribution',
        'Creative Commons - Public Domain Dedication',
        'Public Domain'
    )

    keys = sorted(info_dict.keys())
    # remove bad file ids
    for bas_id in ('112965',):
        keys.remove(bas_id)

    for key in keys:
        info = info_dict[key]
        selected = True
        for sel_key, sel_val in selection.items():
            if info[sel_key] != sel_val:
                selected = False
                break;
        if selected and info['License'] in licenses:
            attribution.append('"{}"({}) by {} is licensed under {}'.format(info['title'].strip(), info['Thing ID'], info['author'], info['License']))
            selected_meshes.append(glob(os.path.join(raw_meshes_dir,key+'.*'))[0])

    return selected_meshes, attribution


def create_data(mesh_paths, output_path):
    data = []
    for path in mesh_paths:
        try:
            mesh = o3d.io.read_triangle_mesh( path )
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            mesh_id = os.path.basename(path)

            hull = mesh.compute_convex_hull()[0]
            hull_vertices = np.asarray(hull.vertices)
            
            
            scale = np.max(np.linalg.norm(hull_vertices - hull_vertices[0], axis=1))
            
            vertices /= scale
            center = 0.5*(vertices.max(axis=0)+vertices.min(axis=0))
            vertices -= center

            feat_dict = {
                    'mesh_id': mesh_id,
                    'vertices': vertices.astype(np.float32),
                    'triangles': triangles.astype(np.int32),
                    }

            data.append(feat_dict)
        except Exception as err:
            print("Failed to generate data for", path)

    write_compressed_msgpack(data, output_path)



def main():
    parser = argparse.ArgumentParser(description="Create data files for training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="The path to the Thingi10k dataset root.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), 't10k'), help="The path to the output dir")
    parser.add_argument("--attribution_file_only", action="store_true", help="Create only the attribution file")

    args = parser.parse_args()

    info_dict = read_compressed_msgpack(os.path.join(os.path.dirname(__file__),'thingi10k_info.msgpack.zst'))

    meshes, attributions = select_good_meshes(info_dict, args.data_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    valid_output_dir = os.path.join(args.output_dir, 'valid')
    os.makedirs(valid_output_dir, exist_ok=True)
    train_output_dir = os.path.join(args.output_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)

    attribution_file = "{}_attribution.txt".format(os.path.basename(args.output_dir))
    with open(os.path.join(args.output_dir,attribution_file), 'w') as f:
        f.write("\n".join(attributions))

    if args.attribution_file_only:
        return

    meshes_sublists = [ [str(ii) for ii in i] for i in np.array_split(meshes, 100) ]
    print('objects per record', len(meshes_sublists[0]))
    output_paths = [ os.path.join(valid_output_dir if i < 5 else train_output_dir,'thingi10k_{0:03d}.msgpack.zst'.format(i)) for i in range(len(meshes_sublists)) ]

    arguments = list(zip(meshes_sublists, output_paths))

    with Pool(16) as pool:
        pool.starmap(create_data, arguments)


if __name__ == '__main__':
    main()
