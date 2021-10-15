# Datasets

## Dataset format

The training code in `models/v0` expects watertight meshes stored as dictionary of the form

```python
{
    'vertices': np.array()  # array with shape [num_vertices,3] and dtype np.float32
    'triangles': np.array() # array with shape [num_triangles,3] and dtype np.int32
}
```
Meshes are normalized such that the unit cube contains all vertices.

For efficieny reasons we store 10 dicts in a list and serialize them with `msgpack` and `zstd`.
The following code snippet can be used to generate a data file.

```python
def write_compressed_msgpack(data, output_path):
    import zstandard as zstd
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()

    compressor = zstd.ZstdCompressor(level=22)
    with open(output_path, 'wb') as f:
        print('writing', outfilepath)
        f.write(compressor.compress(msgpack.packb(data, use_bin_type=True)))
```



## Download links

Download links will be added soon.
