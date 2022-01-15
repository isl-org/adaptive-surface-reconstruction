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
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datareader import PointCloudReconstructionDataFlow, read_data
import time
import argparse
import yaml
from glob import glob
from collections import namedtuple
import numpy as np
import tensorflow as tf
from utils.deeplearningutilities.tf import *
from net_definitions_tf import UNet5

tf_version = tuple(map(int, tf.__version__.split('.')))
gpu_devices = []
gpu_devices = tf.config.list_physical_devices('GPU')
if not gpu_devices:
    gpu_devices = ['CPU']

from datareader import val_files, train_files, train_files_synthetic_colmap

# yapf: disable
EvoParams = namedtuple('EvoParams', ['name', 'stop_step', 'base_lr', 'random_sign'])
evoparams_list = [
                           EvoParams('evo1',  1000,         0.001,    False    ),
                           EvoParams('evo2',  2000,         0.001,    [1,1,-1]    ),
                           EvoParams('evo3',  25000,        0.001,    [1,-1]    ),
        ]
# yapf: enable

BATCH_SIZE = 1
WORKER_BATCH_SIZE = BATCH_SIZE // len(gpu_devices)
if WORKER_BATCH_SIZE * len(gpu_devices) != BATCH_SIZE:
    raise Exception("Cannot distribute batch_size == {} to {} devices".format(
        BATCH_SIZE, len(gpu_devices)))


def create_model(cfg):
    """Returns an instance of the network for training and evaluation"""
    model = UNet5(**cfg)
    return model


def run_model(model, item, reduce_mem_peak=False, keep_threshold=1):
    import gc
    points = item['points']
    item['feats'] = tf.concat(
        [item['normals'], tf.ones_like(item['points'][:, 0:1])], axis=-1)
    if reduce_mem_peak:
        gc.collect()
    tmp = model.aggregate(item)
    if reduce_mem_peak:
        del points
        del item['points']
        del item['radii']
        del item['normals']
        del item['feats']
        gc.collect()

    code = model.unet(tmp, item, keep_threshold)
    if reduce_mem_peak:
        del tmp
        gc.collect()
    value, shifts_grad = model.decode(item['voxel_shifts0'], code)
    output = {'value': value, 'shifts_grad': shifts_grad}

    if reduce_mem_peak:
        gc.collect()
    return output


def loss_sqr(x, y, valid_mask):
    return tf.reduce_mean(valid_mask * (x - y)**2)


def loss_weighted_euclidean(x, y, w):
    return tf.reduce_mean(w * tf.sqrt((x - y)**2 + 1e-4))


def loss_fn(gt_value,
            pr_value,
            valid_mask,
            sdf_clip,
            clip_loss=False,
            squared=True):
    clip_value = sdf_clip
    gt_clip = tf.clip_by_value(gt_value, -clip_value, clip_value)
    if clip_loss:
        valid = valid_mask * tf.cast(tf.abs(gt_value) < clip_value,
                                     dtype=tf.float32)
    else:
        valid = valid_mask

    if squared:
        return loss_sqr(gt_clip, pr_value[..., 0], valid)
    else:
        return loss_weighted_euclidean(gt_clip, pr_value[..., 0], valid)


def loss_normal_fn(gt_normal, pr_normal, gt_value, valid_mask, sdf_clip,
                   loss_cfg):
    valid = valid_mask * tf.reduce_sum(gt_normal * gt_normal, axis=-1)
    if loss_cfg['ramp_normal_loss']:
        w = valid * tf.clip_by_value(1 - tf.abs(gt_value) / sdf_clip, 0, 1)
        if loss_cfg.get('squared_normal_loss', True):
            return tf.reduce_mean(w * tf.reduce_sum(
                (gt_normal - pr_normal)**2, axis=-1))
        else:
            return tf.reduce_mean(w * tf.sqrt(
                tf.reduce_sum((gt_normal - pr_normal)**2, axis=-1) + 1e-4))
    else:
        clipped_gt_normal = gt_normal * tf.dtypes.cast(
            (tf.abs(gt_value) < sdf_clip), tf.float32)[:, None]
        if loss_cfg.get('squared_normal_loss', True):
            return tf.reduce_mean(valid * tf.reduce_sum(
                (clipped_gt_normal - pr_normal)**2, axis=-1))
        else:
            return tf.reduce_mean(valid * tf.sqrt(
                tf.reduce_sum((clipped_gt_normal - pr_normal)**2, axis=-1) +
                1e-4))


def compute_evaluation_loss(model, batch, sdf_clip):
    loss = 0.0
    for item in batch:
        output = run_model(model, item)
        gt_value = item['voxel_shifts_signed_distances0'] / item['voxel_sizes0']
        valid_mask = item['voxel_shifts_valid_signed_distances0']
        loss += loss_fn(gt_value, output['value'], valid_mask, sdf_clip)

        values = output['value'].numpy()
        gt = np.clip(gt_value, -sdf_clip, sdf_clip)
        print('mean', np.mean(values), 'std', np.std(values), np.min(values),
              np.max(values))
        print('mean', np.mean(gt), 'std', np.std(gt), np.min(gt), np.max(gt))

    return loss / len(batch)


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--cfg",
                        type=str,
                        required=True,
                        help="The path to the config file")
    parser.add_argument(
        "--train_dir",
        type=str,
        help=
        "Custom path to the train_dir. The default will be derived from the cfg file."
    )
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        print(cfg)

    train_dir = '{}_{}'.format(
        os.path.splitext(os.path.basename(__file__))[0],
        os.path.splitext(os.path.basename(args.cfg))[0])
    if args.train_dir:
        train_dir = args.train_dir

    trainer = EvolutionTrainer(train_dir,
                               evoparams_list,
                               keep_checkpoint_steps=[])
    current_evo = trainer.current_evolution

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
    # use a single example for fast eval loss computation
    validation_example = next(val_data_iter)

    training_files = {
        'synthetic_colmap': train_files_synthetic_colmap,
        'synthetic': train_files
    }
    if cfg.get('no_colmap_data', False):
        training_files = training_files['synthetic']
    train_dataset = read_data(files=training_files,
                              batch_size=BATCH_SIZE,
                              repeat=True,
                              random_rotation=True,
                              num_points_min=400000,
                              num_points_max=500000,
                              num_meshes_max=3,
                              shuffle_buffer=4,
                              convert_uint64=True,
                              random_sign=current_evo.random_sign,
                              octree_levels=UNet5.octree_levels,
                              **cfg['traindata'])

    # convert to a tf.Dataset
    train_dataset = tf.data.Dataset.from_generator(
        train_dataset.get_data,
        tuple([
            PointCloudReconstructionDataFlow.item_types(
                octree_levels=UNet5.octree_levels)
        ] * WORKER_BATCH_SIZE),
        output_shapes=tuple([
            PointCloudReconstructionDataFlow.item_shapes(
                octree_levels=UNet5.octree_levels)
        ] * WORKER_BATCH_SIZE),
    )
    train_dataset = train_dataset.prefetch(BATCH_SIZE // WORKER_BATCH_SIZE)

    strategy = tf.distribute.MirroredStrategy()

    # create distributed dataset
    def dist_dataset_fn(input_context):
        return train_dataset

    dist_train_dataset = strategy.experimental_distribute_datasets_from_function(
        dist_dataset_fn)
    dist_train_data_iter = iter(dist_train_dataset)

    # learning rate schedule
    boundaries = [
        current_evo.stop_step,
    ]
    lr_values = [
        current_evo.base_lr * 1.0,
        current_evo.base_lr * 1.0,
    ]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_values)

    step_var = tf.Variable(0)  # define step var outside the strategy scope
    with strategy.scope():
        model = create_model(cfg['model'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                             epsilon=1e-4)
        checkpoint = tf.train.Checkpoint(step=step_var,
                                         model=model,
                                         optimizer=optimizer)

    # define training step
    @tf.function
    def train_step():

        # get data
        dist_batch = next(dist_train_data_iter)

        def step_fn(batch):
            with tf.GradientTape() as tape:
                losses = []
                for i in range(WORKER_BATCH_SIZE):
                    item = batch[i]
                    points = item['points']
                    item['feats'] = tf.concat(
                        [item['normals'],
                         tf.ones_like(item['points'][:, 0:1])],
                        axis=-1)

                    output, debug_info = model(item)

                    gt_value = item['voxel_shifts_signed_distances0'] / item[
                        'voxel_sizes0']
                    valid_mask = item['voxel_shifts_valid_signed_distances0']
                    lvalue = loss_fn(
                        gt_value,
                        output['value'][..., :1],
                        valid_mask,
                        cfg['sdf_clip'],
                        clip_loss=cfg['loss'].get('clip_signed_loss', False),
                        squared=cfg['loss'].get('squared_signed_loss', True))
                    lvalue_unsigned = cfg['loss'].get(
                        'unsigned_weight',
                        0.5) * loss_fn(tf.abs(gt_value),
                                       output['value'][..., 1:],
                                       tf.ones_like(valid_mask),
                                       cfg.get('udf_clip', cfg['sdf_clip']),
                                       squared=cfg['loss'].get(
                                           'squared_unsigned_loss', True))
                    lnormal = cfg['loss'].get(
                        'normal_weight', 0.1) * loss_normal_fn(
                            item['voxel_shifts_normals0'],
                            output['shifts_grad'],
                            item['voxel_shifts_signed_distances0'] /
                            item['voxel_sizes0'], valid_mask, cfg['sdf_clip'],
                            cfg['loss'])

                    losses.append(('ValueLoss', lvalue))
                    losses.append(('UnsignedValueLoss', lvalue_unsigned))
                    losses.append(('NormalLoss', lnormal))

                losses_dict = {k: [] for k, _ in losses}
                for k, v in losses:
                    losses_dict[k].append(v)
                losses = {
                    k: tf.add_n(v) / BATCH_SIZE for k, v in losses_dict.items()
                }

                print(losses)
                loss = tf.add_n(losses.values())

            trainable_vars = model.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            return losses, debug_info

        if tf_version >= (2, 4, 0):
            losses, debug_info = strategy.run(step_fn, args=(dist_batch,))
        else:
            losses, debug_info = strategy.experimental_run_v2(
                step_fn, args=(dist_batch,))

        losses = {
            k: strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
            for k, v in losses.items()
        }
        losses['TotalLoss'] = tf.add_n(losses.values())

        # only return the debug info for one device
        if len(gpu_devices) > 1:
            debug_info = {k: v.primary for k, v in debug_info.items()}

        print(
            '=== Training variables ========================================================='
        )
        for x in model.trainable_variables:
            print(x.name)
        print(
            '================================================================================'
        )

        return losses, debug_info

    if trainer.latest_checkpoint:
        print('restoring from ', trainer.latest_checkpoint)
        checkpoint.restore(trainer.latest_checkpoint)
    else:
        print('no checkpoints found!')

    display_str_list = []
    while trainer.keep_training(checkpoint.step,
                                checkpoint,
                                display_str_list=display_str_list):
        with trainer.summary_writer.as_default():

            losses, debug_info = train_step()
            display_str_list = [
                'loss',
                float(losses['TotalLoss']),
            ]

            if trainer.current_step % 10 == 0:
                for k, v in losses.items():
                    tf.summary.scalar(k, v)
                tf.summary.scalar('LearningRate',
                                  optimizer.lr(trainer.current_step))
                for k, v in debug_info.items():
                    if v.shape.rank == 0:
                        tf.summary.scalar(k, v)

            if trainer.current_step % 200 == 0:
                eval_loss = compute_evaluation_loss(model, validation_example,
                                                    cfg['sdf_clip'])
                print('eval_loss', eval_loss)
                tf.summary.scalar('EvalLoss', eval_loss)

    if trainer.current_step == evoparams_list[-1].stop_step:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
