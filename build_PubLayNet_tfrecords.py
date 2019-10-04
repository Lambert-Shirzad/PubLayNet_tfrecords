# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Converts PubLayNet RawImages and SegmentationClass images data to TFRecord file format with Example protos.

The dataset is expected to have the following directory structure:

  + datasets (current working directory)
    - create_PubLayNet_segmentation_mask_png_files.py
    - build_data.py
    - build_PubLayNet_tfrecords.py
    + PubLayNet
        - dev.json
        - train.json
        + RawImages
            + val_data
            + train_data
                + train_0
                ...
                + train_6
        + SegmentationClass
            + train
            + val
        + tfrecord (this script will save tfrecord files here)

Image folder:
  ./PubLayNet/RawImages

Semantic segmentation annotations:
  ./PubLayNet/SegmentationClass

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded raw image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import glob
import os.path
import sys
import build_data
import tensorflow as tf

num_docs_per_shard = 500

def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.

    Args:
        dataset_split: The dataset split (e.g., train, val).
    Raises:
        RuntimeError: If loaded image and label have different shape.
    """
    sys.stdout.write('Processing ' + dataset_split)
    
    if dataset_split == 'train':
        seg_base_dir = 'PubLayNet/SegmentationClass/train/'
        raw_base_dir = 'PubLayNet/RawImages/train_data/'
    else:
        seg_base_dir = 'PubLayNet/SegmentationClass/val/'
        raw_base_dir = 'PubLayNet/RawImages/val_data/val/'
    
    seg_file_names = [f for f in glob.glob(seg_base_dir + "**/*.png", recursive=True)]
    raw_file_names = [f for f in glob.glob(raw_base_dir + "**/*.jpg", recursive=True)]
    raw_name_only = [f.split('/')[-1].split('.')[-2] for f in raw_file_names]
    num_images = len(seg_file_names)
    num_shards = int(math.ceil(num_images / float(num_docs_per_shard)))
    
    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(num_shards):
        output_filename = os.path.join(
                                './PubLayNet/tfrecord', 
                                '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, num_shards))
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_docs_per_shard
            end_idx = min((shard_id + 1) * num_docs_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, num_images, shard_id))
                sys.stdout.flush()
                
                # Read the semantic segmentation annotation.
                seg_name_only = seg_file_names[i].split('/')[-1].split('.')[-2]
                seg_data = tf.io.gfile.GFile(seg_file_names[i], 'rb').read()
                '''print('/n', dataset_split, 'seg', seg_file_names[i])
                print(dataset_split, 'seg', seg_name_only)'''
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                
                # Read the raw image.
                name_ind = raw_name_only.index(seg_name_only)
                '''print(dataset_split, 'raw', raw_file_names[name_ind])
                print(dataset_split, 'raw', raw_name_only[name_ind])'''
                image_data = tf.io.gfile.GFile(raw_file_names[name_ind], 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                
                if height != seg_height or width != seg_width:
                    #raise RuntimeError('Shape mismatched between image and label.')
                    print('The raw image and segmentation mask do not have the same dimensions.')
                    print('Skipping image {}'.format(seg_name_only))
                else:
                    # Convert to tf example.
                    example = build_data.image_seg_to_tfexample(
                                            image_data, seg_name_only, height, width, seg_data)
                    tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\n')
                sys.stdout.flush()


def main(unused_argv):
    dataset_splits = ['train', 'val']
    for dataset_split in dataset_splits:
        _convert_dataset(dataset_split)


if __name__ == '__main__':
    tf.app.run()
    
    
