#!/usr/bin/python3
# coding -*- utf8 -*-
# author: zhangpiu

import json
import os
import struct
import sys
import imp
import logging
import queue
import timeit
import argparse
import threading
from torch.utils.data.dataloader import DataLoader
from imagenet_dataset import ImageNetDataset
from feature_pb2 import FeatureRecord

TIMEOUT = 5  # seconds
LOG_FREQUENCY = 10
FINISHED = False
BUFFER = queue.Queue(10000)


def write_worker(output: str):
  with open(output, 'wb') as writer:
    while not FINISHED or not BUFFER.empty():
      samples = None
      try:
        samples = BUFFER.get(timeout=TIMEOUT)
      except queue.Empty as e:
        logging.warning('The buffer for writing may be empty! {}'.format(
            str(e)))
      if samples is None:
        continue

      try:
        for sample in samples:
          feature = FeatureRecord()
          feature.value.extend(sample[0])
          feature.label = sample[1]
          feature.id = os.path.basename(sample[2])
          feature.payload = json.dumps(dict(img=sample[2])).encode('utf-8')
          data = feature.SerializeToString()
          size = struct.pack('<Q', len(data))
          writer.write(size)
          writer.write(data)
          writer.flush()
      except Exception as e:
        logging.error(str(e))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                      level=logging.INFO,
                      datefmt='%Y-%m-%d %H:%M:%S')
  args = argparse.ArgumentParser()
  args.add_argument('-d', '--dir', required=True, help='Image root directory.')
  args.add_argument('-i',
                    '--input',
                    required=True,
                    help='Text file which contains image relative path.')
  args.add_argument('-o',
                    '--output',
                    required=True,
                    help='Text file used to store features.')
  args.add_argument('--input-field-delimiter',
                    dest='input_field_delimiter',
                    default='\t',
                    help="Specify the field delimiter.")
  args.add_argument('-j',
                    '--job',
                    required=True,
                    help="job file which contains 'transform' object")
  args.add_argument('-b',
                    '--batch-size',
                    dest='batch_size',
                    type=int,
                    default=1,
                    help='Batch size')
  args.add_argument(
      '-w',
      '--num-workers',
      dest='num_workers',
      type=int,
      default=0,
      help=
      'Number of subprocess for data loading, 0 means using the main process.')
  args.add_argument('-f',
                    '--framework',
                    default='pytorch',
                    choices=('pytorch', 'tensorflow'),
                    help="The deep learning framework used in the job file")
  args.add_argument('-l', '--limit', default=None, type=int, help='test limit')
  options = args.parse_args()

  if not os.path.isdir(options.dir):
    raise ValueError("'--dir' should be a valid directory path")

  job = imp.load_source('job', options.job)
  if 'model' not in dir(job):
    raise TypeError("The job file MUST contain an object named 'model'")
  if 'transform' not in dir(job):
    raise TypeError(
        "The job file MUST contain 'transform' used for transforming an image to a tensor"
    )

  target_transform = None
  dataset = ImageNetDataset(options.dir,
                            options.input,
                            delimiter=options.input_field_delimiter,
                            transform=job.transform,
                            target_transform=target_transform,
                            limit=options.limit)
  data_loader = DataLoader(dataset,
                           options.batch_size,
                           num_workers=options.num_workers,
                           pin_memory=False)

  writing_thread = threading.Thread(target=write_worker, args=(options.output,))
  writing_thread.start()

  batch_num = 0
  start = timeit.default_timer()

  try:
    for (inputs, targets, payloads) in data_loader:
      if options.framework == 'pytorch':
        results = job.model(inputs)
      else:
        results = job.model(inputs.numpy())
      BUFFER.put(zip(results, targets, payloads))

      batch_num += 1
      if batch_num % LOG_FREQUENCY == 0:
        logging.info(
            "{} batches that is {} records processed, elapsed {:.3f}(s)".format(
                batch_num, batch_num * options.batch_size,
                timeit.default_timer() - start))
  except Exception as e:
    logging.error(str(e))
    sys.exit(1)

  FINISHED = True
  logging.info(
      "Totally {} batches that is {} records processed, elapsed {:.3f}(s)".
      format(batch_num, len(dataset),
             timeit.default_timer() - start))

  writing_thread.join()
