import os
import timeit
from typing import Callable
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


def default_target_transform(x):
  return int(x)


class ImageNetDataset(Dataset):
  """
    A tailor-made dataset where the samples are arranged in this way:
      ./train/n01440764/n01440764_10026.JPEG  449
      ./train/n01440764/n01440764_10027.JPEG  449
      ./train/n01440764/n01440764_10029.JPEG  449
    Args:
      root (string): Root directory path.
      input_spec(string): Input file path.
      delimiter (string): Delimiter between file path and label.
      dataset_name (string, optional): Dataset name.
      transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
      target_transform (callable): A function/transform that takes in the target and transforms it.
      loader (callable): A function to load a sample given its path.
      limit (int, optional): The limit number of samples to load.
      verbose(bool): If True, it will give verbose outputs.
  """

  def __init__(self,
               root: str,
               input_spec: str,
               delimiter: str = '\t',
               dataset_name: str = None,
               transform: Callable = None,
               target_transform: Callable = None,
               loader: Callable = default_loader,
               limit: int = None,
               verbose: bool = True):
    self.root = root
    self.dataset_name = dataset_name
    self.transform = transform
    self.target_transform = target_transform or default_target_transform
    self.loader = loader
    self.verbose = verbose
    self.samples = []

    record_num = 0
    start = timeit.default_timer()
    with open(input_spec) as file:
      for line in file.readlines():
        if limit is not None and record_num >= limit:
          break
        path, target = line.strip().split(delimiter, 1)
        self.samples.append((os.path.join(root, path), target, path))
        record_num += 1

    if self.verbose:
      print('Totally read {} records, elapsed {:.3f}(s)'.format(
          record_num,
          timeit.default_timer() - start))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    path, target, payload = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return sample, target, payload


if __name__ == '__main__':
  directory = '/path/to/imagenet_1k_rawimgs'
  dataset = ImageNetDataset(root=directory,
                            input_spec=os.path.join(directory, 'train.txt'),
                            limit=10)
  print(len(dataset))
  for (sample, target, payload) in dataset:
    print(sample, target, payload)
