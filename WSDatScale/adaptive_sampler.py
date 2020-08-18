# pylint: disable=not-callable
import logging
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar
import random, torch
from torch.utils.data.sampler import Sampler
from torch.utils import data
logger = logging.getLogger(__name__)
A = TypeVar('A')
PAD_ID = 0

def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise


class BucketBatchSampler(Sampler):

    def __init__(self, data_source: data.Dataset, batch_size: int, sorting_keys: List[str]=None, padding_noise: float=0.1, drop_last: bool=False):
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.batch_size = batch_size
        self.data_source = data_source
        self.drop_last = drop_last

    def _argsort_by_padding(self, instances: Iterable[Dict[(str, torch.tensor)]]) -> Tuple[(List[int], List[List[int]])]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        if not self.sorting_keys:
            raise Exception('No sorting keys given; trying to guess a good one')
        instances_with_lengths = []
        for instance in instances:
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:
                if not hasattr(instance, field_name):
                    raise Exception(f'Sorting key "{field_name}" is not a field in instance. Available fields/keys are {list(instance.fields.keys())}.')
                lengths.append(len(getattr(instance, field_name)))
                noisy_lengths.append(add_noise_to_value(lengths[(-1)], self.padding_noise))

            instances_with_lengths.append((noisy_lengths, lengths, instance))

        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=(lambda x: x[0][0]))
        return (
         [instance_with_index[(-1)] for instance_with_index in with_indices],
         [instance_with_index[0][1] for instance_with_index in with_indices])


class MaxTokensBatchSampler(BucketBatchSampler):

    def __init__(self, data_source, max_tokens=None, sorting_keys=[
 'input_ids'], padding_noise=0.1):
        super().__init__(data_source, -1, sorting_keys, padding_noise, False)
        self.max_tokens = max_tokens

    def __iter__(self) -> Iterable[List[int]]:
        indices, lengths = self._argsort_by_padding(self.data_source)
        max_lengths = [max(length) for length in lengths]
        group_iterator = self._lazy_groups_of_max_size(indices, max_lengths)
        batches = [list(group) for group in group_iterator]
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def _lazy_groups_of_max_size(self, iterable: Iterable[A], sizes: Iterable[int]) -> Iterator[List[A]]:
        """
        Takes an `iterable` of data and an iterable `sizes` of the same length which represents the sizes of each
        corresponding item in `iterable`. The instances from `iterable` are batched such that the total size
        of the batch as computed from `sizes` does not exceed `max_size`.
        """
        cur_max_size = 0
        group = []
        iterator = iter(iterable)
        size_iter = iter(sizes)
        for item, size in zip(iterator, size_iter):
            if size > self.max_tokens:
                logger.warning('Found instance of size %d, which is bigger than the expected size for a batch (%d)', size, self.max_tokens)
            group_size = max(size, cur_max_size) * (len(group) + 1)
            if group_size > self.max_tokens:
                yield group
                cur_max_size = 0
                group = []
            group.append(item)
            cur_max_size = max(cur_max_size, size)

        if len(group) != 0:
            yield group

    def __len__(self):
        return sum((1 for _ in self))


def data_collator_for_adaptive_sampler(features) -> Dict[(str, torch.Tensor)]:
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """
    features = features[0]
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    if 'guid' in first:
        if first['guid'] is not None:
            guid = first['guid'].item() if isinstance(first['guid'], torch.Tensor) else first['guid']
            dtype = torch.long if isinstance(guid, int) else torch.float
            batch['guid'] = torch.tensor([f['guid'] for f in features], dtype=dtype)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    else:
        if 'label_ids' in first:
            if first['label_ids'] is not None:
                if isinstance(first['label_ids'], torch.Tensor):
                    batch['labels'] = torch.stack([f['label_ids'] for f in features])
                else:
                    dtype = torch.long if type(first['label_ids'][0]) is int else torch.float
                    batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)
        for k, v in first.items():
            if k not in ('label', 'label_ids', 'guid') and v is not None:
                if isinstance(v, str) or isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    lengths = [len(f[k]) for f in features]
                    max_length = max(lengths)
                    paddings_lengths = [max_length - l for l in lengths]
                    batch[k] = torch.tensor([f[k] + [PAD_ID] * pl for f, pl in zip(features, paddings_lengths)], dtype=(torch.long))

        return batch