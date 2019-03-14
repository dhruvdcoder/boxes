from torch.utils.data.dataloader import DataLoader, _DataLoaderIter, pin_memory_batch


######################################
# Data Object Wrappers
######################################

class _TensorDataLoaderIter(_DataLoaderIter):
    """
    DataLoaderIter class for Datasets which support native advanced indexing, eg. Tensors.

    The natural DataLoaderIter class in PyTorch returns batches as a list of elements, which
    is much slower when the underlying data already supports advanced indexing.
    """

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            # NOTE: This is the only line which has been changed!
            batch = self.dataset[indices]
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)


class TensorDataLoader(DataLoader):
    """
    DataLoader class for Datasets which support native advanced indexing, eg. Tensors.

    The natural DataLoader class in PyTorch returns batches as a list of elements, which
    is much slower when the underlying data already supports advanced indexing.
    """

    def __iter__(self):
        return _TensorDataLoaderIter(self)
