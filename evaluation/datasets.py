from tester.evaluation.data import SequenceList
from tester.evaluation.resotdataset import RESOTDataset


def load_dataset(dataset_name, dataset_path):
    name = dataset_name.lower()
    if name == "rseot":
        Dataset = RESOTDataset(dataset_path)
        return Dataset.get_sequence_list()
    else:
        assert name in ["rseot"], "只有rseot"


def get_dataset(dataset_name, dataset_path):
    dset = SequenceList()
    dset.extend(load_dataset(dataset_name, dataset_path))
    return dset