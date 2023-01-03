"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataloader given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    return data_loader

# opt, set_name=['trn', 'val', 'tst']
def create_dataset_with_args(opt, **kwargs):
    """Create two dataloader given the option, dataset may have additional args.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_split_dataset
        >>> dataset = create_dataset(opt, set_name=['trn', 'val', 'tst'])
        This will create 3 datasets, each one get different parameters
        for the specific dataset class, __init__ func must get one parameter
        eg: dataset.__init__(self, set_name='trn'): ....
    """
    _kwargs = []
    for key in kwargs: # 'set_name'
        value = kwargs[key] # 'trn', 'val', 'tst'
        if not isinstance(value, (list, tuple)):
            value = [value]
        lens = len(value) # lens = 3
        _kwargs += list(map(lambda x: {}, range(lens))) if len(_kwargs) == 0 else []
        for i, v in enumerate(value):
            _kwargs[i][key] = v 
    
    # _kwargs: [{'set_name': 'trn'}, {'set_name': 'val'}, {'set_name': 'tst'}]    
    dataloaders = tuple(map(lambda x: CustomDatasetDataLoader(opt, **x), _kwargs))
    return dataloaders if len(dataloaders) > 1 else dataloaders[0]


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""
    ## kwargs: [{'set_name': 'trn'}, {'set_name': 'val'}, {'set_name': 'tst'}]
    def __init__(self, opt, **kwargs):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode) # 'data.multimodal_dataset.MultimodalDataset'
        self.dataset = dataset_class(opt, **kwargs)
        # print("dataset [%s] was created" % type(self.dataset).__name__)
        
        ''' Whether to use manual collate function defined in dataset.collate_fn'''
        if self.dataset.manual_collate_fn: 
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads),
                drop_last=False,
                collate_fn=self.dataset.collate_fn
            )

        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads),
                drop_last=False
            )

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
