from data_provider.data_loader import UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy
}


def collate_fn(batch):
    """
    Collate function that pads examples of different lengths.
    
    Args:
        batch (list): A list of examples from the dataset.
    
    Returns:
        A tuple of:
        - padded_inputs: A padded tensor of the input features.
        - input_lengths: A tensor of the original lengths of the inputs.
        - targets: A padded tensor of the target labels.
        - target_lengths: A tensor of the original lengths of the targets.
    """
    # Unpack the examples
    seq_x, seq_y, seq_x_mark, seq_y_mark = tuple(map(list, zip(*batch)))
    
    # Pad the input sequences
    padded_seq_x = pad_sequence([torch.tensor(x) for x in seq_x], batch_first=True, padding_value=0)
    lengths = torch.tensor([len(x) for x in seq_x])
    
    # Pad the target sequences
    padded_seq_y = pad_sequence([torch.tensor(y) for y in seq_y], batch_first=True, padding_value=0)
    
    # Convert seq_x_mark and seq_y_mark to tensors
    seq_x_mark = torch.stack([torch.tensor(mark) for mark in seq_x_mark])
    seq_y_mark = torch.stack([torch.tensor(mark) for mark in seq_y_mark])
    
    return padded_seq_x, padded_seq_y, seq_x_mark, seq_y_mark, lengths

def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.input_token_len, args.output_token_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )
    print(flag, len(data_set))
    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn if 'Utsd' in args.data else None
        )
    return data_set, data_loader
