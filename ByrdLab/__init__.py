import torch

# torch.ShortTensor  ---  torch.int16
# torch.IntTensor    ---  torch.int32 / torch.int
# torch.LongTensor   ---  torch.int64

# torch.HalfTensor   ---  torch.float16
# torch.FloatTensor  ---  torch.float32
# torch.DoubleTensor ---  torch.float64

FEATURE_TYPE = torch.float64
TARGET_TYPE = torch.int16
VALUE_TYPE = torch.float64