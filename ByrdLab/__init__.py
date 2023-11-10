import torch
from argsParser import gpu

# torch.ShortTensor  ---  torch.int16
# torch.IntTensor    ---  torch.int32 / torch.int
# torch.LongTensor   ---  torch.int64

# torch.HalfTensor   ---  torch.float16
# torch.FloatTensor  ---  torch.float32
# torch.DoubleTensor ---  torch.float64

FEATURE_TYPE = torch.float64
TARGET_TYPE = torch.int16
VALUE_TYPE = torch.float64



DEVICE = torch.device("cuda:" + str(gpu))
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda:1")