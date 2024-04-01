from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['ff-native', 'fff-native', 'ff-copying', 'fff-copying', 'ff-cpp', 'fff-cpp', 'ff-cuda', 'fff-cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=16384)
parser.add_argument('-w', '--input-width', type=int, default=768)
parser.add_argument('--depth', type=int, default=11)
parser.add_argument('-p', '--hidden-width', type=int, default=3072)
parser.add_argument('-r', '--runs', type=int, default=1)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
parser.add_argument('-s', '--parallel_size', type=int, default=1)
options = parser.parse_args()

# THIS IF NEEDS A COMPLETE REWORK
if options.example == 'ff-native':
    from ff_native.ff_native import FF
elif options.example == 'fff-native':
    from fff_native.fff_native import FFF
elif options.example == 'fff-bmm':
    from fff_bmm.fff_bmm import FFF
elif options.example == 'ff-bmm':
    from ff_bmm.ff_bmm import FF
elif options.example == 'ff-cuda':
    from ff_cuda.ff import FF
    options.cuda = True # hardcoded override of the arg flag -- you can't do CUDA on a CPU
elif options.example == 'fff-cuda':
    from fff_cuda_folder.fff import FFF
    from fff_cuda_folderOG.fff import FFF as FFFOG
    options.cuda = True # hardcoded override of the arg flag -- you can't do CUDA on a CPU
else:
    raise ValueError('Unknown example: {}'.format(options.example))

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
X = torch.randn(options.batch_size, options.input_width, **kwargs)
if options.example.startswith('ff-'):
    fff = FF(options.input_width, options.hidden_width, options.input_width).to(device, dtype)
else:
    fff = FFF(options.input_width, options.input_width, options.depth, options.parallel_size).to(device, dtype)
    fffOG = FFFOG(options.input_width, options.input_width, options.depth, options.parallel_size).to(device, dtype)

for param_tensor_ff, param_tensor_ffOG in zip(fff.parameters(), fffOG.parameters()):
    param_tensor_ff.data = param_tensor_ffOG.data.clone()

outputs = fff(X)
outputsOG = fffOG(X)

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0

forward_minOG = math.inf
forward_timeOG = 0
backward_minOG = math.inf
backward_timeOG = 0
fff.eval()
fffOG.eval()
with torch.no_grad():
    for i in range(options.runs):
        for param_tensor_ff, param_tensor_ffOG in zip(fff.parameters(), fffOG.parameters()):
            param_tensor_ff.data = param_tensor_ffOG.data.clone()
        X = torch.randn(options.batch_size, options.input_width, **kwargs)
        #X = torch.ones(options.batch_size, options.input_width, **kwargs)*(options.runs-i)
        X_copy = X.clone()

        start = time.time()
        outputs = fff(X)
        elapsed = time.time() - start
        print('Outputs: ', outputs)
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

        startOG = time.time()
        outputsOG = fffOG(X_copy)
        elapsedOG = time.time() - startOG
        print('OutputsOG: ', outputsOG)
        forward_minOG = min(forward_minOG, elapsedOG)
        forward_timeOG += elapsedOG
        percent_diff = (torch.abs(outputs - outputsOG) / torch.abs(outputsOG)).mean()
        print('Percent diff: ', percent_diff)

#TODO: check for correctness

scale = TIME_SCALES[options.scale]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

forward_minOG *= scale
backward_minOG *= scale
forward_averageOG = forward_timeOG / options.runs * scale
backward_averageOG = backward_timeOG / options.runs * scale

print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    options.scale))

print('OG: Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
    forward_minOG, forward_averageOG, backward_minOG, backward_averageOG,
    options.scale))

print('Speedup: {0:.3f}x'.format(forward_averageOG / forward_average))
