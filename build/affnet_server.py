import argparse
import sys
import time
import numpy as np
import zmq
import cv2

# Hardnet-related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#

def decode_msg(message):
    A = cv2.imdecode(np.frombuffer(message,dtype=np.uint8), 0)
    return A

def describe_patches(model, image, DO_CUDA = True, DESCR_OUT_DIM = 3, BATCH_SIZE = 512):
    h,w = image.shape
    n_patches = int(h/w)
    t = time.time()
    patches = np.ndarray((n_patches, 1, 32, 32), dtype=np.float32)
    for i in range(n_patches):
        patches[i,0,:,:] =  image[i*(w): (i+1)*(w), 0:w]
    outs = []
    n_batches = int(n_patches / BATCH_SIZE) + 1
    t = time.time()
    descriptors_for_net = np.zeros((len(patches), DESCR_OUT_DIM))
    for i in range(0, len(patches), BATCH_SIZE):
        data_a = patches[i: i + BATCH_SIZE, :, :, :].astype(np.float32)
        data_a = torch.from_numpy(data_a)
        if DO_CUDA:
            data_a = data_a.cuda()
        data_a = Variable(data_a)
        with torch.no_grad():
            out_a = model(data_a)
        descriptors_for_net[i: i + BATCH_SIZE,:] = out_a.data.cpu().numpy().reshape(-1, DESCR_OUT_DIM)
    assert n_patches == descriptors_for_net.shape[0]
    et  = time.time() - t
    print('processing', et, et/float(n_patches), ' per patch')
    return descriptors_for_net

class AffNetFast(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 3, kernel_size=8, stride=1, padding=0, bias = True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def forward(self, input, return_A_matrix = False):
        xy = self.features(self.input_norm(input)).view(-1,3)
        xy[:,0] +=1
        xy[:,2] +=1
        return xy
    
parser = argparse.ArgumentParser(description='Local affine shape estimator server')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA inference')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--port', default='5556', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:"+ args.port)
    model_weights = 'AffNet.pth'
    model = AffNetFast()
    DESCR_OUT_DIM = 3
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if args.cuda:
        model.cuda()
        print('Extracting on GPU')
    else:
        print('Extracting on CPU')
        model = model.cpu()
    while True:
        #  Wait for next request from client
        message = socket.recv()
        img = decode_msg(message).astype(np.float32)
        descr = describe_patches(model, img, DO_CUDA =  args.cuda, DESCR_OUT_DIM = DESCR_OUT_DIM).astype(np.float32)
        buff = np.getbuffer(descr)
        socket.send(buff)