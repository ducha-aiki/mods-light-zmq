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

def describe_patches(model, image, DO_CUDA = True, DESCR_OUT_DIM = 128, BATCH_SIZE = 512):
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
    descriptors_for_net = np.clip(210*(descriptors_for_net + 0.45), 0 ,255).astype(np.uint8).astype(np.float32)
    print('processing', et, et/float(n_patches), ' per patch')
    return descriptors_for_net

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)
    
parser = argparse.ArgumentParser(description='Local patch descriptor server')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA inference')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--port', default='5555', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')



if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + args.port)
    model_weights = 'HardNet++.pth'
    DESCR_OUT_DIM = 128
    model = HardNet()
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
        descr = describe_patches(model, img, args.cuda, DESCR_OUT_DIM).astype(np.float32)
        buff = memoryview(descr)
        #buff = np.getbuffer(descr)
        socket.send(buff)
