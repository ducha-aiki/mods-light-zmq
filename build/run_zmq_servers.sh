#!/bin/bash
((python -utt affnet_server.py --gpu-id=0 --port=5556 &)&)  
((python -utt orinet_server.py --gpu-id=0 --port=5557 &)&)  
((python -utt desc_server.py --gpu-id=0 --port=5555 &)&)  
