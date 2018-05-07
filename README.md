## MODS with external deep learning components

This is [MODS](https://github.com/ducha-aiki/mods) version, which allows you using state-of-the-art deep descriptors like HardNet without linking MODS to any of deep learning library. 
It contains very small number of detectors and descriptors implemented inside -- for easier compilation. 
Instead it uses [zeromq](http://zeromq.org/) library for communication with separately run CNN daemons. 
Examples with python [PyTorch](https://github.com/pytorch/pytorch) [AffNet](https://github.com/ducha-aiki/affnet) and [HardNet++](https://github.com/DagnyT/hardnet) descriptors is provided, but you can use any language and any DL package you like, just modify corresponding scripts.

## How to compile MODS 

I expect, that you have already installed latest PyTorch (0.5)

    cd build
    cmake ..
    make
 
## Image matching example

Relevant config-files are: config_aff_ori_desc_zeromq.ini and iters_HessianZMQ.ini
With Hessian-[AffNet, OriNet](https://github.com/ducha-aiki/affnet) and [HardNet++](https://github.com/DagnyT/hardnet) 

    ./run_zmq_servers.sh

Wait until initialization on GPU is done and you see:

    Extracting on GPU
    Extracting on GPU
    Extracting on GPU

Now you can run matching:

    ./mods imgs/graf1.png imgs/graf6.png out1_deep.jpg out2_deep.jpg k1.txt k2.txt m.txt l.log 0 0 abcd.txt config_aff_ori_desc_zeromq.ini iters_HessianZMQ.ini
    
Expected output:
    
    Maximum threads can be used: 4
    View synthesis, detection and description...
    Iteration 0
    HessianAffine: 1 synthesis will be done.
    ('processing', 0.07718610763549805, 1.6556436644250974e-05, ' per patch')
    ('processing', 0.061591148376464844, 1.6110684900984786e-05, ' per patch')
    ('processing', 0.07169699668884277, 1.5837640090312078e-05, ' per patch')
    ('processing', 0.05922508239746094, 1.5873782470506817e-05, ' per patch')
    ('processing', 0.1312699317932129, 3.1877108254787004e-05, ' per patch')
    ('processing', 0.10080504417419434, 3.0019369914888128e-05, ' per patch')
    Matching ... 
    Matching ... 
    3358 4118
    264 tentatives found.
    Duplicate filtering before RANSAC with threshold = 2 pixels.
    254 unique tentatives left
    LO-RANSAC(homography) verification is used...
    147 RANSAC correspondences got
    147 true matches are identified in 0.003 seconds
    Done in 1 iterations
    *********************
    Writing files... 
    HessianAffine 2
    HessianAffine 2
    Writing images with matches... done
    Image1: regions descriptors | Image2: regions descriptors 
    3731 3358 | 4527 4118

    True matches | unique tentatives
    147 | 254 | 57.9%  1st geom inc

    Main matching | All Time: 
    2.02 | 2.52 seconds
    Timings: (sec/%) 
    Synth|Detect|Orient|Desc|Match|RANSAC|MISC|Total 
    0.011 0.721 0.568 0.463 0.229 0.003 0.527 2.52
    0.438 28.6 22.5 18.4 9.08 0.119 20.9 100


Don`t forget to kill server process after work done.

Now run with classical HessianAffine(Baumberg) + RootSIFT:

    ./mods imgs/graf1.png imgs/graf6.png out1_classic.jpg out2_classic.jpg k1.txt k2.txt m.txt l.log 0 0 abcd.txt config_affori_classic.ini iters_HessianSIFT.ini
    
Relevant config-files are: config_affori_classic.ini and iters_HessianSIFT.ini

    
Expected output:
    
    Maximum threads can be used: 4
    View synthesis, detection and description...
    Iteration 0
    HessianAffine: 1 synthesis will be done.
    Matching ... 
    Matching ... 
    2331 2912
    76 tentatives found.
    Duplicate filtering before RANSAC with threshold = 2 pixels.
    74 unique tentatives left
    LO-RANSAC(homography) verification is used...
    21 RANSAC correspondences got
    21 true matches are identified in 0.002 seconds
    Done in 1 iterations
    *********************
    Writing files... 
    HessianAffine 2
    HessianAffine 2
    Writing images with matches... done
    Image1: regions descriptors | Image2: regions descriptors 
    2665 2331 | 3287 2912

    True matches | unique tentatives
    21 | 74 | 28.4%  1st geom inc

    Main matching | All Time: 
    0.915 | 1.25 seconds
    Timings: (sec/%) 
    Synth|Detect|Orient|Desc|Match|RANSAC|MISC|Total 
    0.0106 0.183 0.0771 0.439 0.169 0.002 0.37 1.25
    0.85 14.6 6.16 35.1 13.5 0.16 29.6 100


As you can see, deep descriptors are much better, although slower
If you need to match really hard pairs, use iters_MODS_ZMQ.ini config file, or write your own configuation for view synthesis.


## If you need to extract and save features from directory with images:
    
Generate two text files, one with paths to the input images (one path per line) and one with output path for features. Then run extract_features_batch util.

    find imgs/* -type f > imgs_to_extract_list.txt
    mkdir output_features
    python get_output_fnames.py  imgs_to_extract_list.txt output_features extracted_features_fnames.txt
    ./run_zmq_servers.sh
    ./extract_features_batch imgs_to_extract_list.txt  extracted_features_fnames.txt config_aff_ori_desc_zeromq.ini iters_HessianZMQ.ini
    

Extracted features will be in output_features directory, in [OxAff-like](http://www.robots.ox.ac.uk/~vgg/research/affine/) format: x y a b c desc[128]

## Descriptor daemon script structure

It is simple python(might be any other language) script with following three main parts.
See [desc_server.py](build/desc_server.py) for example.

1) zeromq socket initialization: 
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + args.port)
    
port number should be the same, as listening port in corresponding section of [config_aff_ori_desc_zeromq.ini](build/config_aff_ori_desc_zeromq.ini) file:

    [zmqDescriptor]
    port=tcp://localhost:5555
    patchSize=32;  width and height of the patch
    mrSize=5.1962 ;

2)Waiting for input patches. Patches come as grayscale uint8 png image with size (ps * n_patches, ps), where ps is set in [config_aff_ori_desc_zeromq.ini](build/config_aff_ori_desc_zeromq.ini)

    while True:
        #  Wait for next request from client
        message = socket.recv()
        img = decode_msg(message).astype(np.float32)

3) Getting descriptors and sending them back, as numpy float32 (num_patches,desc_dim) array.
    
    descr = describe_patches(model, img, args.cuda, DESCR_OUT_DIM).astype(np.float32)
    buff = np.getbuffer(descr)
    socket.send(buff)

## Citation

Please cite us if you use this code:

    @article{Mishkin2015MODS,
          title = "MODS: Fast and robust method for two-view matching ",
          journal = "Computer Vision and Image Understanding ",
          year = "2015",
          issn = "1077-3142",
          doi = "http://dx.doi.org/10.1016/j.cviu.2015.08.005",
          url = "http://www.sciencedirect.com/science/article/pii/S1077314215001800",
          author = "Dmytro Mishkin and Jiri Matas and Michal Perdoch"
          }
    
And if you use provided deep descriptors, please cite:

    @article{HardNet2017,
    author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins: Local descriptor learning loss}",
    booktitle = {Proceedings of NIPS},
    year = 2017,
    month = dec}
    
    @article{AffNet2017,
    author = {Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Learning Discriminative Affine Regions via Discriminability}",
    year = 2017,
    month = nov}
    
    