## MODS with torchscript deep learning components

This is [MODS](https://github.com/ducha-aiki/mods) version, which allows you using state-of-the-art deep descriptors like HardNet via [LibTorch](https://pytorch.org/cppdocs/). 

Examples come with JIT converted  [PyTorch](https://github.com/pytorch/pytorch) [AffNet](https://github.com/ducha-aiki/affnet) and [HardNet+](https://github.com/DagnyT/hardnet) descriptor. For conversion example see this [notebook](https://github.com/DagnyT/hardnet/blob/master/notebook/convert_HardNet_to_JIT.ipynb).


## How to compile MODS 

I expect, that you have already installed latest PyTorch (0.5)

    cd build
    cmake ..
    make
 
## Image matching example

Relevant config-files are: config_aff_ori_desc_torchscript.ini and iters_HessianTS.ini
With Hessian-[AffNet, OriNet](https://github.com/ducha-aiki/affnet) and [HardNet++](https://github.com/DagnyT/hardnet) 



Now you can run matching:

    ./mods imgs/graf1.png imgs/graf6.png out1_deep.jpg out2_deep.jpg k1.txt k2.txt m.txt l.log 0 0 abcd.txt config_aff_ori_desc_torchscript.ini iters_HessianTS.ini
    
Expected output (on weak mobile GPU GT940):
    
    Maximum threads can be used: 8
    View synthesis, detection and description...
    Iteration 0
    HessianAffine: 1 synthesis will be done.
    Matching ... 
    Matching ... 
    3352 4109
    222 tentatives found.
    Duplicate filtering before RANSAC with threshold = 2 pixels.
    216 unique tentatives left
    LO-RANSAC(homography) verification is used...
    122 RANSAC correspondences got
    122 true matches are identified in 0.023 seconds
    Done in 1 iterations
    *********************
    Writing files... 
    HessianAffine 2
    HessianAffine 2
    Writing images with matches... done
    Image1: regions descriptors | Image2: regions descriptors 
    3730 3352 | 4524 4109

    True matches | unique tentatives
    122 | 216 | 56.5%  1st geom inc

    Main matching | All Time: 
    5.44 | 6.08 seconds
    Timings: (sec/%) 
    Synth|Detect|Orient|Desc|Match|RANSAC|MISC|Total 
    0.00778 1.16 0.919 1.43 0.228 0.023 2.32 6.08
    0.128 19.1 15.1 23.5 3.75 0.378 38.1 100




Now run with classical HessianAffine(Baumberg) + RootSIFT:

    ./mods imgs/graf1.png imgs/graf6.png out1_classic.jpg out2_classic.jpg k1.txt k2.txt m.txt l.log 0 0 abcd.txt config_affori_classic.ini iters_HessianSIFT.ini
    
Relevant config-files are: config_affori_classic.ini and iters_HessianSIFT.ini

    
Expected output:
    
    Maximum threads can be used: 8
    error loading the Affnet model
    error loading the OriNet model
    error loading the TorchScriptDescriptor model
    View synthesis, detection and description...
    Iteration 0
    HessianAffine: 1 synthesis will be done.
    Matching ... 
    Matching ... 
    2331 2909
    70 tentatives found.
    Duplicate filtering before RANSAC with threshold = 2 pixels.
    70 unique tentatives left
    LO-RANSAC(homography) verification is used...
    20 RANSAC correspondences got
    20 true matches are identified in 0.019 seconds
    Done in 1 iterations
    *********************
    Writing files... 
    HessianAffine 2
    HessianAffine 2
    Writing images with matches... done
    Image1: regions descriptors | Image2: regions descriptors 
    2665 2331 | 3285 2909

    True matches | unique tentatives
    20 | 70 | 28.6%  1st geom inc

    Main matching | All Time: 
    0.771 | 1.27 seconds
    Timings: (sec/%) 
    Synth|Detect|Orient|Desc|Match|RANSAC|MISC|Total 
    0.0112 0.132 0.0606 0.348 0.157 0.019 0.545 1.27
    0.881 10.4 4.76 27.3 12.3 1.49 42.8 100



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

## Saving in .npz format
Now you can save keypoints in .npz format. To do this, just pass k1.npz instead k1.txt in command line.
It will create .npz file with keys "xy", "responses", "scales", "A", "descs". 

https://github.com/ducha-aiki/mods-light-zmq/blob/master/imagerepresentation.cpp#L1266

Powered by great library https://github.com/rogersce/cnpy/


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
    
    
