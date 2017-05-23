
1) Place All the training image folders under train/
2) Place All the stage 1 test images under test_stg1/
3) Place All the stage 2 test images under test_stg2/
4) download the pre-trained alexnet weights file "bvlc_alexnet.caffemodel" and place it in folder scripts_alexnet/models/bvlc_alexnet_after_augmentation/
4) cd scripts_alexnet
5) Run caffe_train.py 
//this would store images in lmdb format which is a requirement for caffe

6) ~/caffe/build/tools/caffe train --solver=models/bvlc_alexnet_after_augmentation/solver.prototxt -weights models/bvlc_alexnet_after_augmentation/bvlc_alexnet.caffemodel 
//this would train the model

7) Run caffe_generate_test_results.py
//this would generate the test results file that can be uploaded in kaggle after inserting the appropriate header consisting of column names

~/caffe/build/tools/caffe train --solver=models/bvlc_alexnet_new/solver.prototxt --weights=models/bvlc_alexnet_new/bvlc_alexnet.caffemodel
