# DSH_tensorflow
implemement of DEEP SUPERVISED HASHING FOR FAST IMAGE RETRIEVAL_CVPR2016
train:
    python demo.py
    First you need to modify demo.py to ensure TRAIN=True,and you can modify HASHING_BITS to generate any length hashing bit.
	After training, in dir /logs will save trained model.

test:
    After training, you can change TRAIN=False,then run python.demo.py,result.txt will be generated.

calculate mAP:
    run python calculate_mAP.py to get mAP of this result.  
