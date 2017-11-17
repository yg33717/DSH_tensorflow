# DSH_tensorflow
Implemement of DEEP SUPERVISED HASHING FOR FAST IMAGE RETRIEVAL_CVPR2016


## Train: 
`python demo.py`

First you need to modify `demo.py` to ensure `TRAIN=True`,and you can modify `HASHING_BITS` to generate any length hashing bit.After training, in dir `/logs` will save trained model.

## Test:
After training, you can change `TRAIN=False`,then run `python.demo.py`,result.txt will be generated.

## Calculate mAP:
run `python calculate_mAP.py` to get mAP of this result.
