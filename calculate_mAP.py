import numpy as np
import time
import os
# read train and test binarayCode
CURRENT_DIR = os.getcwd()

def getNowTime():
    return '['+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+']'


def getCode(train_codes,train_groudTruth,test_codes,test_groudTruth):

    line_number = 0
    with open(CURRENT_DIR+'/result.txt','r') as f:
        for line in f:
            temp = line.strip().split('\t')
            if line_number < 10000:
                test_codes.append([i if i==1 else -1  for i in map(int, list(temp[0]))])
                list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                list2[int(temp[1])] = 1
                test_groudTruth.append(list2) # get test ground truth(0-9)
            else:
                train_codes.append([i if i==1 else -1  for i in map(int, list(temp[0]))]) # change to -1, 1
                list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                list2[int(temp[1])] = 1
                train_groudTruth.append(list2) # get test ground truth(0-9)

            line_number += 1
    print getNowTime(),'read data finish'

def getHammingDist(code_a,code_b):
    dist = 0
    for i in range(len(code_a)):
         if code_a[i]!=code_b[i]:
             dist += 1
    return dist 


if __name__ =='__main__':
    print getNowTime(),'start!'

    train_codes = []
    train_groudTruth =[]
    
    test_codes = []
    test_groudTruth = []
    # get g.t. and binary code
    getCode(train_codes,train_groudTruth,test_codes,test_groudTruth)
    train_codes = np.array(train_codes)
    train_groudTruth = np.array(train_groudTruth)
    test_codes = np.array(test_codes)
    test_groudTruth = np.array(test_groudTruth)
    numOfTest = 1000

    # generate hanmming martix, g.t. martix  10000*50000
    gt_martix = np.dot(test_groudTruth, np.transpose(train_groudTruth))
    print getNowTime(),'gt_martix finish!'
    ham_martix = np.dot(test_codes, np.transpose(train_codes)) # hanmming distance map to dot value 
    print getNowTime(),'ham_martix finish!'

    # sort hanmming martix,Returns the indices that would sort an array.
    sorted_ham_martix_index = np.argsort(ham_martix,axis=1)
    
    # calculate mAP
    print getNowTime(),'sort ham_matrix finished,start calculate mAP'

    apall = np.zeros((numOfTest,1),np.float64)
    for i in range(numOfTest):
        x = 0.0
        p = 0
        test_oneLine = sorted_ham_martix_index[i,:]
        length = test_oneLine.shape[0]
        num_return_NN = 5000 # top 1000
        for j in range(num_return_NN):
             if gt_martix[i][test_oneLine[length-j-1]] == 1: # reverse
                 x += 1
                 p += x/(j+1)
        if p == 0:
            apall[i]=0
        else:
            apall[i]=p/x

    mAP = np.mean(apall)
    print getNowTime(),'mAP:',mAP
    

