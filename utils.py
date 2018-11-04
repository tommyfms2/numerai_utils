

import numpy as np
import csv


def refinePreds(predsDatas):
    # adjustment
    for (j,_) in enumerate(predsDatas):
        predsDatas[j][predsDatas[j]>1.0] = 1.0
        predsDatas[j][predsDatas[j]<0.0] = 0.0
    # only majority rate plused version.
    numofPreds = len(predsDatas)
    finalPreds = np.zeros(len(predsDatas[0]))
    for i in np.arange(len(predsDatas[0])):

        num_of_1 = 0
        forPlus = [0,0]
        for j in np.arange(numofPreds):
            rnd = np.round(predsDatas[j][i])
            num_of_1 += rnd
            forPlus[int(rnd)] += (predsDatas[j][i]-0.5)
        idx = int(num_of_1//(numofPreds//2+1))
        fordiv = abs(num_of_1-idx-2)+3
        finalPreds[i] = 0.5 + forPlus[idx]/fordiv
        if finalPreds[i]>1.0 or finalPreds[i]<0.0:
            break

    return finalPreds

def mean_ensemble(predsDatas):
    npPredsDatas = np.array(predsDatas, dtype=np.float32)
    print(npPredsDatas.shape)
    finalPreds = npPredsDatas.mean(axis=0)
    return finalPreds


def getLogloss(preds, targets):
    s = 0
    eps = 1e-15
    preds = np.array(preds, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    targets = np.array(targets, dtype=np.int32)
    preds = np.clip(preds, eps, 1-eps)

    for (p, t) in zip(preds, targets):
        if t==0:
            s += np.log(1-p)
        else:
            s += np.log(p)

    return -s / len(preds)


def getConsystancy(preds, targets, eras):
    erasmap = {}
    for (pred, target, era) in zip(preds, targets, eras):
        if not era in erasmap:
            erasmap[era] = []
        erasmap[era].append([pred, target])

    consistancy = 0.0
    for erakey in erasmap:
        dntofthisera = np.array(erasmap[erakey])
        logloss = getLogloss(dntofthisera.transpose(1,0)[0], dntofthisera.transpose(1,0)[1])
        if logloss < 0.693:
            consistancy += 1.0

    consistancy = consistancy/len(erasmap.keys()) * 100
    return consistancy


def saveProbability(filename, ids, finalPreds, firstraw=['id', 'probability']):
    mat2d = []
    mat2d.append(firstraw)
    for (id, pred) in zip(ids, finalPreds):
        mat2d.append([id,pred])
    with open(str(filename)+'.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(mat2d)

