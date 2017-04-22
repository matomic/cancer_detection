import pandas as pd
import numpy as np
from keras import backend as K

def hist_summary(folds_history):
    res = None
    for i,h in folds_history.items():
        if res is None:
            val_names = [k for k in h.keys() if k.startswith('val_')]
            res = {n:[] for n in val_names}
            res['epoch'] = []
        e = np.argmin(h['val_loss'])
        res['epoch'].append(e)
        for n in val_names:
            res[n].append(h[n][e])
    res = pd.DataFrame(res)
    if res.shape[0]>1:#calcualte the mean
        res = res.append(res.mean,ignore_index=True)
    return res

def clean_segmentation_img(img, cut=200):
    seg_binary = np.zeros_like(img)
    _,sb = cv2.threshold(np.copy(img)*255, 127, 255, cv2.THRESH_BINARY)
    im = sb.astype(np.uint8)
    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    L = len(contours[1])
    if L == 0:
        return seg_binary
    cnt_area = [cv2.contourArea(cnt) for cnt in contours[1]]

    idx = np.argmax(cnt_area)
    cnts = [contours[1][idx]]
    area = cnt_area[idx]
    if area<cut:
        return seg_binary

    cimg = np.zeros_like(im)
    cv2.drawContours(cimg, cnts, 0, color=255, thickness=-1)
    seg_binary[cimg == 255] = 1
    return seg_binary

def dice_coef(y_true, y_pred, smooth, pred_mul, p_ave, negate=False):
    '''Return the dice coefficent between {y_true} and {y_pred}
    Normally defined as:
        dice = 2*sum(y_true && y_pred) / sum(y_true || y_pred)
    where the sum is over all samples

    Additionally can be turned by parameters smooth, pred_mul:
        pred_mul : multiplicative factor to predict
        smooth   : smooth out division by zero when y~0?
        dice' = (2*sum(y_true&&y_pred*pred_mul)+smooth)/(sum(y_true||y_pred*pred_mul)+smooth)
    '''
    #y_true_f = K.flatten(y_true)
    #y_pred_f = K.flatten(y_pred)
    ### this method: combine all as a single big image
    #intersection = K.sum(y_true_f * y_pred_f)
    #return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    #each image has its own score
    isect_avg = K.sum(y_true*y_pred,axis=[1,2,3])
    union_avg = K.sum(y_true, axis=[1,2,3])+K.sum(y_pred,axis=[1,2,3])*pred_mul
    #return K.mean((2. * intersection + smooth) / (union + smooth))

    isect_all = K.sum(y_true*y_pred)
    union_all = K.sum(y_true)+K.sum(y_pred)*pred_mul

    dice_avg = K.mean((2.0*isect_avg+smooth)/(union_avg+smooth))
    dice_all =        (2.0*isect_all+smooth)/(union_all+smooth)
    # mix the two by p_ave
    dice_mix = (1.-p_ave) * dice_avg + p_ave * dice_all
    return -dice_mix if negate else dice_mix

def dice_coef_loss_gen(smooth=10, pred_mul=1.0, p_ave=0.6):
    '''Generate dice_coef loss function by partially binding arguments'''
    def loss(y_true, y_pred):
        '''loss function'''
        return -dice_coef(y_true, y_pred, smooth, pred_mul, p_ave)
    return loss
