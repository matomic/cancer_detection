# -*- coding: utf-8 -*-
'''Utility functions'''
from __future__ import print_function
from __future__ import division

import collections
import functools
import simplejson as json

import pandas as pd
import numpy as np
from keras import backend as K

# debugging
#from pprint import pprint
#from ipdb import set_trace


def hist_summary(folds_history):
    '''(つ・▽・)つ⊂(・▽・⊂)'''
    res = None
    for h in folds_history.values():
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

#def clean_segmentation_img(img, cut=200):
#    ''''''
#    seg_binary = np.zeros_like(img)
#    _,sb = cv2.threshold(np.copy(img)*255, 127, 255, cv2.THRESH_BINARY)
#    im = sb.astype(np.uint8)
#    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    L = len(contours[1])
#    if L == 0:
#        return seg_binary
#    cnt_area = [cv2.contourArea(cnt) for cnt in contours[1]]
#
#    idx = np.argmax(cnt_area)
#    cnts = [contours[1][idx]]
#    area = cnt_area[idx]
#    if area<cut:
#        return seg_binary
#
#    cimg = np.zeros_like(im)
#    cv2.drawContours(cimg, cnts, 0, color=255, thickness=-1)
#    seg_binary[cimg == 255] = 1
#    return seg_binary

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

####### Generic utils #######
class dotdict(dict):
    """ A dictionary whose attributes are accessible by dot notation.
    This is a variation on the classic `Bunch` recipe (which is more limited
    and doesn't give you all of dict's methods). It is just like a dictionary,
    but its attributes are accessible by dot notation in addition to regular
    `dict['attribute']` notation. It also has all of dict's methods.
    .. doctest::
        >>> dd = dotdict(foo="foofoofoo", bar="barbarbar")
        >>> dd.foo
        'foofoofoo'
        >>> dd.foo == dd['foo']
        True
        >>> dd.bar
        'barbarbar'
        >>> dd.baz
        >>> dd.qux = 'quxquxqux'
        >>> dd.qux
        'quxquxqux'
        >>> dd['qux']
        'quxquxqux'
    NOTE:   There are a few limitations, but they're easy to avoid
            (these should be familiar to JavaScript users):
        1.  Avoid trying to set attributes whose names are dictionary methods,
            for example 'keys'. Things will get clobbered. No good.
        2.  You can store an item in a dictionary with the key '1' (`foo['1']`),
            but you will not be able to access that attribute using dotted
            notation if its key is not a valid Python variable/attribute name
            (`foo.1` is not valid Python syntax).
    FOR MORE INFORMATION ON HOW THIS WORKS, SEE:
    - http://stackoverflow.com/questions/224026/
    - http://stackoverflow.com/questions/35988/
    """
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(attr)

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__

    @classmethod
    def recursiveFrom(cls, d):
        '''Generate instance recursively from dict {d}'''
        return cls({
            k : cls.recursiveFrom(v) if isinstance(v, dict) else v
            for k, v in d.items()})

def get_json_type(obj):
    """Serialize any object to a JSON-serializable structure.
    # Arguments
        obj: the object to serialize
    # Returns
        JSON-serializable structure representing `obj`.
    # Raises
        TypeError: if `obj` cannot be serialized.
    """
    # if obj is a serializable Keras class instance
    # e.g. optimizer, layer
    if hasattr(obj, 'get_config'):
        return {'class_name': obj.__class__.__name__,
                'config': obj.get_config()}

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        return obj.item()

    # misc functions (e.g. loss function)
    if callable(obj):
        return obj.__name__

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable:', obj)

jsondumps = functools.partial(json.dumps, indent=1, sort_keys=True, separators=(',',':'), default=get_json_type)
jsondump = functools.partial (json.dump,  indent=1, sort_keys=True, separators=(',',':'), default=get_json_type)

def safejsondump(j, f, *args, **kwds):
    '''Dump json serializable object {j} to file {f}, break on exception so that we don't end up have partially written file without review'''
    j = numpy4json(j)
    jsondumps(j)
    jsondump(j, open(f,'w'), *args, **kwds)

def numpy4json(j, seen_set=None):
    '''clean up object {j} for JSON serialization'''
    if isinstance(j, collections.Mapping):
        return { k : numpy4json(v) for k, v in j.items() }
    if isinstance(j, tuple(np.typeDict.values())):
        return j.tolist()
    #if isinstance(j, collections.Iterable):
    #    return [numpy4json(v) for v in j]
    return j

def config_json(config_dict):
    '''Return a JSON-serializable structure of config'''
    return {
            k: v for k, v in config_dict.items()
            if not k.startswith('_')
            and isinstance(v, (int, float, str, dict, list, tuple, set))
            }

# eof
