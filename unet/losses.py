'''loss function modules'''
import functools

from keras import backend as K

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
	isect_avg = K.sum(y_true*y_pred, axis=[1,2,3])
	union_avg = K.sum(y_true, axis=[1,2,3])+K.sum(y_pred, axis=[1,2,3])*pred_mul
	#return K.mean((2. * intersection + smooth) / (union + smooth))

	isect_all = K.sum(y_true*y_pred)
	union_all = K.sum(y_true)+K.sum(y_pred)*pred_mul

	dice_avg = K.mean((2.0*isect_avg+smooth)/(union_avg+smooth))
	dice_all =        (2.0*isect_all+smooth)/(union_all+smooth)
	# mix the two by p_ave
	dice_mix = (1.-p_ave) * dice_avg + p_ave * dice_all
	return -dice_mix if negate else dice_mix


def dice_coef_loss(smooth=10, pred_mul=1.0, p_ave=0.6):
	'''Return a partial for use as training loss function'''
	func = functools.partial(dice_coef, negate=True, smooth=smooth, pred_mul=pred_mul, p_ave=p_ave)
	func.__name__ = 'dice_coef_loss'
	return func

# eof
