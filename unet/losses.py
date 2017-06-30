'''loss function modules'''
import functools

from keras import backend as K

def get_loss(loss_func, loss_args=None):
	'''Return {loss_func}, either defined in this module or just a string'''
	if loss_func in globals():
		func = globals()[loss_func]
		if loss_args is None:
			return func
		try:
			return func(**loss_args)
		except TypeError:
			func = functools.partial(func, **loss_args)
			func.__name__ = loss_func
			return func
		#else:
		#	return globals()[loss_func](**loss_args)
	return loss_func # not defined in this module

def dice_coef(y_true, y_pred, smooth, pred_mul, p_ave, is_loss=False, per_sample=False):
	'''
	Return the dice coefficent between {y_true} and {y_pred} Normally defined as:
	 dice = 2*sum(y_true && y_pred) / sum(y_true || y_pred)
	where the sum is over all samples

	Additionally can be tuned by parameters smooth, pred_mul:
	    pred_mul : multiplicative factor to predict
	    smooth   : smooth out division by zero when y~0?
	and computed by summing over samples (of a batch) or averaging:

	That is, compute:
	    dice = (1-p_ave)*dice_sum + dice_avg * p_ave,
	where,
	    dice_sum =  |2*sum(y_true && y_pred) + smooth| / |sum(y_true || y_pred * pred_mul) + smooth|
	    dice_avg = <(2*sum(y_true && y_pred) + smooth) / (sum(y_true || y_pred * pred_mul) + smooth)>
	where,
	    sum(*)  = sum over pixels and channels
	    |*|     = sum over batch samples
	    <*>     = average over batch samples

	If {is_loss} return -dice to be minimized.
	if {per_sample} is True, return value evaluated on each sample of {y_true}
	and {y_pred} rather than sum/averaged over the batch axis.
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

	if per_sample:
		dice_final = (2.0*isect_avg+smooth)/(union_avg+smooth)
	else:
		isect_all = K.sum(isect_avg)
		union_all = K.sum(union_avg)

		dice_avg = K.mean((2.0*isect_avg+smooth)/(union_avg+smooth))
		dice_all =       ((2.0*isect_all+smooth)/(union_all+smooth))
		# mix the two by p_ave
		dice_final = (1.-p_ave) * dice_avg + p_ave * dice_all
	return -dice_final if is_loss else dice_final

def dice_coef_loss(smooth=10, pred_mul=1.0, p_ave=0.6, per_sample=False):
	'''Return a partial for use as training loss function'''
	func = functools.partial(dice_coef, is_loss=True,
			smooth     = smooth,
			pred_mul   = pred_mul,
			p_ave      = p_ave,
			per_sample = per_sample)
	func.__name__ = 'dice_coef_loss'
	return func

## basically the same as dice_coef_loss:
def weighted_dice_coef(y_true, y_pred, smooth, pred_weight, all_weight, is_loss=False, per_sample=False):
	'''Alternative definition of dice coeffcient, where we use component weights parameters, between 0 and 1, instead of straight multipliers.
	That is, let
	   L(x, y; p) = (1-p)*x+p*y
	compute
	  dice = L(dice_avg, dice_all; all_weight)
	where
	  dice_all =  |sum(y_true*y_pred)+smooth| / |sum(L(y_true, y_pred; pred_weight))+smooth|
	  dice_avg = <(sum(y_true*y_pred)+smooth) / (sum(L(y_true, y_pred; pred_weight))+smooth)>
	where
	  sum(*), |*|, and <*> are as defined above.
	'''
	isect_avg = K.sum(y_true*y_pred, axis=(1,2,3))
	union_avg = K.sum(y_true, axis=(1,2,3))*(1-pred_weight) + K.sum(y_pred, axis=(1,2,3))*pred_weight

	if per_sample: # return value for each sample rather than sum/mean over batch
		dice_final = (isect_avg+smooth) / (union_avg+smooth)
	else:
		isect_all = K.sum(isect_avg)
		union_all = K.sum(y_true)*(1-pred_weight) + K.sum(y_pred)*pred_weight

		dice_avg  = K.mean((isect_avg+smooth) / (union_avg+smooth))
		dice_all  =        (isect_all+smooth) / (union_all+smooth)
		dice_final = (1-all_weight)*dice_avg + all_weight*dice_all

	return -dice_final if is_loss else dice_final

def weighted_dice_coef_loss(smooth=5, pred_weight=0.5, all_weight=0.6, per_sample=False):
	'''Return a partial for use as training loss function'''
	func = functools.partial(weighted_dice_coef, is_loss=True,
			smooth      = smooth,
			pred_weight = pred_weight,
			all_weight  = all_weight,
			per_sample  = per_sample)
	func.__name__ = 'weighted_dice_coef_loss'
	return func

## binary cross entropy
def weighted_binary_crossentropy(y_true, y_pred, pos_weight=0.5, axis=None):
	'''Evaluate weighted binary crosstropy as:
		k p log q + (1-k) (1-p) log (1-q)
	with y_true -> p, y_pred -> q, pos_weight -> k
	'''
	epsilon = K.epsilon()
	y_pred = K.clip(y_pred, epsilon, 1-epsilon)
	xentropy = pos_weight * y_true * K.log(y_pred) + (1-pos_weight)*(1-y_true) * K.log(1-y_pred)
	return K.tf.negative(K.mean(xentropy, axis=axis))

def combination_loss(y_true, y_pred, xe_multi=5, xe_power=1.0, xe_kwds=None, dc_kwds=None):
	'''Implements a loss function that combines dice-coefficent and binary crossentropy

	  loss = 1 + weighted_dice_coef_loss + (xe_multi * weighted_binary_crossentropy)**xe_power

	:param dc_kwds: is the keyword arguments to be passed to `weighted_dice_coef_loss`
	:param xe_kwds: is the keyword arguments to be passed to `weighted_binary_crossentropy`
	'''
	loss_dc = weighted_dice_coef_loss(**(dc_kwds or {}))(y_true, y_pred)
	loss_xe = weighted_binary_crossentropy(y_true, y_pred, **(xe_kwds or {}))
	return 1 + loss_dc + K.pow((xe_multi * loss_xe), xe_power)

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
