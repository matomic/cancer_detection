# -*- coding: utf-8 -*-
'''Script to Preprocess LUNA2016 images'''
from __future__ import print_function
from __future__ import division

# stdlib
import argparse
import glob
import os
import sys

# 3rd-party
#import SimpleITK as sitk
#import csv
#from matplotlib import pyplot as plt
from pandas import read_csv
from skimage import draw, measure, transform
#from skimage.feature import canny
from skimage.morphology import binary_dilation, binary_erosion, disk#, square
from tqdm import tqdm
import SimpleITK as sitk
import h5py
import numpy as np

# in-house
from console import PipelineApp

## debugging
from pprint import pprint
try:
	from ipdb import set_trace
except Exception:
	pass # jupyter notebook doesn't like reimporting ipdb

#def load_itk_image(filename):
#    '''load mhd/raw {filename}'''
#    itkimage = sitk.ReadImage(filename)
#    numpyImage = sitk.GetArrayFromImage(itkimage) # axis, sagittal, coronal
#
#    numpyOrigin = np.array(itkimage.GetOrigin()) #x,y,z
#    numpySpacing = np.array(itkimage.GetSpacing())
#
#    return numpyImage, numpyOrigin, numpySpacing
#
#def worldToVoxelCoord(worldCoord, origin, spacing):
#    '''
#    only valid if there is no rotation component
#    '''
#    voxelCoord = np.rint((worldCoord-origin)/spacing).astype(np.int) # pylint: disable=no-member
#    return voxelCoord

def normalize(x, vmin=-1000, vmax=200, left=0, right=255, dtype=np.uint8):
	'''Normalize scan, mapping -1000 HU to 0 in gs and 200 HU to 255 gs'''
	y = np.interp(x, [vmin, vmax], [left, right], left=left, right=right).astype(dtype)#0-255, to save disk space
	return y

def get_img_mask(scan, h, nodules, nth=-1, z=None, rho_min=3):
	"""
	h = spacing_z/spacing_xy
	nodules = list (x,y,z,d) of the nodule, in Voxel space
	specify nth or z. nth: the nth nodule

	Returns:
		img : The z-th axis-slice of scan
		res : mask with disks with diameter d, centered on (x,y,z), for each (x,y,z,d) of nodules
	"""
	if z is None:
		z = int(nodules[nth][2])
	img = normalize(scan[z,...])
	res = np.zeros(img.shape)
	#draw nodules
	for n_x, n_y, n_z, n_d in nodules:
		r = n_d / 2.
		dz = np.abs((n_z-z)*h)
		if dz >= r:
			continue
		rho = np.sqrt(r**2-dz**2) # on-slice radius rho
		if rho < rho_min:
			continue
		# create contour at xyzd[0],xyzd[1] with radius rho
		rr, cc = draw.circle(n_y, n_x, rho)
		res[rr, cc] = 1
	return img, res

#def segment_lung_mask2(image, fill_lung_structures=True, speedup=4):
#    ''' ((*´-ω・｀人´・ω-｀*)) '''
#    def largest_label_volume(im, bg=-1):
#        '''missing-docstring'''
#        vals, counts = np.unique(im, return_counts=True)
#
#        counts = counts[vals != bg]
#        vals = vals[vals != bg]
#
#        if len(counts) > 0:
#            return vals[np.argmax(counts)]
#        else:
#            return None
#
#    if speedup>1:
#        smallImage = transform.downscale_local_mean(image,(1,speedup,speedup))
#    else:
#        smallImage = image
#    # not actually binary, but 1 and 2.
#    # 0 is treated as background, which we do not want
#    binary_image = np.array(smallImage > -320, dtype=np.int8)+1
#    labels = measure.label(binary_image)
#
#    # Pick the pixel in the very corner to determine which label is air.
#    #   Improvement: Pick multiple background labels from around the patient
#    #   More resistant to "trays" on which the patient lays cutting the air
#    #   around the person in half
#    background_label = np.median([labels[0,0,0], labels[-1,-1,-1], labels[0,-1,-1],
#        labels[0,0,-1], labels[0,-1,0], labels[-1,0,0], labels[-1,0,-1]])
#
#    #Fill the air around the person
#    binary_image[background_label == labels] = 2
#
#    # Method of filling the lung structures (that is superior to something like
#    # morphological closing)
#    if fill_lung_structures:
#        # For every slice we determine the largest solid structure
#        for i, axial_slice in enumerate(binary_image):
#            axial_slice = axial_slice - 1
#            labeling = measure.label(axial_slice)
#            l_max = largest_label_volume(labeling, bg=0)
#
#            if l_max is not None: #This slice contains some lung
#                binary_image[i][labeling != l_max] = 1
#
#    binary_image -= 1 #Make the image actual binary
#    binary_image = 1-binary_image # Invert it, lungs are now 1
#
#    # Remove other air pockets insided body
#    labels = measure.label(binary_image, background=0)
#    m = labels.shape[0]//2
#    l_max = largest_label_volume(labels[m-12:m+20:4,:,:], bg=0)
#    if l_max is not None: # There are air pockets
#        binary_image[labels != l_max] = 0
#
#    if speedup<=1:
#        return binary_image
#    else:
#        res = np.zeros(image.shape,dtype=np.uint8)
#        for i,x in enumerate(binary_image):
#            y = transform.resize(x*1.0,image.shape[1:3])
#            res[i][y>0.5]=1
#            #res[i] = binary_dilation(res[i],disk(4))
#            #res[i] = binary_erosion(res[i],disk(4))
#        return res

def segment_lung_mask(image, speedup=4):
	'''Given {image} in HU, generate a mask, in which the value
	- 0 is background
	- 1 is lung
	- 2 is ...
	'''
	def largest_label_volume(label, bg=-1):
		'''Find the largest region in {label} that is not equal to {bg}'''
		vals, counts = np.unique(label, return_counts=True)

		counts = counts[vals != bg]
		vals = vals[vals != bg]

		if counts.size:
			return vals[np.argmax(counts)]
		return None
	if speedup > 1:
		smallImage = transform.downscale_local_mean(image, (1, speedup, speedup))
	else:
		smallImage = image
	# not actually binary, but 1 and 2.
	# 0 is treated as background, which we do not want
	# ~air-ish...lung-ish -> 1, else -> 0
	binary_image = np.array((smallImage < -320) & (smallImage>-1400), dtype=np.int8)
	#return binary_image
	for i, axial_slice in enumerate(binary_image):
		axial_slice = 1-axial_slice # lung-ish
		labeling = measure.label(axial_slice, background=0) # labels connected region
		l_max = largest_label_volume(labeling, bg=0)
		if l_max is not None: #This slice contains some lung
			binary_image[i][(labeling!=l_max)] = 1

	# Remove other air pockets insided body
	labels = measure.label(binary_image, background=0)
	m = labels.shape[0]//2
	check_layers = labels[m-12:m+20:4,:,:]
	l_max = largest_label_volume(check_layers, bg=0)

	while l_max is not None: # There are air pockets
		idx = np.where(check_layers==l_max)
		ii = np.vstack(idx[1:]).flatten()
		if np.max(ii)>labels.shape[1]-24/speedup or np.min(ii)<24/speedup:
			binary_image[labels==l_max] = 0
			labels = measure.label(binary_image, background=0)
			m = labels.shape[0]//2
			check_layers = labels[m-12:m+20:4,:,:]
			l_max = largest_label_volume(check_layers, bg=0)
		else:
			binary_image[labels != l_max] = 0
			break

	if speedup <= 1:
		return binary_image

	res = np.zeros(image.shape,dtype=np.uint8)
	for i, x in enumerate(binary_image):
		orig = x.copy()
		x = binary_dilation(x, disk(5))
		x = binary_erosion(x, disk(5))
		x |= orig.astype(bool)
		y = transform.resize(x*1.0,image.shape[1:3])
		res[i][y>0.5]=1

	return res

def np_save(f, npy):
	'''wrapper for saving {npy} array to {f}'''
	print("saving {}".format(f))
	np.save(f, npy)

def get_lung_mask_npy(case, lmask_path, save_npy=True, lazy=False):
	'''Generate or load lung mask for {case}, if {save_npy}, save generated mask to disk'''
	if lazy and os.path.isfile(lmask_path):
		#print("loaded {}".format(lmask_path))
		lung_mask = np.load(lmask_path)
	else: # otherwise, generate
		assert not save_npy
		lung_mask = segment_lung_mask(case.image,speedup=2) # SLOW
		if save_npy:
			np.save(lmask_path, lung_mask)
	return lung_mask

def get_img_mask_npy(case, lung_mask, z, img_path, mask_path, save_npy=True, lazy=False):
	'''Extract CT slice at index {z} from {case} with {lung_mask}. If {save_npy}, save generate image and mask to disk.'''
	z_is_in_nodule = case.isInNodule(z, min_pixel=3) # min_pixel matches rho_min argument below
	if lazy and os.path.isfile(img_path) and (os.path.isfile(mask_path) or not z_is_in_nodule):
		img = np.load(img_path)
		#print("loaded {}".format(img_path))
		if os.path.isfile(mask_path):
			mask = np.load(mask_path)
		else:
			assert not z_is_in_nodule
			mask = None
	else:
		#assert not save_npy
		img, mask = get_img_mask(case.image, case.h, case.nodules, nth=-1, z=z, rho_min=3) # SLOW!
		img = (img*(1 if lung_mask is None else lung_mask[z])).astype(np.uint8)
		mask = mask.astype(np.uint8)
		if save_npy:
			np_save(img_path, img)
			if mask_path and np.any(mask):
				np_save(mask_path, mask)
			elif not mask_path: # warning
				print("not saving mask for {}/{} due to null mask_path value.".format(case.hashid, z))
			#else:
			#    print("not saving mask for {}/{} due to null mask array value.".format(case.hashid, z))
	return img, mask


class LunaCase(object):
	'''Represents a LUNA case, keeps tracks of subset, suid, image file and nodules list of patient'''
	@staticmethod
	def readNodulesAnnotation(csv_dir):
		'''read LUNA anootation csv'''
		return read_csv(os.path.join(csv_dir, 'annotations.csv'))

	@classmethod
	def iterLunaCases(cls, data_dir, subset, df_node, use_tqdm=True):
		'''Iterator for all cases in {subset}, with annotation read in {df_node}'''
		subset_path = os.path.join(data_dir, 'subset{:d}'.format(subset))
		file_list = sorted(glob.glob(os.path.join(subset_path, '*.mhd')))
		for img_file in tqdm(file_list) if use_tqdm else file_list:
			# <dir>/<suid>.mhd, <suid>=<orgid>.<hash>
			# <dir>/<suid>.mhd, <suid>=<orgid>.<hash>
			suid = os.path.basename(img_file)[:-4]
			sid_node = df_node[df_node["seriesuid"]==suid] # get all nodules associate with file
			yield cls(subset, suid, img_file, sid_node)

	def __init__(self, subset, suid, img_file, sid_node):
		self.subset   = subset
		self.suid     = suid
		self.img_file = img_file
		self.sid_node = sid_node

		# populate by calling self.readImageFile
		self.image   = None
		self.origin  = None
		self.spacing = None
		self.h = None
		self.nodules = None

		self.readImageFile()

	@property
	def hashid(self):
		'''The random part of series uid'''
		return self.suid.split('.')[-1]

	def readImageFile(self):
		'''read in image file and populate relevant attributes'''
		# load iamges
		itkimage = sitk.ReadImage(self.img_file)
		self.image   = sitk.GetArrayFromImage(itkimage) # axis, sagittal, coronal
		self.origin  = np.array(itkimage.GetOrigin()) #x,y,z
		self.spacing = np.array(itkimage.GetSpacing())
		self.h = self.spacing[2] / self.spacing[0] # ratio between slice spacing and pixel size
		#load nodules infomation
		self.nodules = []
		for i in range(self.sid_node.shape[0]):
			xyz_world = np.array([self.sid_node.coordX.values[i],self.sid_node.coordY.values[i],self.sid_node.coordZ.values[i]])
			xyz_voxel = itkimage.TransformPhysicalPointToIndex(xyz_world),
			#xyz_voxel = worldToVoxelCoord(xyz_world, self.origin, self.spacing),
			d_world   = self.sid_node.diameter_mm.values[i]
			d_voxel   = d_world/self.spacing[0]
			xyzd      = tuple(np.append(xyz_voxel, d_voxel))
			self.nodules.append(xyzd)

	def isInNodule(self, z, min_pixel=0):
		'''True if slice {z} is in {nod}ule'''
		for nod in self.nodules:
			_x, _y, nz, nd = nod
			rho2 = (nd/2)**2 - ((z-nz)*self.h)**2
			if rho2>0 and np.sqrt(rho2) >= min_pixel:
				return True
		return False

def h5group_append(group, ds, name=None):
	'''Add dataset to group by name'''
	name = name or '{:d}'.format(len(group))
	if isinstance(ds, h5py.Dataset):
		group[name] = ds
	else:
		ds = group.create_dataset(name, data=ds)
	return ds


class LunaImageMaskApp(PipelineApp):
	'''Console app for generating slice image with lung mask and label mask'''
	def __init__(self):
		super(LunaImageMaskApp, self).__init__()
		self.h5file = None

	def arg_parser(self):
		parser = super(LunaImageMaskApp, self).arg_parser()
		parser = argparse.ArgumentParser(add_help=True,
				description='Generate slice, label mask and lung mask npy files',
				parents=[parser], conflict_handler='resolve')

		parser.add_argument('--no-lung-mask', action='store_true',
				help='If True, do not use lung segmentation.')

		parser.add_argument('--lazy', action='store_true',
				help='If True, use slice image, nodule mask and lung mask npy files saved in the output directory by previous session.')

		return parser

	def argparse_postparse(self, parsedArgs=None):
		super(LunaImageMaskApp, self).argparse_postparse(parsedArgs)
		self.input_dir = self.dirs.data_dir
		if parsedArgs.result_dir: # specified --result-dir
			pass
		elif parsedArgs.session:  # specified --session
			self._reslt_dir = os.path.join(self._reslt_dir, 'img_mask')
		else:
			self._reslt_dir = os.path.join(self.root, 'img_mask') # not saving to the usual result directory since these can be reused.
		print("Generate output in: {}".format(self.result_dir))

		if self.parsedArgs.hdf5:
			h5file = os.path.join(self.result_dir, 'luna06.h5')
			print("Will be saving to HDF5 file {}".format(h5file))
			#assert False #DEBUG
			self.h5file = h5py.File(h5file, 'w')

	def subset_result_dir(self, subset):
		'''Auto-provisioned output directory for processing {subset}'''
		ret = os.path.join(self.result_dir, 'subset{:d}'.format(subset))
		self.provision_dirs(ret)
		return ret

	def lung_mask_path(self, subset, hashid):
		'''npy file path to save lung mask'''
		return os.path.join(self.subset_result_dir(subset), 'lungmask_{}.npy'.format(hashid))

	def ct_image_path(self, subset, hashid, z):
		'''npy file path to save numpy array for slice {z} of patient {hashid} from {subset}'''
		return os.path.join(self.subset_result_dir(subset), 'image_{}_{:03d}.npy'.format(hashid, z))

	def nod_mask_path(self, subset, hashid, z):
		'''npy file path to save numpy array for label mask'''
		return os.path.join(self.subset_result_dir(subset), 'mask_{}_{:03d}.npy'.format(hashid, z))

	def processLunaSubset(self, subset, df_node):
		'''Process LUNA2016 data {subset}, using annotation CSV file {df_node}'''

		for case in LunaCase.iterLunaCases(self.dirs.data_dir, subset, df_node):
			# case.readImageFile()
			Nz, Nx, Ny = case.image.shape
			assert (Nx, Ny) == (512, 512)
			assert case.spacing[0]==case.spacing[1], 'CT data not evenly space'

			#save images (to save disk, only save every other image/mask pair, and the nodule location slices)
			nodule_z = {int(x[2]) for x in case.nodules} # nodule slices
			minPixels = 0.02*Nx*Ny

			# Save lung mask (used later in NoduleDetect.py
			if self.parsedArgs.no_lung_mask:
				lung_mask = None
			else:
				lmask_path = self.lung_mask_path(case.subset, case.hashid)
				lung_mask = get_lung_mask_npy(case, lmask_path,
						save_npy=self.h5file is None,
						lazy=self.parsedArgs.lazy)
				if self.h5file:
					self.h5file.require_group('lung_masks').create_dataset(case.hashid, data=lung_mask)

			#
			case_has_label = False
			for z in range(Nz):
				#z_is_in_nodule = any(LunaCase.isInNodule(z, nod) for nod in case.nodules)
				is_nod_center = z in nodule_z # nodule is centered at z
				#if not (is_nod_center or z%2):
				#    continue # skip even slices without nodules to save space.
				if lung_mask is not None:
					mask_sum = np.sum(lung_mask[z])
					if mask_sum < minPixels and not is_nod_center:
						continue

				img_path = self.ct_image_path(case.subset, case.hashid, z)
				msk_path = self.nod_mask_path(case.subset, case.hashid, z)

				img, mask = get_img_mask_npy(case, lung_mask, z,
						img_path  = img_path,
						mask_path = msk_path,
						save_npy  = self.h5file is None,
						lazy      = self.parsedArgs.lazy)
				if self.h5file:
					haslabel = np.any(mask)
					case_has_label = case_has_label or haslabel or False
					self.save_case_slice_to_h5file(case, z, img, haslabel=haslabel, mask=mask)
			if self.h5file:
				suid_path = '/by-suid/{}'.format(case.suid)
				self.h5file[suid_path].attrs.update({ 'subset' : case.subset, 'Nz' : Nz, 'has_label' : case_has_label })

	def save_case_slice_to_h5file(self, case, z, img, haslabel, mask=None):
		'''Store images and masks in HDF5 file with multiple names corresponding to the many ways we might want to address them'''
		suid_path = '/by-suid/{}'.format(case.suid)
		suid_grp = self.h5file.require_group(suid_path)
		default_path = os.path.join(suid_path, 'images-z', '{}'.format(z))
		attrs = { 'subset' : case.subset, 'suid' : case.suid, 'z' : z, 'haslabel' : bool(haslabel) }
		subset_grp = self.h5file.require_group('by-subset').require_group(str(case.subset))

		if default_path in self.h5file:
			print('seen z={} slice of {}'.format(z, case.suid)) # This happens when there is more than one annotation/label on the same slice.
			img_ds = self.h5file[default_path]
		else:
			# Default save dataset to /by-suid/{suid}/images-z/{z}
			img_ds = self.h5file.create_dataset(default_path, data=img) # NOTE: create_dataset if path already exist, that is, when the {suid}/{z} slice is already saved.
			img_ds.attrs.update(attrs)
			assert suid_grp.attrs.setdefault('subset', case.subset) == case.subset, 'suid {!r} in two different subsets: {!r} != {!r}'.format(case.suid, case.subset, suid_grp.attrs['subset'])
			# Other references: Sequentially by suid
			img_grp = suid_grp.require_group('images')
			h5group_append(img_grp, img_ds) # /by-suid/{suid}/images/{index}
			# Sequential by subset
			subset_grp = self.h5file.require_group('by-subset').require_group(str(case.subset))
			img_grp = subset_grp.require_group('images')
			h5group_append(img_grp, img_ds) # /by-subset/{subset}/images/{index}
			# Sequential by label
			img_grp = self.h5file.require_group('by-label').require_group('labeled' if haslabel else 'unlabeled')
			h5group_append(img_grp, img_ds) # /by-label/[labeled|unlabeled]/{index}
			# Sequential by all
			img_grp = self.h5file.require_group('images')
			h5group_append(img_grp, img_ds) # /images/{index}
		# process labels
		if haslabel:
			mask_grp = suid_grp.require_group('masks')
			msk_ds = mask_grp.create_dataset('{:d}'.format(len(mask_grp)), data=mask) # /by-suid/{suid}/masks/{index}
			# update attrs and cross reference image and mask
			msk_ds.attrs.update(attrs)
			img_ds.attrs['mask_name'] = msk_ds.name
			msk_ds.attrs['image_name'] = img_ds.name
			# Sequential by subset
			mask_grp = subset_grp.require_group('masks')
			h5group_append(mask_grp, msk_ds) # /by-subset/{subset}/masks/{index}
			# Sequential by all
			mask_grp = self.h5file.require_group('masks')
			h5group_append(mask_grp, msk_ds) # /masks/{index}

	def _main_impl(self):
		'''conole script entry point'''
		df_node = LunaCase.readNodulesAnnotation(self.dirs.csv_dir)
		for subset in range(10):
			if self.parsedArgs.subset and subset not in self.parsedArgs.subset:
				continue
			print("processing subset ",subset)
			self.processLunaSubset(subset, df_node)
		# finally add labels list
		if self.h5file:
			self.h5file['labels'] = [ds.attrs['haslabel'] for ds in self.h5file['images'].values()]

if __name__ == '__main__':
	sys.exit(LunaImageMaskApp().main() or 0)

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
