#!/usr/bin/python3
'''Script to Preprocess LUNA2016 images'''
from __future__ import print_function, division

# stdlib
import os
import collections
from glob import glob

# 3rd-party
#import SimpleITK as sitk
#import csv
#from matplotlib import pyplot as plt
from skimage import draw, measure, transform
#from skimage.feature import canny
from skimage.morphology import binary_dilation, binary_erosion, disk#, square
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import pandas as pd
import h5py

# in-house
import config as cfg

# debugging
from pprint import pprint
from ipdb import set_trace


def load_itk_image(filename):
    '''load mhd/raw {filename}'''
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage) # axis, sagittal, coronal

    numpyOrigin = np.array(itkimage.GetOrigin()) #x,y,z
    numpySpacing = np.array(itkimage.GetSpacing())

    return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
    """
    only valid if there is no rotation component
    """
    voxelCoord = np.rint((worldCoord-origin)/ spacing).astype(np.int)
    return voxelCoord

def normalize(x):
    '''Normalize scan, mapping -1000 HU to 0 in gs and 200 HU to 255 gs'''
    y = np.interp(x, [-1000,200], [0, 255], left=0, right=255).astype(np.uint8)#0-255, to save disk space
    return y

def get_img_mask(scan, h, nodules, nth=-1, z=None):
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
    img = normalize(scan[z,:,:])
    res = np.zeros(img.shape)
    #draw nodules
    for n_x, n_y, n_z, n_d in nodules:
        r = n_d /2.0
        dz = np.abs((n_z-z)*h)
        if dz >= r:
            continue
        rlayer = np.sqrt(r**2-dz**2)
        if rlayer < 3:
            continue
        #create contour at xyzd[0],xyzd[1] with radius rlayer
        rr, cc = draw.circle(n_y, n_x, rlayer)
        res[rr, cc] = 1
    return img, res

def segment_lung_mask2(image, fill_lung_structures=True, speedup=4):
    '''missing-docstring'''
    def largest_label_volume(im, bg=-1):
        '''missing-docstring'''
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    if speedup>1:
        smallImage = transform.downscale_local_mean(image,(1,speedup,speedup))
    else:
        smallImage = image
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(smallImage > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = np.median([labels[0,0,0], labels[-1,-1,-1], labels[0,-1,-1],
        labels[0,0,-1], labels[0,-1,0], labels[-1,0,0], labels[-1,0,-1]])

    #Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    m = labels.shape[0]//2
    l_max = largest_label_volume(labels[m-12:m+20:4,:,:], bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    if speedup<=1:
        return binary_image
    else:
        res = np.zeros(image.shape,dtype=np.uint8)
        for i,x in enumerate(binary_image):
            y = transform.resize(x*1.0,image.shape[1:3])
            res[i][y>0.5]=1
            #res[i] = binary_dilation(res[i],disk(4))
            #res[i] = binary_erosion(res[i],disk(4))
        return res

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

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
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

    if speedup<=1:
        return binary_image
    else:
        res = np.zeros(image.shape,dtype=np.uint8)
        for i, x in enumerate(binary_image):
            orig = x.copy()
            x = binary_dilation(x, disk(5))
            x = binary_erosion(x, disk(5))
            x |= orig.astype(bool)
            y = transform.resize(x*1.0,image.shape[1:3])
            res[i][y>0.5]=1

        return res

def iterSubsetImagesAndMasks(subset, subset_path, df_node):
    '''Preprocess LUNA2016 data {subset} at {subset_path}, using annotations CSV file {df_node}
    Generate single HDF5 file.
    '''
    print("processing subset ",subset)
    file_list = sorted(glob(os.path.join(subset_path,"*.mhd")))

    # Looping over the image files in the subset
    for img_file in tqdm(file_list):
        # <dir>/<suid>.mhd, <suid>=<orgid>.<hash>
        # <dir>/<suid>.mhd, <suid>=<orgid>.<hash>
        suid = os.path.basename(img_file)[:-4]
        hashid = suid.split('.')[-1]
        sid_node = df_node[df_node["seriesuid"]==suid] #get all nodules associate with file
        #load images
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_file)
        Nz, Nx, Ny = numpyImage.shape
        assert (Nx, Ny) == (512, 512)
        assert numpySpacing[0]==numpySpacing[1], 'CT data not evenly space'

        #load nodules infomation
        nodules = []
        for i in range(sid_node.shape[0]):
            xyz_world = np.array([sid_node.coordX.values[i],sid_node.coordY.values[i],sid_node.coordZ.values[i]])
            xyz       = worldToVoxelCoord(xyz_world, numpyOrigin, numpySpacing)
            d_world   = sid_node.diameter_mm.values[i]
            diameter  = d_world/numpySpacing[0]
            xyzd      = tuple(np.append(xyz, diameter))
            nodules.append(xyzd)
        h = numpySpacing[2]/numpySpacing[0]

        #Lung mask
        lungMask = segment_lung_mask(numpyImage,speedup=2)

        #save images (to save disk, only save every other image/mask pair, and the nodule location slices)
        Nz, Nx, Ny = numpyImage.shape
        zs = list(range(1, Nz, 2)) #odd slices
        zs = sorted(zs + [int(x[2]) for x in nodules if x[2]%2==0])
        minPixels = 0.02*Nx*Ny
        for z in zs:
            if np.sum(lungMask[z])<minPixels:
                continue
            img, mask = get_img_mask(numpyImage, h, nodules, nth=-1,z=z)
            img = (img*lungMask[z]).astype(np.uint8)
            mask = mask.astype(np.uint8)

            yield (suid, hashid), z, img, mask

def h5group_append(group, ds, name=None):
    '''Add dataset to group by name'''
    name = name or '{:d}'.format(len(group))
    if isinstance(ds, h5py.Dataset):
        group[name] = ds
    else:
        ds = group.create_dataset(name, data=ds)
    return ds

def processLunaSubset(subset, subset_path, df_node, output_path, h5file=None):
    '''Process LUNA2016 data {subset} at {luna_subset_path}, using annotation CSV file {df_node}'''
    suid2z_set = collections.defaultdict(set)
    for (suid, hashid), z, img, mask in iterSubsetImagesAndMasks(subset, subset_path, df_node):
        img_path = os.path.join(output_path, "image_{}_{:03d}.npy".format(hashid, z))
        msk_path = os.path.join(output_path, "mask_{}_{:03d}.npy".format(hashid, z))
        haslabel = np.any(mask)
        if h5file is not None:
            # Store images and masks in HDF5 file with multiple names corresponding to the many ways we might want to address them
            if z in suid2z_set[suid]:
                print('seen z={} slice of {}'.format(z, suid))
            else:
                # Sequential by suid
                attrs = { 'subset' : subset, 'suid' : suid, 'z' : z, 'haslabel' : haslabel }
                suid_grp = h5file.require_group('by-suid').require_group(suid)
                assert suid_grp.attrs.setdefault('subset', subset) == subset, 'suid {!r} in two different subsets: {!r} != {!r}'.format(suid, subset, suid_grp.attrs['subset'])
                suid2z_set[suid].add(z) # seen slice before
                img_grp = suid_grp.require_group('images')
                img_ds = img_grp.create_dataset('{:d}'.format(len(img_grp)), data=img) # /by-suid/{suid}/images/{index}
                img_ds.attrs.update(attrs)
                img_ds.attrs['path'] = img_path
                # Sequential by subset
                subset_grp = h5file.require_group('by-subset').require_group(str(subset))
                img_grp = subset_grp.require_group('images')
                h5group_append(img_grp, img_ds) # /by-subset/{subset}/images/{index}
                # Sequential by label
                img_grp = h5file.require_group('by-label').require_group('labeled' if haslabel else 'unlabeled')
                h5group_append(img_grp, img_ds) # /by-label/[labeled|unlabeled]/{index}
                # Sequential by all
                img_grp = h5file.require_group('images')
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
                mask_grp = h5file.require_group('masks')
                h5group_append(mask_grp, msk_ds) # /masks/{index}
        else:
            # Save to individual .npy files
            np.save(img_path, img)
            if haslabel:
                np.save(msk_path, mask)
    #set_trace()

def main():
    '''conole script entry point'''
    df_node = pd.read_csv(os.path.join(cfg.csv_dir, "annotations.csv"))
    h5file = os.path.join(cfg.root, 'img_mask', 'luna06.h5')
    h5file = h5py.File(h5file, 'w')
    for subset in range(10):
        luna_subset_path = os.path.join(cfg.root, "data", "subset{}".format(subset))
        output_path = os.path.join(cfg.root,'img_mask', 'subset{}'.format(subset))
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        processLunaSubset(subset, luna_subset_path, df_node, output_path, h5file=h5file)
    # finally add labels list
    h5file['labels'] = [ds.attrs['haslabel'] for ds in h5file['images'].values()]
if __name__ == '__main__':
    main()

# eof
