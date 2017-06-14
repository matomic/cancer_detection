'''flaskapp'''
import collections
import datetime
import glob
import os
import tempfile
import zipfile
import simplejson as json

import dicom
from matplotlib.image import imsave
import numpy as np

from flask import Flask, abort, make_response, render_template, url_for, request, jsonify, redirect

from unet.predict import Case, DicomCase, CancerDetection

# DEBUGGING
from pprint import pprint

## Initialize Flask app
app = Flask(__name__)
app.permanent_session_life_time = datetime.timedelta(days=365)
root_dir = os.path.dirname(__file__)
asset_dir = os.path.join(root_dir, 'assets')

def get_context(**kwargs):
	'''return rendering context'''
	ctx = {**kwargs}
	return ctx

@app.route('/', methods=['GET'])
def index():
	'''renders index.html'''
	ctx = get_context()

	ctx.setdefault('dicompng_url', [])
	for mhd_file in glob.glob(os.path.join(asset_dir, 'mhd', '*.zip')):
		suid = os.path.basename(os.path.splitext(mhd_file)[0])
		ctx.setdefault('suid2urls_dic', collections.OrderedDict())[suid] = (url_for('.get_ct_tile_image', ct_type='mhdraw', suid=suid), 'mhdraw')
	for dcz_file in glob.glob(os.path.join(asset_dir, 'dicom', '*.zip')):
		suid = os.path.basename(os.path.splitext(dcz_file)[0])
		ctx.setdefault('suid2urls_dic', collections.OrderedDict())[suid] = (url_for('.get_ct_tile_image', ct_type='dicom', suid=suid), 'dicom')
	suid = request.args.get('suid')
	if suid in ctx['suid2urls_dic']:
		ctx['selected_suid'] = suid
	else:
		ctx['selected_suid'] = None
	pprint(ctx)
	return render_template('index.html', **ctx)

#def get_dicom(suid):
#	asset_dir = os.path.join(os.path.dirname(__file__), 'asset', suid)
#	assert os.path.isdir(asset_dir), suid
#	for f in glob.glob(os.path.join(asset_dir, '*')):
#		try:
#			return dicom.read_file(f)
#		except Exception:
#			continue

@app.route('/asset/<subdir>/<filename>', methods=['GET'])
@app.route('/asset/<filename>', methods=['GET'])
def get_asset(filename, subdir=''):
	'''get static asset file'''
	if subdir:
		path = os.path.join(root_dir, 'assets', subdir, filename)
	else:
		path = os.path.join(root_dir, 'assets', filename)
	if not os.path.isfile(path):
		abort(404)
	resp = make_response(open(path, 'rb').read())
	resp.mimetype = 'image/png'
	return resp

@app.route('/<ct_type>tile/<suid>.png', methods=['GET'])
def get_ct_tile_image(ct_type, suid, rows=100):
	'''Response with tiled PNG for DICOM scan of a particular series UID'''
	if ct_type == 'dicom':
		case_cls = DicomCase
		asset_tag = 'dicom'
	elif ct_type == 'mhdraw':
		case_cls = Case
		asset_tag = 'mhd'
	else:
		abort(404)

	zip_path = os.path.join(asset_dir, asset_tag, '{}.zip'.format(suid))
	if not os.path.isfile(zip_path):
		abort(404)
	tile_dir = os.path.join(asset_dir, 'tilepng')
	png_path = os.path.join(tile_dir, '{}.png'.format(suid))
	jsn_path = os.path.join(tile_dir, '{}.json'.format(suid)) # metadata

	reload_data = request.args.get('reload', False)
	if request.args.get('json', False) and not os.path.isfile(jsn_path):
		reload_data = True

	if reload_data or not os.path.isfile(png_path):
		case = case_cls.fromZip(zip_path)
		tile_npy = generate_tiles_png_from(case.image, rows)
		imsave(png_path, tile_npy, vmin=-1000, vmax=200, cmap='gray')
		json.dump({
			'shape': case.image.shape,
			'rows' : rows,
			'suid' : case.suid,
			},
			open(jsn_path, 'w'))

	if request.args.get('json', False):
		return jsonify(json.load(open(jsn_path,'r')))

	resp = make_response(open(png_path, 'rb').read())
	resp.mimetype = 'image/png'
	return resp

#@app.route('/mhdtile/<suid>.png', methods=['GET'])
#def get_mhd_ct_tile_image(suid, rows=100):
#	'''responds with png for mhd/raw CT scan'''
#	tile_dir = os.path.join(asset_dir, 'tilepng')
#	mhd_path = os.path.join(asset_dir, 'mhd', '{}.zip'.format(suid))
#	png_path = os.path.join(tile_dir, '{}.png'.format(suid))
#	if not os.path.isfile(mhd_path):
#		abort(404)
#	if not os.path.exists(tile_dir):
#		os.makedirs(tile_dir)
#	if not os.path.isfile(png_path) or request.args.get('reload',False):
#		case = Case.fromZip(mhd_path)
#		tile_npy = generate_tiles_png_from(case.image, rows)
#		imsave(png_path, tile_npy, vmin=-1000, vmax=200, cmap='gray')
#	resp = make_response(open(png_path, 'rb').read())
#	resp.mimetype = 'image/png'
#	return resp

def generate_tiles_png_from(ct3d_npy, rows=100):
	'''
	Take a [LxWxH] 3D numpy array {ct3d_npy}, generate a tile PNG such that each {rows} [WxH] slices form a column.
	'''
	length, width, height = ct3d_npy.shape
	cols = int(np.ceil(length/rows))
	buf = np.zeros((rows*height, width*cols), dtype=ct3d_npy.dtype)
	for idx, img in enumerate(ct3d_npy):
		offsetx = (idx % rows) * height
		offsety = (idx // rows) * width
		buf[offsetx:(offsetx+width), offsety:(offsety+height)] = img
	return buf

#def list_mhd_assets():
#	mhd_dir = os.path.join(root_dir, 'assets', 'mhd')
#	root2files_list = {}
#	for root, _d, files in os.walk(mhd_dir):
#		mhd_files = [x for x in files if x.endswith('.mhd')]
#		if mhd_files:
#			root2files_list[os.path.relpath(root, mhd_dir)] = mhd_files
#	return list_mhd_assets

@app.route('/predict/<ct_type>-<suid>.json', methods=['GET'])
def get_cancer_prediction_json(suid, ct_type):
	'''Responds with result of CancerDetection'''
	if ct_type == 'dicom':
		dcm_path = os.path.join(root_dir, 'assets', 'dicom', '{}.zip'.format(suid))
		case = DicomCase.fromZip(dcm_path)
	elif ct_type == 'mhdraw':
		mhd_path = os.path.join(root_dir, 'assets', 'mhd', '{}.zip'.format(suid))
		case = Case.fromZip(mhd_path)
	else:
		abort(500)
	session_dir = os.path.join(os.environ['LUNA_DIR'], 'results', '2017-05-17-no-lung-segmentation')
	candidate_list, n3d_out = CancerDetection.do_prediction(case, session_dir)
	prediction = [
			[(candidate.x, candidate.y, candidate.z) , probability.tolist()]
			for candidate, probability in zip(candidate_list, n3d_out)
			if probability[0] > 0.25 ]
#	prediction = [[[101,102,103], [0.4]]]
	prediction = sorted(prediction, key=lambda p : p[0][-1]) # sort by z-index
	return jsonify(prediction)

def create_mhd_zip(mhd_path):
	'''create zip archive of {mhd_path} and its RAW partner'''
	suid = os.path.splitext(mhd_path)[0]
	raw_path = suid + '.raw'
	if os.path.isfile(raw_path):
		suid = os.path.basename(suid)
		out_pth = os.path.join(asset_dir, 'mhd', '{}.zip'.format(suid))
		out_zip = zipfile.ZipFile(out_pth, 'w')
		out_zip.write(mhd_path)
		out_zip.write(raw_path)
		out_zip.close()
		print("File created: {}".format(out_pth))
		return suid

def create_dicom_zip(dcm_dir):
	'''Create zip archive of DICOM files in {dcm_dir}'''
	suid = None
	dcm_list = []
	for f in glob.glob(os.path.join(dcm_dir, '*.dcm')):
		if suid is None:
			dcm = dicom.read_file(f)
			suid = dcm.SeriesInstanceUID
		dcm_list.append(f)
	if not dcm_list:
		print("No DICOM file found in {}".format(dcm_dir))
		return
	out_pth = os.path.join(asset_dir, 'dicom', '{}.zip'.format(suid))
	out_zip = zipfile.ZipFile(out_pth, 'w')
	for f in dcm_list:
		out_zip.write(f)
	out_zip.close()
	print("File created: {}".format(out_pth))
	return suid

@app.route('/upload_file', methods=['POST'])
def upload_file():
	'''Upload zip files of MHD/RAWs or DICOMs'''
	suid_list = []
	with tempfile.TemporaryDirectory() as tmp_dir, \
		 tempfile.NamedTemporaryFile(suffix='.zip') as tmp_zip:
		request.files['upload_file'].save(tmp_zip.name) # save uploaded zipfile
		zf = zipfile.ZipFile(tmp_zip.name, 'r')
		zf.extractall(tmp_dir)
		for r, _, fs in os.walk(tmp_dir):
			has_dicom = False
			for f in fs:
				if f.endswith('.mhd'):
					suid = create_mhd_zip(os.path.join(r,f))
					if suid is not None:
						suid_list.append(suid)
				elif f.endswith('.dcm'): # DICOM files
					has_dicom = True
			if has_dicom:
				suid = create_dicom_zip(r)
				if suid is not None:
					suid_list.append(suid)
	return redirect(url_for('.index', suid=suid_list[0]))

# eof vim: set noet sw=4 ts=4:
