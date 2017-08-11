'''Evaluate the ROC of model'''
import os.path

import numpy as np

from NoduleDetect import NoduleDetectApp
from console import PipelineApp
from img_mask_gen import LunaCase
from train import load_unet, UnetTrainer
from trainNodule import load_n3dnet
from utils import safejsondump

# DEBUGGING
from pprint import pprint
try:
	from ipdb import set_trace
except:
	pass


class Evaluator(PipelineApp):
	''''''
	def _main_impl(self):
		output_shape = (self.n3d_cfg.net.WIDTH, self.n3d_cfg.net.HEIGHT, self.n3d_cfg.net.CHANNEL)
		# Unet model
		models = [load_unet(self.unet_cfg, checkpoint_path=os.path.join(self.result_dir, model['checkpoint_path']))
				for fold, model in self.session_json['unet']['models'].items() ]
		# N3D model
		n3d_model = load_n3dnet(self.n3d_cfg,
				checkpoint_path=os.path.join(self.result_dir, self.session_json['n3d']['models']['0']['checkpoint_path']))

		df_node = LunaCase.readNodulesAnnotation(self.dirs.csv_dir)

		evaluation = {}

		for subset in range(10):
			nodules = cases = nodule_cases = 0
			subset_eval ={
			    't+' : 0,
			    'f+' : 0,
			    't-' : 0,
			    'f-' : 0,
			}
			for case in LunaCase.iterLunaCases(self.dirs.data_dir, subset, df_node):
				model_in, model_out = NoduleDetectApp.predict_nodules(case, models, lung_mask=None, cut=self.unet_cfg.keep_prob)
				candidate_list, detected_set = NoduleDetectApp.detect_nodule_candidate(model_in, model_out, output_shape, lung_mask=None, training_case=case)

				case_eval = {
				    't+' : 0,
				    'f+' : 0,
				    't-' : 0,
				    'f-' : 0,
				}

				for xyzd in case.nodules:
					if xyzd not in detected_set:
						case_eval['f-'] += 1 # false negative: case nodule not detected as candidate

				found_any = False
				if candidate_list:
					n3d_in  = np.asarray([x.volume_array for x in candidate_list])
					n3d_out = n3d_model.predict(np.expand_dims(n3d_in,-1), batch_size=10)
					for candidate, probability in zip(candidate_list, n3d_out):
						if probability >= 0.5:
							found_any = True
							if candidate.is_nodule:
								case_eval['t+'] += 1 # true positive: nodule detected
							else:
								case_eval['f+'] += 1 # false positive: none-nodule detected
						else:
							if candidate.is_nodule:
								case_eval['f-'] += 1 # false negative: nodule not detected
							#else:
							#	case_eval['t-'] += 1 # true negative: none-nodule excluded
				if not case.nodules and not found_any:
					case_eval['t-'] += 1 # true negative: no nodule, no candidates

				for k in case_eval:
					subset_eval[k] += case_eval[k]
#				print("Case {}: {} nodules {}".format(case.suid, len(case.nodules), case_eval))
				node_list = df_node[df_node['seriesuid']==case.suid]

				nodules += len(node_list)
				nodule_cases += int(not node_list.empty)
				cases += 1

			evaluation[subset] = {
					'cases'        : cases,
					'nodules'      : nodules,
					'nodule_cases' : nodule_cases,
					'evaluation'   : subset_eval,
					}
			pprint(evaluation[subset])
		safejsondump(evaluation, os.path.join(self.result_dir, 'evaluation.json'))

if __name__ == '__main__':
	Evaluator().main()

# eof vim: set noet ci pi sts=0 sw=4 ts=4:
