from cocoapi.PythonAPI.pycocotools.mask import *
import glob
import os
import imageio
import random
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import numpy as np
from imantics import Polygons, Mask
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from tqdm import tqdm

import json
import collections as cl

import uuid

IMAGE_HEIGHT=240 * 2
IMAGE_WIDTH=320 * 2
TRAIN_PERC=0.9

current_path=os.path.dirname(os.path.realpath(__file__))
base_path = "/media/mostafa/Data/Mostafa/sens_decoder/c++/x64/Debug/out_"
labels_base_path = "/media/mostafa/Data/RealTimeSegmentation/ScanNet/scans/extracted_labels"
semantic_label_base_path = "/media/mostafa/Data/RealTimeSegmentation/ScanNet/scans/extracted_labels_"

scene_index_file = "sens_paths.txt"
id_mappings_file = "scannetv2-labels.combined.tsv"
scannet_mapping_file = "nyu40labels_scannet.csv"

f = open(scene_index_file)
lines = f.readlines()

# We are getting the scenes in the same order from the text file that was used in the sens decoder.
# This is to have the same matching scenes in the 2D labels, and in the output of the decoder,
# keeping in mind that the output folders of the decoder lacks the scene name.
scenes = [line.split("\\")[-1].split(".")[0] for line in lines]

dataset_dicts=[]
record_index=0

f = open(id_mappings_file)
lines = f.readlines()[1:] # Skip headers
lines = [line.split("\t") for line in lines]
id_to_nyu_map = {int(line[0]):int(line[4]) for line in lines}

f = open(scannet_mapping_file)
lines = f.readlines()[1:]
lines = [line.split(",") for line in lines]
nyu_to_scannet_map = {int(line[0]):int(line[3]) for line in lines}

json_img = []
json_annot = []

def tile(vector, mat_dims):
	vector = np.squeeze(vector) # To avoid double expansion

	for idx, dim in enumerate(mat_dims):
		repeater = np.ones(dim)

		repeater = np.expand_dims(repeater, 0)
		vector   = np.expand_dims(vector, idx + 1)

		vector   = vector * repeater

	return vector


def predict(img, predictor): 
	outputs = predictor(img)
	instance_label = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

	# print(outputs['instances'].get_fields().keys())

	semantic_ids = outputs['instances'].get_fields()["pred_classes"].cpu().detach().numpy()
	pred_masks = outputs['instances'].get_fields()["pred_masks"].cpu().detach().numpy()
	scores = outputs['instances'].get_fields()["scores"].cpu().detach().numpy()

	if scores.shape[0] == 0: # No detected objects
		return (instance_label, semantic_ids, scores)

	scores = np.insert(scores, 0, 0) # Shift array to the right by one, so the first object wouldn't have an id of zero

	zeros = np.zeros(pred_masks.shape[1:])
	pred_masks = np.insert(pred_masks, 0, zeros, axis=0) # Do the same shifting for the prediction masks

	# print("scores", scores.shape)

	tiled = tile(scores, pred_masks.shape[1:])


	pred_masks = pred_masks * tiled
	instance_label = np.argmax(pred_masks, 0)

	return (np.uint8(instance_label), semantic_ids, scores)

def read_scene(scene, idx):
	images = sorted(glob.glob(os.path.join(base_path, str(idx), "*.jpg")), key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[1]))
	#semantic = sorted(glob.glob(os.path.join(semantic_label_base_path, scene, "label-filt", "*.png")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
	#labels = sorted(glob.glob(os.path.join(labels_base_path, scene, "instance-filt", "*.png")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
	#path='/media/mostafa/Data/Mostafa/sens_decoder/c++/x64/Debug/out'
	#images = sorted(glob.glob(os.path.join(path, str(idx), "*.jpg")), key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[1]))
	semantic = sorted(glob.glob(os.path.join(semantic_label_base_path, scene, "label-filt", "*.png")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
	labels = sorted(glob.glob(os.path.join(labels_base_path, scene, "instance-filt", "*.png")), key=lambda x: int(x.split("/")[-1].split(".")[0]))

	return images, labels, semantic

# Maps from object id to nyu40 or scannet reduced 20 classes
def process_semantic(semantic):
	unq = np.unique(semantic)
	res = semantic.copy()

	unq =np.delete(unq, np.argwhere(unq == 0))
	for obj_id in unq:
		res[semantic == obj_id] = nyu_to_scannet_map[id_to_nyu_map[obj_id]]

	return res

def get_category(mask, semantic):
	points = list(np.argwhere(mask != 0))
	query_points = random.sample(points, min(5, len(points)))
	xs = [point[0] for point in query_points]
	ys = [point[1] for point in query_points]
	vals = semantic[xs, ys]
	counts = np.bincount(vals)
	category = np.argmax(counts)
	return category

def get_scannet_dicts(d):
	global record_index
	lower = 0 if d == "train" else int(TRAIN_PERC * float(len(scenes)))
	upper = int(TRAIN_PERC * float(len(scenes))) if d == "train" else len(scenes)

	# for idx, scene in tqdm(enumerate(scenes[lower:upper])):
	# for idx, scene in tqdm(enumerate(scenes[:2])):
	for idx, scene in tqdm(enumerate(scenes[lower:lower + 2])):
		print("Register scene "+scene+" as "+d)
		images,labels, semantic=read_scene(scene, idx)
		labels = [np.array(Image.open(label).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)) for label in labels]
		semantics = [np.array(Image.open(label).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)) for label in semantic]
		semantics = [process_semantic(semantic) for semantic in semantics]
		# for image,label, semantic in list(zip(images,labels, semantics)):
		for image,label, semantic in list(zip(images,labels, semantics))[:1]:
			record={}
			if idx == 0 and "003780" in image:
				continue
			record["file_name"]=image	
			record["image_id"]=record_index	
			record["height"]=IMAGE_HEIGHT
			record["width"]=IMAGE_WIDTH
			indices = np.unique(label)
			# Remove background index
			indices = np.delete(indices, np.argwhere(indices == 0))
			objs = []
			# for index in indices[1:]: 
			for index in indices: 
				temp = np.copy(label)
				temp = np.where(temp == index, index, 0)
				mask = Mask(temp)
				polygons = [a.astype(np.float32) + 0.5 for a in list(mask.polygons())]
				polygons = [a for a in polygons if len(a) >= 6]
				if len(polygons) != 0:
					category_id = get_category(temp, semantic)
					if category_id == 19:
						continue

					bbox = mask.bbox()

					polygons = [list(p) for p in polygons]
					polygons = [[float(q) for q in p] for p in polygons]

					category_id = int(category_id)

					obj = {
						"bbox": list(bbox),
						"bbox_mode": BoxMode.XYXY_ABS,
						"segmentation": polygons,
						"category_id": category_id # NUM CATEGORIES + 1 => VOID
					}
					objs.append(obj)
			record["annotations"] = objs

			record_index+=1
			dataset_dicts.append(record)
	return dataset_dicts
			

ImageFile.LOAD_TRUNCATED_IMAGES = True

##################################################################################################
#                                 REGISTER TRAINING AND TEST DATA                                #
##################################################################################################
from detectron2.data import DatasetCatalog
for d in ["train", "val"]:
	print("Start register "+d+" set...")
	DatasetCatalog.register("scannet_dataset_" + d, lambda d=d: get_scannet_dicts(d))
	MetadataCatalog.get("scannet_dataset_" + d).set(thing_classes=[" "]) # TODO SET LABELS FOR THING CLASSES
scannet_metadata = MetadataCatalog.get("scannet_dataset_train")
# MetadataCatalog.get("scannet_dataset_val").set(json_file="output/scannet_dataset_val_coco_format.json")

##################################################################################################
#                                             TRAINING                                           #
##################################################################################################
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

#TODO works with panoptic labels instead of instance labels or implement two models (semantic and instance)
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("scannet_dataset_train",)
cfg.DATASETS.TEST = ("scannet_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 7
# # cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
# # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/media/mostafa/Data/RealTimeSegmentation/detectron2_train_1/model_0006071.pth")
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
# cfg.MODEL.WEIGHTS = os.path.join("/media/mostafa/Data/RealTimeSegmentation/checkpoints_new", "model_0139999.pth")
# cfg.MODEL.WEIGHTS = os.path.join("/media/mostafa/Data/RealTimeSegmentation/checkpoints_new__", "model_0309999.pth")
cfg.MODEL.WEIGHTS = os.path.join("/media/mostafa/Data/RealTimeSegmentation/checkpoints_new__", "model_0369999.pth")
cfg.SOLVER.IMS_PER_BATCH = 1
#cfg.SOLVER.BASE_LR = 0.00025
#cfg.SOLVER.BASE_LR = 1e-3
cfg.SOLVER.BASE_LR = 1e-5
cfg.SOLVER.MAX_ITER = int(40e6)
# cfg.SOLVER.MAX_ITER = int(1)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 256   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18  # only has one class (ballon) WITHOUT VOID
cfg.SOLVER.CHECKPOINT_PERIOD = 10000
cfg.OUTPUT_DIR = "/media/mostafa/Data/RealTimeSegmentation/checkpoints_new__"

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# evaluator = COCOEvaluator("scannet_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
# val_loader = build_detection_test_loader(cfg, "scannet_dataset_val")
 
# cfg.TEST.EVAL_PERIOD = 1

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# evaluator = COCOEvaluator("scannet_dataset_val", cfg, False, output_dir="./output/")
#val_loader = build_detection_test_loader(cfg, "scannet_dataset_val")
# metrics = DefaultTrainer.test(cfg, trainer.model, evaluator)
# print(metrics)
# trainer.resume_or_load(resume=True)
# trainer.train()
##################################################################################################
#                                           PREDICTION                                           #
##################################################################################################
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/media/mostafa/Data/RealTimeSegmentation/checkpoints_new/model_0139999.pth")
cfg.MODEL.WEIGHTS = os.path.join("/media/mostafa/Data/RealTimeSegmentation/checkpoints_new__", "model_0369999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # get the testing threshold for this model
cfg.DATASETS.TEST = ("scannet_dataset_val", )
predictor = DefaultPredictor(cfg)

##################################################################################################
#                                       INFERENCE AND TEST                                       #
##################################################################################################
# import timeit

# from detectron2.utils.visualizer import ColorMode
# # # path = "/home/mostafa/git/detectron2/scannet_train_images/scene0000_00/color/40.jpg"
# path = "/media/mostafa/Data/Mostafa/sens_decoder/c++/x64/Debug/out/0/frame-000020.color.jpg"
# im   = cv2.imread(path)
# # # print(timeit.default_timer())

# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], 
#               scale=0.8, 
#               metadata=scannet_metadata,
#               instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
# )

# # print(outputs["instances"])
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# # pred_shape=outputs['instances'].get_fields()["pred_masks"].shape
# pred_masks=outputs['instances'].get_fields()["pred_masks"].cpu().detach().numpy()
# scores = outputs['instances'].get_fields()["scores"].cpu().detach().numpy()

# # print(pred_masks.shape) # Handle no instances
# # print(scores.shape) # Num instances

# scores = np.insert(scores, 0, 0) # Shift array to the right by one, so the first object wouldn't have an id of zero

# zeros = np.zeros(pred_masks.shape[1:])
# pred_masks = np.insert(pred_masks, 0, zeros, axis=0) # Do the same shifting for the prediction masks

# tiled = tile(scores, pred_masks.shape[1:])
# # print(tiled.shape)


# instance_label = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

# pred_masks = pred_masks * tiled

# # for row in range(pred_masks.shape[1]):
# # 	for col in range(pred_masks.shape[2]):
# # 		for idx in range(pred_masks.shape[0]): # This idx would override local ones in functions
# # 			if pred_masks[idx][row][col]:
# # 				instance_label[row][col] = idx


# # print()
# # print(pred_masks[:, 37, 189])
# instance_label = np.argmax(pred_masks, 0) 
# print(instance_label.shape)

# # cv2.imshow("sdfds", v.get_image()[:, :, ::-1])
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# plt.figure()
# plt.imshow(v.get_image()[:, :, ::-1])



# plt.figure()
# plt.imshow(instance_label)
# plt.show()

##################################################################################################
#                                       Write predictions                                        #
##################################################################################################
# USE ALL FRAMES, not every 20th

def val_img_pair(pred, gt):
	ids_pred = np.unique(pred)
	ids_gt = np.unique(gt)

	ids_pred = np.delete(ids_pred, np.argwhere(ids_pred == 0))
	ids_gt = np.delete(ids_gt, np.argwhere(ids_gt == 0))

	thresh = 0.5
	association = {} # gt -> pred
	scores = {} # gt -> iou
	recall_precision_map = {}

	tp = 0.0
	fp = 0.0
	fn = len(ids_gt)

	for _pred in ids_pred:
		found_match = False
		for _gt in ids_gt:
			mask_pred = np.where(pred == _pred, 1, 0)
			mask_gt = np.where(gt == _gt, 1, 0)

			intersection = np.count_nonzero(np.logical_and(mask_pred, mask_gt))
			union = np.count_nonzero(np.logical_or(mask_pred, mask_gt))

			iou = intersection / union if union > 0 else 0
			if iou >= thresh:
				found_match = True
				if _gt not in association:
					tp+=1
					fn-=1
					association[_gt] = _pred
					scores[_gt] = iou
				else:
					fp+=1
					if iou > scores[_gt]:
						association[_gt] = _pred
						scores[_gt] = iou

		if not found_match:
			fp+=1

		curr_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
		curr_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

		recall_precision_map[curr_recall] = curr_precision

	average_precision = 0.0

	for r in np.arange(0, 1.01, 0.01):
		curr_max_precision = -1

		for _r in recall_precision_map.keys():
			if _r >= r and recall_precision_map[_r] > curr_max_precision:
				curr_max_precision = recall_precision_map[_r]

		if curr_max_precision == -1:
			break
		average_precision += curr_max_precision

	average_precision /= 101.0

	# print(ids_pred)
	# print(ids_gt)
	# print(association)
	# print(scores)
	# print(tp, fp, fn)

	return average_precision



os.makedirs(os.path.join(cfg.OUTPUT_DIR, "predictions"), exist_ok=True)
# for idx, scene in tqdm(enumerate(scenes[:1])):
ap = 0.0
q = 0
d = "val"
lower = 0 if d == "train" else int(TRAIN_PERC * float(len(scenes)))
upper = int(TRAIN_PERC * float(len(scenes))) if d == "train" else len(scenes)
for idx, scene in enumerate(scenes[lower:lower + 20]):
	# if "scene0000_00" in scene:
		os.makedirs(os.path.join(cfg.OUTPUT_DIR, "predictions", scene), exist_ok=True)
		images,labels, semantic=read_scene(scene, idx + lower)
		print(len(images), len(labels), len(semantic))
		q += len(images)
		_tmp = 0
		for idx, image in tqdm(enumerate(images)):
			if "3780" not in image:
				_image = cv2.imread(image)
				_image = cv2.resize(_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
				instance_label, semantic_ids, scores = predict(_image, predictor)

				_semantic = np.array(Image.open(semantic[idx]).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST))
				_semantic = process_semantic(_semantic)

				label_gt = np.array(Image.open(labels[idx]).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST))

				# stuff = [0, 1, 2, 22, 13, 15, 17, 18, 19, 20, 21, 23, 25, 26, 27, 29, 30, 31, 32, 35, 37, 38, 40]

				label_gt = np.where(_semantic == 19, 0, label_gt)


				_tmp += val_img_pair(instance_label, label_gt) 
		ap += _tmp
		print(scene)
		print(_tmp / len(images))


ap /= q

print(ap)

			# path = os.path.join(cfg.OUTPUT_DIR, "predictions", scene, str(int(images[idx].split("/")[-1].split(".")[0].split("-")[1])))
			# imageio.imwrite(path + ".png", instance_label)
			# with open(path + ".txt", "w") as f:
			# 	f.write(" ".join(str(n) for n in semantic_ids))
			# 	f.write("\n")
			# 	f.write(" ".join(str(n) for n in scores))
			# 	f.write("\n")

		#_images = [cv2.imread(image) for image in images if not(idx == 0 and "003780" in image)]
	# 	_images = [cv2.imread(image) for image in images]
	# #	semantics = [np.array(Image.open(label).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)) for label in semantic]
	# #	semantics = [process_semantic(semantic) for semantic in semantics]

	# 	# instance_labels = [predict(image, predictor) for image in _images]
	# 	preds = []

	# 	for image in tqdm(_images):
	# 		preds.append(predict(image, predictor))

	# 	for idx, pred in enumerate(preds):
	# 		instance_label, semantic_ids, scores = pred

	# 		path = os.path.join(cfg.OUTPUT_DIR, "predictions", scene, str(int(images[idx].split("/")[-1].split(".")[0].split("-")[1])))
	# 		imageio.imwrite(path + ".png", instance_label)
	# 		with open(path + ".txt", "w") as f:
	# 			f.write(" ".join(str(n) for n in semantic_ids))
	# 			f.write("\n")
	# 			f.write(" ".join(str(n) for n in scores))
	# 			f.write("\n")



##################################################################################################
#                                       Validation                                               #
##################################################################################################
# for idx, scene in enumerate(scenes):
# 	# os.makedirs(os.path.join(cfg.OUTPUT_DIR, "predictions", scene), exist_ok=True)
# 	images,labels, semantic=read_scene(scene, idx)
# 	_images = [cv2.imread(image) for image in images]
# 	labels = [np.array(Image.open(label).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST)) for label in labels]
##################################################################################################
#                                       Validation on one ckpt                                   #
##################################################################################################

#THIS DOESN'T WORK PROPERLY!!!!!!!!!
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.data import build_detection_test_loader
## evaluator = COCOEvaluator("scannet_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
#evaluator = COCOEvaluator("scannet_dataset_val", cfg, False, output_dir="output")
#val_loader = build_detection_test_loader(cfg, "scannet_dataset_val")
#metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
#print(metrics)


#(detectron2) mostafa@TUINI15-MA21:~/git/detectron2$ python _register_scannet.py 
#Start register train set...
#Start register val set...
# 66%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                              | 3701/5578 [21:16<52:51,  1.69s/it]Killed





# scene0639_00
# 0.4448476544082978
# 146 146 146
# 146it [00:43,  3.35it/s]
# scene0640_00
# 0.4244306955533728
# 152 152 152
# 152it [00:45,  3.36it/s]
# scene0640_01
# 0.4341342639903086
# 143 143 143
# 143it [00:42,  3.40it/s]
# scene0640_02
# 0.4179796900768998
# 70 70 70
# 70it [00:22,  3.10it/s]
# scene0641_00
# 0.6513863376133532
# 211 211 211
# 211it [01:01,  3.45it/s]
# scene0642_00
# 0.5174178064387348
# 212 212 212
# 212it [01:01,  3.46it/s]
# scene0642_01
# 0.4586231685837048
# 98 98 98
# 98it [00:28,  3.42it/s]
# scene0642_02
# 0.5989755025065188
# 122 122 122
# 122it [00:34,  3.50it/s]
# scene0642_03
# 0.6062725593402432
# 97 97 97
# 97it [00:27,  3.55it/s]
# scene0643_00
# 0.4251763173371829
# 101 101 101
# 101it [00:29,  3.42it/s]
# scene0644_00
# 0.23833363017511613
# 226 226 226
# 226it [01:04,  3.49it/s]
# scene0645_00
# 0.39812849173665804
# 262 262 262
# 262it [01:15,  3.49it/s]
# scene0645_01
# 0.4569498580174269
# 146 146 146
# 146it [00:43,  3.38it/s]
# scene0645_02
# 0.4315242874580998
# 244 244 244
# 244it [01:08,  3.57it/s]
# scene0646_00
# 0.27870871689042453
# 246 246 246
# 246it [01:09,  3.55it/s]
# scene0646_01
# 0.24020547525135794
# 195 195 195
# 195it [00:54,  3.56it/s]
# scene0646_02
# 0.2651135772917949
# 55 55 55
# 55it [00:15,  3.54it/s]
# scene0647_00
# 0.28748574857485726
# 60 60 60
# 60it [00:17,  3.47it/s]
# scene0647_01
# 0.34357260726072597
# 210 210 210
# 210it [00:56,  3.69it/s]
# scene0648_00
# 0.3213012729844413
# 0.39872288077480206
