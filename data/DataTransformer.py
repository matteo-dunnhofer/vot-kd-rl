"""
Written by Matteo Dunnhofer - 2020

Data transformer class
"""
from torchvision import transforms
import utils as ut


class DataTransformer(object):

	def __init__(self, cfg):
		self.cfg = cfg

		self.transform = transforms.Compose([
			transforms.Resize((self.cfg.SIZE[0], self.cfg.SIZE[1])),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	def preprocess_img(self, img, crop_bb):
		crop = ut.xywh2xyxy(crop_bb)
		img = img.crop(crop)
		img = self.transform(img)
		return img
