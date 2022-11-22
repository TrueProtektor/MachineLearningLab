from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import PIL
import requests
import numpy as np
import streamlit as st

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

uploaded_image = st.file_uploader(label='Загрузить изображение', type=['png', 'jpg'])
if (uploaded_image is not None):
	image = Image.open(uploaded_image)
	st.caption('Исходное изображение')
	st.image(image)

	inputs = feature_extractor(images=image, return_tensors="pt")
	outputs = model(**inputs)

	target_sizes = torch.tensor([image.size[::-1]])
	results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

	object_detected = False
	for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
		box = [round(i, 2) for i in box.tolist()]
		img_draw = PIL.ImageDraw.Draw(image)
		if score > 0.9:
			object_detected = True
			img_draw.rectangle([box[0], box[1], box[2], box[3]], outline='red')
			
			name = model.config.id2label[label.item()]
			img_draw.text([box[0], box[1]], name)
			pass

	if (object_detected):
		st.image(image)
	else:
		st.caption('Объекты на изображении не найдены')


with(open('./README.md', encoding='UTF-8') as readme):
	st.markdown(readme.read())