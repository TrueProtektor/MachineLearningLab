from transformers import pipeline, ImageClassificationPipeline
from operator import itemgetter
from PIL import Image
import PIL
import streamlit as st

classify: ImageClassificationPipeline = pipeline('object-detection', model='facebook/detr-resnet-50', feature_extractor='facebook/detr-resnet-50')

uploaded_image = st.file_uploader(label='Загрузить изображение', type=['png', 'jpg'])
if (uploaded_image is not None):
	image = Image.open(uploaded_image)
	st.caption('Исходное изображение')
	st.image(image)

	results = classify(image)

	object_detected = False
	for result in results:
		score, label, box = itemgetter('score', 'label', 'box')(result)
		img_draw = PIL.ImageDraw.Draw(image)
		if score > 0.9:
			object_detected = True
			img_draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline='red')
			
			img_draw.text([box['xmin'], box['ymin']], label)
			pass

	if (object_detected):
		st.image(image)
	else:
		st.caption('Объекты на изображении не найдены')


with(open('./README.md', encoding='UTF-8') as readme):
	st.markdown(readme.read())