from transformers import pipeline, ImageClassificationPipeline
from operator import itemgetter
from PIL import Image
import PIL
from fastapi import FastAPI, UploadFile, Response
from io import BytesIO

classify: ImageClassificationPipeline = pipeline('object-detection', model='facebook/detr-resnet-50', feature_extractor='facebook/detr-resnet-50')

app = FastAPI()

@app.post('/')
async def root(uploaded_image: UploadFile) -> Response:
	if (uploaded_image is not None):
		image = Image.open(uploaded_image.file)

		results = classify(image)
		result_image = image.copy()
		img_draw = PIL.ImageDraw.Draw(result_image)

		object_detected = False
		for result in results:
			score, label, box = itemgetter('score', 'label', 'box')(result)
			if score > 0.9:
				object_detected = True
				img_draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline='red')
				
				img_draw.text([box['xmin'], box['ymin']], label)
				pass

		if (object_detected):
			result_image.save
			
			img_byte_arr = BytesIO()
			result_image.save(img_byte_arr, format='PNG')
			return Response(content=img_byte_arr.getvalue(), media_type='image/png')
		else:
			return Response(content='Объекты на изображении не найдены', media_type='text/plain')