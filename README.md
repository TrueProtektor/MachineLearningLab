# MachineLearningLab
Выбранная модель: https://huggingface.co/facebook/detr-resnet-50  
Фейсбуковская модель для нахождения объектов на изображении.  
Использовать будем для проектного практикума, тема - галлерея
для фоточек с ML фичами.  
## Упрощеная архитектура
1. Основа - сверточная сеть, извлекает признаки
в плоской форме, снабжает их информацией о местоположении
на исходном изображении;
1. Полученные признаки подаются в encoder-decoder Transformer
с 6 encoder'ами и decored'ами;
1. Выход каждого decoder'a подается в трехслойный перцептрон
который предсказывает либо "попадание" (класс, координаты),
либо класс "нет объекта".