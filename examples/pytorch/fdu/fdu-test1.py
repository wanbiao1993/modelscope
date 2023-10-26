from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# model_id='damo/cv_resnet50_face-detection_retinaface'
model_id="wanbiao/resnet18"
# model_id="damo/cv_vit-base_image-classification_Dailylife-labels"
retina_face_detection = pipeline(Tasks.plant_classification, model_id)
img_path = './eggplant.jpg'
result = retina_face_detection(img_path)
p,l = result
print(p[0][1].item())
print(l)
print(f'face detection output: {result}.')