from fitur_ekstrak import preprocess
import joblib
import cv2
import os

PATH_MODEL = os.path.join('output', 'rfc_model.sav')

model = joblib.load(PATH_MODEL)

PATH = 'dataset\\train\\cordana\\1615455264278.jpg'
label_name = PATH.split('\\')[-2]
image_feature = preprocess(PATH, show=True).reshape(1, -1)

predicted_label = model.predict(image_feature)

if predicted_label == 0:
    predicted_label_name = 'cordana'
elif predicted_label == 1:
    predicted_label_name = 'healthy'
elif predicted_label == 2:
    predicted_label_name = 'pestalotiopsis'
elif predicted_label == 3:
    predicted_label_name = 'sigatoka'
else:
    predicted_label_name = 'HAYOLOH'

print(f'Label asli: {label_name}')
print(f'Label diprediksi: {predicted_label_name}')

image = cv2.imread(PATH)
image = cv2.resize(image, (500, 500))
THICKNESS = 2
cv2.putText(image, f'Label asli: {label_name}',
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, THICKNESS)
cv2.putText(image, f'Label diprediksi: {predicted_label_name}',
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, THICKNESS)
cv2.imshow('image', image)
cv2.waitKey(50)
cv2.destroyAllWindows()
