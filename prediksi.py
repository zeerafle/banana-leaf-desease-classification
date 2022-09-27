from fitur_ekstrak import preprocess
import joblib
import cv2
import os
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

THICKNESS = 2
PATH_MODEL = os.path.join('output', 'rfc_model.sav')

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
# show an "Open" dialog box and return the path to the selected file
filename = askopenfilename()
print(f"File path: \n{filename}")

model = joblib.load(PATH_MODEL)

# filename = 'dataset\\train\\pestalotiopsis\\IMG_20210309_123810.jpg'
label_name = filename.split('/')[-2]
predicted_label_name = {0: 'cordana',
                        1: 'healthy',
                        2: 'pestalotiopsis',
                        3: 'sigatoka'}
image_feature = preprocess(filename, show=True).reshape(1, -1)

# predict gambar
predicted_label = model.predict(image_feature)[0]

# tampilkan gambar
image = cv2.imread(filename)
image = cv2.resize(image, (500, 500))

if label_name in predicted_label_name.values():
    cv2.putText(image, f'Label asli: {label_name}',
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, THICKNESS)
    print(f'Label asli: {label_name}')


cv2.putText(image, f'Label diprediksi: {predicted_label_name[predicted_label]}',
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, THICKNESS)
cv2.imshow('image', image)
cv2.waitKey(50000)
cv2.destroyAllWindows()

print(f'Label diprediksi: {predicted_label_name[predicted_label]}')
