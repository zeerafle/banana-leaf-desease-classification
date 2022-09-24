from fitur_ekstrak import preprocess
import h5py
import numpy as np
import os
import warnings
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
warnings.filterwarnings('ignore')

NUM_TREES = 100
SEED = 9
FIXED_SIZE = (500, 500)
TRAIN_PATH = os.path.join('dataset', 'train')
TEST_PATH = os.path.join('dataset', 'test')
H5_TRAIN_DATA = os.path.join('output', 'train_data.h5')
H5_TRAIN_LABELS = os.path.join('output', 'train_labels.h5')

# dapatkan training labels
train_labels = os.listdir(TRAIN_PATH)

# sort the training labels
train_labels.sort()

if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)

# variabel untuk menyimpan hasil dan nama
results = []
names = []

# import fitur vektor dan label yang telah ditrain
h5f_data  = h5py.File(H5_TRAIN_DATA, 'r')
h5f_label = h5py.File(H5_TRAIN_LABELS, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# cek shape fitur vektor dan label
print(f'Ukuran vektor fitur training: {global_features.shape}')
print(f'Ukuran training label: {global_labels.shape}')

X_train = global_features
y_train = global_labels

model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=SEED)
model.fit(X_train, y_train)

# EVALUATION MODEL ====================

# dapatkan label training
test_labels = os.listdir(TEST_PATH)

# urutkan training label
test_labels.sort()
print(test_labels)

# variabel penampung vektor fitur dan label
X_test = []
y_test = []

# loop sub-folder pada training data
for test_name in test_labels:
    # join path training data dan setiap sub-folder nya
    dir = os.path.join(TEST_PATH, test_name)

    # dapatkan training label
    current_label = test_name

    # loop gambar di setiap sub-folder
    for image_name in os.listdir(dir):
        # dapatkan path ke file image
        file_name = os.path.join(dir, image_name)
        
        global_feature = preprocess(file_name)
        
        # tambahkan ke dalam variabel penampung yang sudah dibuat
        y_test.append(current_label)
        X_test.append(global_feature)

    print(f"Memroses folder: {current_label}")

print("Selesai global feature extraction")

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Ukuran vektor fitur test: {X_test.shape}")
print(f"Ukuran test label: {y_test.shape}")

# encode target labels
target_names = np.unique(test_labels)
le = LabelEncoder()
encoded_y_test = le.fit_transform(y_test)
print("training label encoded")

# ubah skala fitur menjadi (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X_test = scaler.fit_transform(X_test)
print("feature vector normalized...")

# tampilkan confusion_matrix
y_predict = model.predict(rescaled_X_test)
ConfusionMatrixDisplay.from_predictions(encoded_y_test, y_predict).plot()

# cetak classification_report
print(classification_report(encoded_y_test, y_predict))

# save model
model_name = 'rfc_model.sav'
joblib.dump(model, os.path.join('output', model_name))
