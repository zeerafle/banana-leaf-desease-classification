from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
from sklearn.preprocessing import MinMaxScaler

TRAIN_PATH = os.path.join('dataset', 'train')
TEST_PATH = os.path.join('dataset', 'test')
FIXED_SIZE = (500, 500)
BINS = 8
H5_TRAIN_DATA = os.path.join('output', 'train_data.h5')
H5_TRAIN_LABELS = os.path.join('output', 'train_labels.h5')


def rgb_bgr(image):
    """convert gambar dari rgb ke bgr"""
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img


def bgr_hsv(rgb_img):
    """convert gambar dari rgb ke hsv"""
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img


def img_segmentation(rgb_img, hsv_img):
    """segmentasi gambar untuk ekstraksi warna hijau dan coklat"""
    lower_green = np.array([25, 0, 20])
    upper_green = np.array([100, 255, 255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img, rgb_img, mask=healthy_mask)
    lower_brown = np.array([10, 0, 10])
    upper_brown = np.array([30, 255, 255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result


def fd_hu_moments(image):
    """feature-descriptor-1: Hu Moments"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    """feature-descriptor-2: Haralick Texture"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def fd_histogram(image, mask=None):
    """feature-descriptor-3: Color Histogram"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        BINS, BINS, BINS], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def preprocess(file_name, show=False):
    """Preprocess gambar. Baca gambar, resize, ubah ke bgr,
    ubah ke hsv, segmentasi, feature descriptor ho moments,
    haralick, dan histogram."""
    # baca image dan resize ke fixed-size
    image = cv2.imread(file_name)
    image = cv2.resize(image, FIXED_SIZE)

    # jalankan fungsi preprocessing satu per satu
    bgr_image = rgb_bgr(image)
    hsv_image = bgr_hsv(bgr_image)
    segmented_img = img_segmentation(bgr_image, hsv_image)

    # jalankan fungsi feature descriptor
    fv_hu_moments = fd_hu_moments(segmented_img)
    fv_haralick = fd_haralick(segmented_img)
    fv_histogram = fd_histogram(segmented_img)

    if show:
        cv2.imshow(f'resize ke ukuran {FIXED_SIZE}', image)
        cv2.imshow('convert ke bgr', bgr_image)
        cv2.imshow('convert ke hsv', hsv_image)
        cv2.imshow('segmentasi', segmented_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # gabungin
    return np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

def main():
    # dapatkan label training
    train_labels = os.listdir(TRAIN_PATH)

    # urutkan training label
    train_labels.sort()
    print(train_labels)

    # variabel penampung vektor fitur dan label
    global_features = []
    labels = []

    # loop sub-folder pada training data
    for training_name in train_labels:
        # join path training data dan setiap sub-folder nya
        dir = os.path.join(TRAIN_PATH, training_name)

        # dapatkan training label
        current_label = training_name

        # loop gambar di setiap sub-folder
        for image_name in os.listdir(dir):
            # dapatkan path ke file image
            file_name = os.path.join(dir, image_name)

            global_feature = preprocess(file_name)

            # tambahkan ke dalam variabel penampung yang sudah dibuat
            labels.append(current_label)
            global_features.append(global_feature)

        print(f"Memroses folder: {current_label}")

    print("Selesai global feature extraction")

    print(f"Ukuran vektor fitur: {np.array(global_features).shape}")
    print(f"Ukuran training label: {np.array(labels).shape}")

    # encode target labels
    target_names = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print("training label encoded")

    # ubah skala fitur menjadi (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print("feature vector normalized...")

    print(f"Target labels: {target}")
    print(f"Target labels shape: {target.shape}")

    # simpan vektor feature ke file bentuk hdf5
    h5f_data = h5py.File(H5_TRAIN_DATA, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    # simpan label training ke file bentuk hdf5
    h5f_label = h5py.File(H5_TRAIN_LABELS, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

if __name__ == 'main':
    main()
