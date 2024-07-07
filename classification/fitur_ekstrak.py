import os

import cv2
import h5py
import mahotas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from classification.constans import TRAIN_PATH, FIXED_SIZE


class FeatureExtract():
    def __init__(self, training_path=TRAIN_PATH):
        self.training_path = training_path
        self.train_labels = os.listdir(self.training_path)
        self.train_labels.sort()

    def rgb_bgr(self, image):
        """convert gambar dari rgb ke bgr"""
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_img

    def bgr_hsv(self, image):
        """convert gambar dari rgb ke hsv"""
        hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv_img

    def img_segmentation(self, rgb_img, hsv_img):
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

    def fd_hu_moments(self, segmented_img):
        """feature-descriptor-1: Hu Moments"""
        gray_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(gray_img)).flatten()
        return feature

    def fd_haralick(self, segmented_img):
        """feature-descriptor-2: Haralick Texture"""
        gray_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray_img).mean(axis=0)
        return haralick

    def fd_histogram(self, bins, mask=None):
        """feature-descriptor-3: Color Histogram"""
        image = cv2.cvtColor(self, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None,
                            [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def preprocess(self, image, size, show=False):
        """Preprocess gambar. Baca gambar, resize, ubah ke bgr,
        ubah ke hsv, segmentasi, feature descriptor ho moments,
        haralick, dan histogram."""
        # baca image dan resize ke fixed-size
        image = cv2.imread(image)
        image = cv2.resize(image, size)

        # jalankan fungsi preprocessing satu per satu
        bgr_image = self.rgb_bgr(image)
        hsv_image = self.bgr_hsv(bgr_image)
        segmented_img = self.img_segmentation(bgr_image, hsv_image)

        # jalankan fungsi feature descriptor
        fv_hu_moments = self.fd_hu_moments(segmented_img)
        fv_haralick = self.fd_haralick(segmented_img)
        fv_histogram = self.fd_histogram(segmented_img)

        if show:
            cv2.imshow(f'resize ke ukuran {size}', image)
            cv2.imshow('convert ke bgr', bgr_image)
            cv2.imshow('convert ke hsv', hsv_image)
            cv2.imshow('segmentasi', segmented_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # gabungin
        return np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    def feature_extraction(self, size=FIXED_SIZE):
        global_features = []
        labels = []
        # loop sub-folder pada training data
        for label in self.train_labels:
            # join path training data dan setiap sub-foldernya
            label_path = os.path.join(self.training_path, label)

            # loop gambar di setiap sub-folder
            for image_name in os.listdir(label_path):
                filename = os.path.join(label_path, image_name)
                global_feature = self.preprocess(filename, size)

                labels.append(label)
                global_features.append(global_feature)

            print(f"Memproses folder: {label}")
        print("Selesai global feature extraction")

        return global_features, labels


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
