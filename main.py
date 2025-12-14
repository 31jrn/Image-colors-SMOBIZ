import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("depositphotos_14034535-stock-photo-forest.jpeg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_original = img_rgb.copy()
img = img_rgb.astype(np.float32)

M_0 = np.array([100, 130, 100], dtype=np.float32)  # (R,G,B)


def euclidean(pixel, ref):
    return np.linalg.norm(pixel - ref)


def manhattan_distance(pixel, ref):
    return np.sum(np.abs(pixel - ref))


def cosine_distance(pixel, ref):
    if np.linalg.norm(pixel) == 0 or np.linalg.norm(ref) == 0:
        return 1.0
    else:
        cos_sim = np.dot(pixel, ref) / (np.linalg.norm(pixel) * np.linalg.norm(ref))
        return 1 - cos_sim


def pixel_find(img, M_0, metric_type, threshold):
    result = img.copy()
    height, width, _ = img.shape
    highlight = np.array([255, 0, 0], dtype=np.float32)
    alpha = 0.6
    for i in range(height):
        for j in range(width):
            pixel = img[i, j]

            if metric_type == "euclidean":
                d = euclidean(pixel, M_0)
            elif metric_type == "manhattan":
                d = manhattan_distance(pixel, M_0)
            elif metric_type == "cosine":
                d = cosine_distance(pixel, M_0)
            else:
                raise ValueError("Неизвестная метрика")

            if d < threshold:
                result[i, j] = [255,0,0]
    return result


def plot(original,result):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(original.astype(np.uint8))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Пиксели, близкие по цвету")
    plt.imshow(result.astype(np.uint8))
    plt.axis('off')
    plt.show()


def menu():
    while True:
        choice = int(input(
            "Выберите метрику: \n 1) Евклидова метрика\n 2) Манхеттенова метрика\n 3) Косинус угла между векторами\n 0) Выход из программы\n"))
        if choice == 0:
            return None, None
        elif choice == 1:
            print("Выбрана евклидова метрика")
            return "euclidean", 40
        elif choice == 2:
            print("Выбрана манхеттенова метрика")
            return "manhattan", 70
        elif choice == 3:
            print("Выбрана метрика косинуса между углами векторами цвета")
            return "cosine", 0.05
        else:
            print("Неверный ввод, попробуйте снова")
            return None, None



if __name__ == '__main__':
    metric_type, threshold = menu()
    result_img = pixel_find(img, M_0, metric_type, threshold)
    plot(img_original ,result_img)

