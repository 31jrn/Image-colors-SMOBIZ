"""
segmentation_by_color.py

Упрощённая и аккуратно оформленная версия программы:
- загрузка изображения (BGR -> RGB)
- показ изображения и размеров
- выбор опорной точки (ввод координат)
- выбор метрики расстояния и порога
- быстрый векторный поиск пикселей, близких по цвету
- визуализация: слева — исходное изображение с помеченной точкой,
  справа — изображение с выделенными пикселями

Запуск:
    python segmentation_by_color.py

Требования:
    opencv-python, numpy, matplotlib
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


# -------------------------
# Утилиты по загрузке/визуализации
# -------------------------
def load_image_rgb(path: str) -> np.ndarray:
    """Загрузить изображение (возвращает RGB uint8). Вызывает FileNotFoundError при ошибке."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Не найден файл изображения: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def show_image_and_size(img: np.ndarray) -> Tuple[int, int]:
    """Показать изображение и вывести его размеры (ширина, высота). Возвращает (h, w)."""
    h, w = img.shape[:2]
    print(f"Размер изображения: ширина = {w} px, высота = {h} px")
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title("Исходное изображение")
    plt.axis("off")
    plt.show()
    return h, w


def plot(
    original: np.ndarray,
    result: np.ndarray,
    x: Optional[int] = None,
    y: Optional[int] = None,
) -> None:
    """
    Отобразить исходное изображение (с помеченной точкой, если заданы x,y)
    и результат сегментации.
    original: RGB uint8
    result: RGB (или float) изображение — будет приведено к uint8 для показа
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(original.astype(np.uint8))
    if x is not None and y is not None:
        # Маркер точки: крестик и подпись координат
        plt.scatter([x], [y], c="red", s=80, marker="x")
        plt.text(x + 5, y + 5, f"({x},{y})", color="red", fontsize=9)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Пиксели, близкие по цвету")
    plt.imshow(result.astype(np.uint8))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# Метрики расстояния (векторные варианты)
# -------------------------
def euclidean_distance_array(pixels: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    pixels: (N,3) float
    ref: (3,) float
    Возвращает массив расстояний (N,)
    """
    diff = pixels - ref.reshape(1, 3)
    return np.linalg.norm(diff, axis=1)


def manhattan_distance_array(pixels: np.ndarray, ref: np.ndarray) -> np.ndarray:
    diff = np.abs(pixels - ref.reshape(1, 3))
    return np.sum(diff, axis=1)


def cosine_distance_array(pixels: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # 1 - cosine_similarity; защищаемся от нулевых векторов
    ref_norm = np.linalg.norm(ref)
    pixels_norm = np.linalg.norm(pixels, axis=1)
    # если ref_norm == 0 или пиксель нулевой -> считаем дистанцию = 1 (максимум для 1-cos)
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = pixels.dot(ref)
        denom = pixels_norm * ref_norm
        cos_sim = np.divide(dot, denom, out=np.zeros_like(dot), where=(denom != 0))
    return 1.0 - cos_sim


# -------------------------
# Основная логика: поиск похожих пикселей (векторно)
# -------------------------
def find_similar_pixels_by_color(
    img_float: np.ndarray, reference_color: np.ndarray, metric: str, threshold: float
) -> np.ndarray:
    """
    img_float: изображение float32, форма (h, w, 3)
    reference_color: float32 (3,)
    metric: 'euclidean' | 'manhattan' | 'cosine'
    threshold: порог соответствует выбранной метрике
    Возвращает: result_img (h, w, 3) uint8 — копия img, где подходящие пиксели помечены красным.
    """
    h, w = img_float.shape[:2]
    pixels = img_float.reshape(-1, 3)  # (N,3)

    if metric == "euclidean":
        distances = euclidean_distance_array(pixels, reference_color)
    elif metric == "manhattan":
        distances = manhattan_distance_array(pixels, reference_color)
    elif metric == "cosine":
        distances = cosine_distance_array(pixels, reference_color)
    else:
        raise ValueError(f"Неизвестная метрика: {metric}")

    mask = distances < threshold  # булев массив длины N
    mask2d = mask.reshape(h, w)

    result = (img_float.copy()).astype(np.uint8)
    # Помечаем выбранные пиксели красным (R,G,B) — в формате RGB
    result[mask2d] = np.array([255, 0, 0], dtype=np.uint8)

    return result


# -------------------------
# Ввод координат опорной точки
# -------------------------
def choose_point(h: int, w: int) -> Tuple[int, int]:
    """Запрашивает у пользователя координаты x,y (целые). Возвращает (x,y)."""
    while True:
        try:
            raw_x = input(
                f"Введите координату x (0 … {w - 1}) или 'q' для выхода: "
            ).strip()
            if raw_x.lower() == "q":
                print("Выход.")
                sys.exit(0)
            raw_y = input(
                f"Введите координату y (0 … {h - 1}) или 'q' для выхода: "
            ).strip()
            if raw_y.lower() == "q":
                print("Выход.")
                sys.exit(0)

            x = int(raw_x)
            y = int(raw_y)

            if 0 <= x < w and 0 <= y < h:
                return x, y
            else:
                print("Координаты вне изображения. Повторите ввод.")
        except ValueError:
            print("Ошибка ввода. Введите целые числа.")


# -------------------------
# Меню выбора метрики и порога
# -------------------------
def menu() -> Tuple[Optional[str], Optional[float]]:
    """Возвращает (metric_name, threshold) или (None, None) для выхода."""
    print("\nВыберите метрику:")
    print(" 1) Евклидова метрика")
    print(" 2) Манхэттенова метрика")
    print(" 3) Косинусная метрика (1 - cos)")
    print(" 0) Выход")
    while True:
        try:
            choice = int(input("Ваш выбор: "))
        except ValueError:
            print("Неверный ввод, повторите.")
            continue

        if choice == 0:
            return None, None
        elif choice == 1:
            print("Выбрана евклидова метрика.")
            # Рекомендация порога зависит от яркости изображения; 40 — пример
            return "euclidean", 40.0
        elif choice == 2:
            print("Выбрана манхэттенова метрика.")
            return "manhattan", 70.0
        elif choice == 3:
            print("Выбрана косинусная метрика.")
            return "cosine", 0.05
        else:
            print("Неверный выбор, повторите.")


# -------------------------
# Main
# -------------------------
def main(image_path: str = "hvoynyi_les.jpeg") -> None:
    try:
        img_rgb = load_image_rgb(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Показываем изображение и размеры
    h, w = show_image_and_size(img_rgb)

    # Выбор точки (пользователь вводит x,y)
    x, y = choose_point(h, w)

    # Опорный цвет — берём из изображения (numpy indexing: [y, x])
    ref_color = img_rgb[y, x].astype(np.float32)
    print(
        f"Опорный цвет M0 = (R={int(ref_color[0])}, G={int(ref_color[1])}, B={int(ref_color[2])})"
    )

    # Выбор метрики и порога
    metric, threshold = menu()
    if metric is None:
        print("Завершение работы.")
        return

    # Для вычислений используем float32
    img_float = img_rgb.astype(np.float32)

    # Поиск похожих пикселей
    result_img = find_similar_pixels_by_color(img_float, ref_color, metric, threshold)

    # Визуализация: исходное с точкой и результат
    plot(img_rgb, result_img, x, y)


if __name__ == "__main__":
    # При необходимости, передайте путь к файлу как аргумент:
    # python segmentation_by_color.py path/to/image.jpg
    import argparse

    parser = argparse.ArgumentParser(
        description="Сегментация по цвету (выбор опорного пикселя)."
    )
    parser.add_argument(
        "--image", "-i", type=str, default="hvoynyi_les.jpeg", help="Путь к изображению"
    )
    args = parser.parse_args()
    main(args.image)
