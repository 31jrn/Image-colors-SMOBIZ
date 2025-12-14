import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_colors(img_rgb, bin_size=10):
    """
    Анализ цветового распределения.
    Вход:
      img_rgb: numpy array, shape (H, W, 3), dtype uint8, в порядке RGB
      bin_size: int, размер ячейки по каждой компоненте (например, 10)
    Возврат:
      most_freq_color: (R,G,B) uint8 - центральный цвет кубика самого частого
      rare_color: (R,G,B) uint8 - центральный цвет кубика самого редкого (с count>0)
      mask_most: boolean mask (H,W) пикселей, попавших в кубик most_freq
      mask_rare: boolean mask (H,W) пикселей, попавших в кубик rare
      counts: 1D array частот для всех кубиков (полезно для анализа)
      bins_shape: tuple (nx, ny, nz) - число бинов по каждой оси
    """
    if img_rgb.dtype != np.uint8:
        raise ValueError("Ожидается uint8 изображение (0..255).")

    h, w, _ = img_rgb.shape
    # число бинов по каждой оси (последний бин может быть чуть шире если 256 % bin_size != 0)
    nx = int(np.ceil(256 / bin_size))
    ny = nx
    nz = nx

    # Индексы бинов для каждого пикселя
    inds = img_rgb // bin_size  # целочисленное деление
    inds = inds.astype(np.int32)

    # Линейный индекс бина: i*ny*nz + j*nz + k
    lin_inds = (inds[:, :, 0] * ny * nz) + (inds[:, :, 1] * nz) + inds[:, :, 2]
    lin_inds_flat = lin_inds.ravel()

    # Подсчёт частот
    total_bins = nx * ny * nz
    counts = np.bincount(lin_inds_flat, minlength=total_bins)

    # Отбрасываем нулевые бины при поиске минимума
    nonzero_mask = counts > 0
    if not np.any(nonzero_mask):
        raise RuntimeError("В изображении нет пикселей (или некорректный вход).")

    # Индексы линейные самых частых и самых редких (с count > 0)
    idx_most = int(np.argmax(counts))
    # для минимального положительного:
    counts_nonzero = counts.copy()
    counts_nonzero[~nonzero_mask] = np.iinfo(counts_nonzero.dtype).max
    idx_rare = int(np.argmin(counts_nonzero))

    # Восстановим 3D индексы
    def unravel_index_lin(idx):
        i = idx // (ny * nz)
        rem = idx % (ny * nz)
        j = rem // nz
        k = rem % nz
        return int(i), int(j), int(k)

    ix_most, iy_most, iz_most = unravel_index_lin(idx_most)
    ix_rare, iy_rare, iz_rare = unravel_index_lin(idx_rare)

    # Центральный цвет соответствующего кубика (округление и ограничение 0..255)
    def bin_center(i):
        c = i * bin_size + bin_size / 2.0
        c = int(round(min(255, max(0, c))))
        return c

    most_freq_color = np.array([bin_center(ix_most), bin_center(iy_most), bin_center(iz_most)], dtype=np.uint8)
    rare_color = np.array([bin_center(ix_rare), bin_center(iy_rare), bin_center(iz_rare)], dtype=np.uint8)

    # Маски пикселей, попавших в эти кубики
    mask_most = (inds[:, :, 0] == ix_most) & (inds[:, :, 1] == iy_most) & (inds[:, :, 2] == iz_most)
    mask_rare = (inds[:, :, 0] == ix_rare) & (inds[:, :, 1] == iy_rare) & (inds[:, :, 2] == iz_rare)

    return most_freq_color, rare_color, mask_most, mask_rare, counts, (nx, ny, nz)

def visualize_results(img_rgb, most_color, rare_color, mask_most, mask_rare, save=False, out_prefix="out"):
    """
    Показывает исходное изображение, маски и цветовые квадраты.
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    axs[0].imshow(img_rgb)
    axs[0].set_title("Исходное изображение")
    axs[0].axis("off")

    # Маска самого частого: покажем пиксели цветом и остальное затемнённым
    overlay_most = img_rgb.copy().astype(np.float32)
    overlay_most[~mask_most] = (overlay_most[~mask_most] * 0.15)  # затемнить невыделенные пиксели
    axs[1].imshow(overlay_most.astype(np.uint8))
    axs[1].set_title(f"Самый частый цвет: {tuple(int(x) for x in most_color)}")
    axs[1].axis("off")

    overlay_rare = img_rgb.copy().astype(np.float32)
    overlay_rare[~mask_rare] = (overlay_rare[~mask_rare] * 0.15)
    axs[2].imshow(overlay_rare.astype(np.uint8))
    axs[2].set_title(f"Самый редкий цвет: {tuple(int(x) for x in rare_color)}")
    axs[2].axis("off")

    # Цветовые квадраты
    sw = np.ones((100, 100, 3), dtype=np.uint8)
    sw[:, :] = most_color
    sw2 = np.ones((100, 100, 3), dtype=np.uint8)
    sw2[:, :] = rare_color

    combined = np.vstack((sw, sw2))
    axs[3].imshow(combined)
    axs[3].set_title("Вверх: частый, вниз: редкий")
    axs[3].axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(f"{out_prefix}_analysis.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    # Пример использования:
    img_bgr = cv2.imread("depositphotos_14034535-stock-photo-forest.jpeg") # hvoynie_lesa.jpg
    if img_bgr is None:
        raise FileNotFoundError("Не удалось найти файл изображения. Проверь путь.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Параметр из методички: bin_size = 10
    bin_size = 10

    most_color, rare_color, mask_most, mask_rare, counts, bins_shape = analyze_colors(img_rgb, bin_size=bin_size)

    print("bins_shape:", bins_shape)
    print("Самый частый цвет (центроид кубика):", most_color, "пикселей:", counts.max())
    # наименьшее >0
    counts_nonzero = counts[counts > 0]
    print("Число пикселей в самом редком ненулевом кубике:", counts_nonzero.min())
    print("Самый редкий цвет (центроид кубика):", rare_color)

    visualize_results(img_rgb, most_color, rare_color, mask_most, mask_rare, save=False)
