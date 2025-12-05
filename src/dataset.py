from pathlib import Path
import numpy as np

from .preprocessing import preprocess_image


# Basit label map:
# 0 = Demans yok (NonDemented)
# 1 = Demans var (VeryMild / Mild / Moderate)
BINARY_LABELS = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 1,
    "ModerateDemented": 1,
}


def load_image_paths(data_root="data/raw"):
    """
    data_root altındaki sınıf klasörlerini ve içlerindeki .jpg dosyaları listeler.

    Dönüş:
        samples: [(image_path (Path), class_name (str)), ...]
    """
    data_root = Path(data_root)
    samples = []

    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        for img_path in sorted(class_dir.glob("*.jpg")):
            samples.append((img_path, class_name))

    return samples


def load_dataset(
    data_root="data/raw",
    use_autocontrast=True,
    normalize=True,
    limit_per_class=None,
):
    """
    Tüm dataset'i RAM'e yükler ve preprocess eder.

    Parametreler:
        data_root (str): Sınıf klasörlerinin bulunduğu kök dizin.
        use_autocontrast (bool): preprocess_image'e geçilir.
        normalize (bool): preprocess_image'e geçilir.
        limit_per_class (int veya None): Her sınıftan yüklenecek maksimum görüntü sayısı.
                                         None ise tüm görüntüler yüklenir.

    Dönüş:
        X (np.ndarray): Şekli (N, H, W) olan float32 array (0-1 aralığında).
        y (np.ndarray): Uzunluğu N olan int32 label vektörü (0 = yok, 1 = var).
        class_names (list[str]): Her örneğin orijinal sınıf adı.
    """
    data_root = Path(data_root)

    X_list = []
    y_list = []
    class_names = []

    # Sınıf klasörlerini dolaş
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        binary_label = BINARY_LABELS.get(class_name)

        if binary_label is None:
            print(f"Warning: class '{class_name}' is not in BINARY_LABELS, skipping.")
            continue

        count = 0
        for img_path in sorted(class_dir.glob("*.jpg")):
            if limit_per_class is not None and count >= limit_per_class:
                break

            img_proc = preprocess_image(
                img_path,
                use_autocontrast=use_autocontrast,
                normalize=normalize,
            )

            X_list.append(img_proc)
            y_list.append(binary_label)
            class_names.append(class_name)
            count += 1

        print(f"Loaded {count} images from class '{class_name}' (label={binary_label}).")

    X = np.stack(X_list).astype("float32")  # (N, H, W)
    y = np.array(y_list, dtype="int32")

    print("Final dataset shape:", X.shape)
    print("Label vector shape:", y.shape)

    return X, y, class_names
