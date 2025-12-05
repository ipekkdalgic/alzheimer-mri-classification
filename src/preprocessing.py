from pathlib import Path

from PIL import Image, ImageOps
import numpy as np


def load_image(image_path):
    image_path = Path(image_path)
    img = Image.open(image_path).convert("L")  # "L" = grayscale
    return img


def preprocess_image(
    image_path,
    use_autocontrast=True,
    normalize=True,
    target_size=(176, 208),  # (width, height) - PIL için
):
    """
    Tek bir görüntü için temel preprocessing adımlarını uygular:
      - Gri tonlamaya çevirme
      - İsteğe bağlı auto-contrast
      - İsteğe bağlı resize (hepsini aynı boyuta getirme)
      - İsteğe bağlı normalization (0-1 aralığına)

    Dönüş:
        img_proc (np.ndarray): Şekli (H, W) olan float32 NumPy array.
    """
    # 1) Görüntüyü yükle
    img = load_image(image_path)

    # 2) Auto-contrast (istersek)
    if use_autocontrast:
        img = ImageOps.autocontrast(img)

    # 3) Hepsini aynı boyuta getir (PIL'de (width, height))
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)

    # 4) NumPy array'e çevir
    img_array = np.array(img).astype("float32")  # shape: (height, width)

    # 5) Normalize et (istersek)
    if normalize:
        min_val = img_array.min()
        max_val = img_array.max()
        img_array = (img_array - min_val) / (max_val - min_val + 1e-8)

    return img_array