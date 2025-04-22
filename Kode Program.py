import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# load Afiv dan processing
def load_avif_images(folder_path, img_size=(28, 28)):
    from pillow_avif import AvifImagePlugin 

    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.avif'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("L")
            img = img.resize(img_size)
            img_array = np.array(img).astype("float32") / 255.0
            images.append(img_array)
    return np.expand_dims(np.array(images), axis=-1)

# penambahkan Noise 
def add_noise(images, noise_factor=0.4):
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0., 1.)

# Load Dataset
folder_path = r"d:/autoencoder/bunga"
images_clean = load_avif_images(folder_path)
images_noisy = add_noise(images_clean)

# Tambahan cek folder (opsional tapi berguna)
import os
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"Folder tidak ditemukan: {folder_path}")

# Membangun Autoencoder
input_img = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Buat Model Autoencoder
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Training Model
autoencoder.fit(images_noisy, images_clean,
                epochs=100,
                batch_size=4,
                shuffle=True)

# Prediksi
decoded_imgs = autoencoder.predict(images_noisy)

# Visualisasi
n = min(10, len(images_clean))
plt.figure(figsize=(20, 6))
for i in range(n):
    # Input rusak
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(images_noisy[i].reshape(28, 28), cmap='gray')
    ax.set_title("Input Rusak")
    ax.axis('off')
    
    # Output hasil autoencoder
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Output Bersih")
    ax.axis('off')
    
    # Gambar asli
    ax = plt.subplot(3, n, i + 1 + n*2)
    plt.imshow(images_clean[i].reshape(28, 28), cmap='gray')
    ax.set_title("Gambar Asli")
    ax.axis('off')

plt.tight_layout()
plt.show()
