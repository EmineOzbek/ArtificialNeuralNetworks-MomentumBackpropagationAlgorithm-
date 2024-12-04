import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # Tablo formatında yazdırmak için

# Sigmoid aktivasyon fonksiyonu ve türevi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR veri seti (2 giriş, 1 çıkış)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])  # Gerçek çıktılar

# Kullanıcıdan model parametrelerini alma
print("Ağırlık değerlerini (w1, w2, w3, w4, w5, w6) ve bias değerlerini (b1, b2, b3) giriniz:")
w1 = float(input("w1 (Giriş 1 -> Ara Nöron 1): "))
w2 = float(input("w2 (Giriş 2 -> Ara Nöron 1): "))
w3 = float(input("w3 (Giriş 1 -> Ara Nöron 2): "))
w4 = float(input("w4 (Giriş 2 -> Ara Nöron 2): "))
w5 = float(input("w5 (Ara Nöron 1 -> Çıkış): "))
w6 = float(input("w6 (Ara Nöron 2 -> Çıkış): "))

b1 = float(input("Bias (Ara Nöron 1 için): "))
b2 = float(input("Bias (Ara Nöron 2 için): "))
b3 = float(input("Bias (Çıkış Nöronu için): "))

learning_rate = float(input("Öğrenme hızı ζ (0-1 aralığında): "))
momentum = float(input("Momentum katsayısı μ (0-1 aralığında): "))
epoch_max = int(input("Epochmax değeri: "))

# Ağırlıklar ve bias'lar
weights_input_middleLayer = np.array([[w1, w3], [w2, w4]])  # Girişten ara katmana
bias_middleLayer = np.array([[b1, b2]])                     # Ara katman biasları
weights_middleLayer_output = np.array([[w5], [w6]])         # Ara katmandan çıkışa
bias_output = np.array([[b3]])                              # Çıkış katmanı biası

# Önceki ağırlık güncellemeleri elemanları 0 olan diziler oluşturuldu.
prev_weight_update_middleLayer_output = np.zeros_like(weights_middleLayer_output)
prev_weight_update_input_middleLayer = np.zeros_like(weights_input_middleLayer)
prev_bias_update_output = np.zeros_like(bias_output)
prev_bias_update_middleLayer = np.zeros_like(bias_middleLayer)

# Ağırlık ve bias değerlerini saklamak için listeler
epoch_data = []
error_list = []  # Hata değerlerini saklayacağımız liste

# Eğitim döngüsü
for epoch in range(epoch_max):
    # İleri yayılım
    middleLayer_input = np.dot(X, weights_input_middleLayer) + bias_middleLayer # z = x1.w1 + x1.w3 + b1 + x2.w2 + x2.w4 + b2
    middleLayer_output = sigmoid(middleLayer_input)                             # y = 1 / 1 + exp(-z)

    final_input = np.dot(middleLayer_output, weights_middleLayer_output) + bias_output # z = y1.w5 + y2.w6 + b3
    final_output = sigmoid(final_input)                                                # y = 1 / 1 + exp(-z)

    # Hata hesapla
    error = y - final_output

    # Geri yayılım
    delta_output = error * sigmoid_derivative(final_output) # d = e.y.(1-y)
    delta_middleLayer = np.dot(delta_output, weights_middleLayer_output.T) * sigmoid_derivative(middleLayer_output)

    # Momentumlu ağırlık güncelleme
    weight_update_middleLayer_output = learning_rate * np.dot(middleLayer_output.T, delta_output) + momentum * prev_weight_update_middleLayer_output
    weights_middleLayer_output += weight_update_middleLayer_output
    prev_weight_update_middleLayer_output = weight_update_middleLayer_output

    bias_update_output = learning_rate * np.sum(delta_output, axis=0, keepdims=True) + momentum * prev_bias_update_output
    bias_output += bias_update_output
    prev_bias_update_output = bias_update_output

    weight_update_input_middleLayer = learning_rate * np.dot(X.T, delta_middleLayer) + momentum * prev_weight_update_input_middleLayer
    weights_input_middleLayer += weight_update_input_middleLayer
    prev_weight_update_input_middleLayer = weight_update_input_middleLayer

    bias_update_middleLayer = learning_rate * np.sum(delta_middleLayer, axis=0, keepdims=True) + momentum * prev_bias_update_middleLayer
    bias_middleLayer += bias_update_middleLayer
    prev_bias_update_middleLayer = bias_update_middleLayer

    # Hata kontrolü
    total_error = np.mean(np.abs(error))

    # Hata değerini listeye ekle
    error_list.append(total_error)

    # Ağırlık ve bias değerlerini sakla (ilk 50 ve son 50 için)
    if epoch < 50 or epoch >= epoch_max - 50:
        epoch_data.append([
            epoch + 1,
            weights_input_middleLayer[0, 0], weights_input_middleLayer[1, 0],
            weights_input_middleLayer[0, 1], weights_input_middleLayer[1, 1],
            bias_middleLayer[0, 0], bias_middleLayer[0, 1],
            weights_middleLayer_output[0, 0], weights_middleLayer_output[1, 0],
            bias_output[0, 0],
            total_error
        ])

    # Hedef hata değerine ulaşıldıysa eğitimi sonlandır
    if total_error < 0.01:
        print(f"\nEğitim, {epoch + 1} epoch sonunda ve {total_error:.4f} hatayla tamamlanmıştır.")
        break
else:
    print(f"\nEpoch sınırına ulaşıldı ({epoch_max} epoch), ancak öğrenme tamamlanamadı. Hata: {total_error:.4f}")

# İlk 50 ve Son 50 Epoch'u konsola yazdır
headers = ["Epoch", "w1", "w2", "w3", "w4", "b1", "b2", "w5", "w6", "b3", "Toplam Hata"]
print("\nİlk 50 Epoch:")
print(tabulate(epoch_data[:50], headers=headers, tablefmt="grid"))

print("\n. \n. \n.")
print("\nSon 50 Epoch:")
print(tabulate(epoch_data[-50:], headers=headers, tablefmt="grid"))

# Eğitim Sonucu
print("\nEğitim tamamlandı!")
print("Girişler -> Tahminler")
for i in range(len(X)):
    middleLayer_input = np.dot(X[i], weights_input_middleLayer) + bias_middleLayer
    middleLayer_output = sigmoid(middleLayer_input)

    final_input = np.dot(middleLayer_output, weights_middleLayer_output) + bias_output
    prediction = sigmoid(final_input)
    print(f"{X[i]} -> {1 if prediction >= 0.5 else 0} (Gerçek: {y[i][0]})")

# Hata grafiği oluşturma
plt.plot(error_list, label="Hata")
plt.xlabel("Epoch")
plt.ylabel("Toplam Hata")
plt.title("Eğitim Süresince Toplam Hata")
plt.legend()
plt.grid(True)
plt.show()