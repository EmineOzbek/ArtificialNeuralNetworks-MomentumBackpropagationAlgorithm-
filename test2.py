import numpy as np
from tabulate import tabulate  # Tablo formatında yazdırmak için

# Sigmoid aktivasyon fonksiyonu ve türevi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid aktivasyon fonksiyonunun türevi
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

# Kullanıcıdan parametreleri alma
print("Ağırlık değerlerini (w1, w2, w3, w4, w5, w6) ve bias değerlerini (b1, b2, b3) giriniz:")
w1 = float(input("w1 (Giriş 1 -> Gizli Nöron 1): "))
w2 = float(input("w2 (Giriş 2 -> Gizli Nöron 1): "))
w3 = float(input("w3 (Giriş 1 -> Gizli Nöron 2): "))
w4 = float(input("w4 (Giriş 2 -> Gizli Nöron 2): "))
w5 = float(input("w5 (Gizli Nöron 1 -> Çıkış): "))
w6 = float(input("w6 (Gizli Nöron 2 -> Çıkış): "))

b1 = float(input("Bias (Gizli Nöron 1 için): "))
b2 = float(input("Bias (Gizli Nöron 2 için): "))
b3 = float(input("Bias (Çıkış Nöronu için): "))

learning_rate = float(input("Öğrenme hızı ζ (0-1 aralığında): "))
epoch_max = int(input("Epochmax değeri: "))

# Kullanıcıdan alınan ağırlıkları ve biasları vektör olarak başlatma
weights_input_hidden = np.array([[w1, w3], [w2, w4]])  # Girişten gizli katmana
bias_hidden = np.array([[b1, b2]])                     # Gizli katman biasları
weights_hidden_output = np.array([[w5], [w6]])         # Gizli katmandan çıkışa
bias_output = np.array([[b3]])                         # Çıkış katmanı biası

# Ağırlık ve bias değerlerini saklamak için listeler
epoch_data = []

# Eğitim döngüsü
for epoch in range(epoch_max):
    # İleri yayılım
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Hata hesapla
    error = y - final_output

    # Geri yayılım
    delta_output = error * sigmoid_derivative(final_output)
    delta_hidden = np.dot(delta_output, weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Ağırlık ve bias güncellemeleri
    weights_hidden_output += np.dot(hidden_output.T, delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += np.dot(X.T, delta_hidden) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    # Hata kontrolü
    total_error = np.mean(np.abs(error))

    # Ağırlık ve bias değerlerini sakla
    if epoch < 50 or epoch >= epoch_max - 50:
        epoch_data.append([
            epoch + 1,
            weights_input_hidden[0, 0], weights_input_hidden[1, 0],
            weights_input_hidden[0, 1], weights_input_hidden[1, 1],
            bias_hidden[0, 0], bias_hidden[0, 1],
            weights_hidden_output[0, 0], weights_hidden_output[1, 0],
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

print("\n.\n.\n.")
print("\nSon 50 Epoch:")
print(tabulate(epoch_data[-50:], headers=headers, tablefmt="grid"))

# Eğitim Sonucu
print("\nEğitim tamamlandı!")
print("Girişler -> Tahminler")
for i in range(len(X)):
    hidden_input = np.dot(X[i], weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    prediction = sigmoid(final_input)
    print(f"{X[i]} -> {1 if prediction >= 0.5 else 0} (Gerçek: {y[i][0]})")
