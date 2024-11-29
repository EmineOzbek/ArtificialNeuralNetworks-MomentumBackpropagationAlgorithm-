import numpy as np

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

    # Her epoch'ta ağırlık ve bias değerlerini yazdır
    print(f"\nEpoch: {epoch + 1}")
    print(f"Toplam Hata: {total_error:.4f}")
    print(f"Ağırlıklar (W1, W2, W3, W4):\n{weights_input_hidden}")
    print(f"Bias (B1, B2):\n{bias_hidden}")
    print(f"Ağırlıklar (W5, W6):\n{weights_hidden_output}")
    print(f"Bias (B3):\n{bias_output}")

    # Hedef hata değerine ulaşıldıysa eğitimi sonlandır
    if total_error < 0.01:
        print(f"\nEğitim, {epoch + 1} epoch sonunda ve {total_error:.4f} hatayla tamamlanmıştır.")
        break
else:
    print(f"\nEpoch sınırına ulaşıldı ({epoch_max} epoch), ancak öğrenme tamamlanamadı. Hata: {total_error:.4f}")

# Eğitim Sonucu
print("\nEğitim tamamlandı!")
print("Girişler -> Tahminler")
for i in range(len(X)):
    hidden_input = np.dot(X[i], weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    prediction = sigmoid(final_input)
    print(f"{X[i]} -> {1 if prediction >= 0.5 else 0} (Gerçek: {y[i][0]})")
