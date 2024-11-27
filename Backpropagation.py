import numpy as np

''' 
 XOR Veri Seti
x1     x2      y
0       0      0
0       1      1
1       0      1
1       1      0

'''

# Modelin başlangıç ağırlıklarını (w1, w2, w3, Ø), öğrenme hızı parametresini (ζ) ve eğitimin en fazla kaç tekrar
# yapılacağına ilişkin epoch sınırını(epoch_max) kullanıcıdan alma
print(
    "Ağırlık değerlerini(w1, w2, w3, Ø), öğrenme hızı parametresini(ζ) ve eğitimin en fazla kaç tekrar yapılacağına "
    "ilişkin epoch sınırını(epoch_max) giriniz:")
w1 = float(input("w1 (0-1 aralığında): "))
w2 = float(input("w2 (0-1 aralığında): "))
bias = float(input("Bias (0-1 aralığında): "))
learning_rate = float(input("Öğrenme hızı ζ (0-1 aralığında): "))
momentum = 0.8  # Momentum katsayısı
epoch_max = int(input("Epochmax değeri: "))

# Ağırlıkların ve bias'ın vektör olarak tanımlanması
weights = np.array([w1, w2])
bias_weight = bias

# XOR veri seti
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])  # Gerçek çıktılar

# Test verileri (son iki örnek)
X_test = np.array([
    [1, 0],
    [1, 1]
])
# Test seti için gerçek çıktılar
y_test = np.array([1, 0])


# Geriye yayılım sigmoid fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # σ(x) = 1 / 1 + e^-x

# Sigmoid fonksiyonunun türevi
def sigmoid_derivative(x):
    return x * (1 - x)


# Momentumlu geriye yayılım algoritması
previous_weight_update = np.zeros_like(
    weights)  # Önceki ağırlık güncelleme değerini tutar. Başlangıçta değer olmadığı için weights dizisiyle aynı
# boyutta elemanları 0 olan bir dizi oluşturuyoruz.
previous_bias_update = 0
targeted_error_threshold = 0.01  # Hedeflenen hata eşiği
epoch = 0
total_error = float(
    "inf")  # ilk etapta herhangi bir koşula takılmamak için başlangıç değeri olarak sonsuz değer veriyoruz.

while epoch < epoch_max and total_error > targeted_error_threshold:
    total_error = 0
    for i in range(len(X)):
        # İleri yayılım
        weighted_sum = np.dot(X[i], weights) + bias_weight  # z = x1.w1 + x2.w2 + θ
        prediction = sigmoid(weighted_sum)                  # y = 1 / 1 + exp(-z)

        # Hata hesapla
        error = y[i] - prediction  # e = d - y
        total_error += error ** 2

        # Geri yayılım (ağırlık ve bias güncelleme)
        delta = error * sigmoid_derivative(prediction)  # σ = e.y.(1-y)
        weight_update = learning_rate * delta * X[i] + momentum * previous_weight_update  # ΔW = η.σ.i + μ⋅ΔW(previous)
        bias_update = learning_rate * delta + momentum * previous_bias_update             # Δθ = η.σ.1 + μ⋅Δθ(previous)

        # Ağırlıkları güncelleyelim
        weights += weight_update
        bias_weight += bias_update

        # Momentum için güncellemeyi düzenleyelim
        previous_weight_update = weight_update
        previous_bias_update = bias_update

    # Hata kontrolü
    epoch += 1
    if total_error <= targeted_error_threshold:
        print(f"Öğrenme, {epoch} epoch sonunda ve {total_error:.4f} hatayla tamamlanmıştır.")
        break
else:
    print(
        "Epoch sınırına ulaşıldı, ancak öğrenme tamamlanamadı. Programdan çıkış yapılıyor. Farklı parametrelerle tekrar deneyiniz.")

# Test aşaması
for i in range(len(X_test)):
    # Test verisinde ileri yayılım
    weighted_sum = np.dot(X_test[i], weights) + bias_weight  # y = x1.w1 + x2.w2 + θ
    prediction = sigmoid(weighted_sum)                       # y = 1 / 1 + exp(-x1.w1 + x2.w2 + θ)
    prediction = 1 if prediction >= 0.5 else 0  # Tahmini 1 veya 0 olarak yuvarla

    # Hata ve mesaj çıktısı
    error = y_test[i] - prediction
    print(f"{i + 3}. veri vektöründe tahmin edilen değer {prediction}, gerçek değer {y_test[i]} ve hata {error}.")
