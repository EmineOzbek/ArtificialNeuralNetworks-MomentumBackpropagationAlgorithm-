import numpy as np

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

# Ağ mimarisi: 2 giriş, 2 gizli, 1 çıkış nöronu
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Rastgele ağırlıkları ve biasları başlatma ([-1, 1] arasında)
np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
bias_output = np.random.uniform(-1, 1, (1, output_neurons))

# Öğrenme oranı
learning_rate = 0.1

# Eğitim döngüsü
epoch_max = 100000
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
    if epoch % 10000 == 0:
        total_error = np.mean(np.abs(error))
        print(f"Epoch: {epoch}, Hata: {total_error:.4f}")
        if total_error < 0.01:
            break

# Eğitim Sonucu
print("\nEğitim tamamlandı!")
print("Girişler -> Tahminler")
for i in range(len(X)):
    hidden_input = np.dot(X[i], weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    prediction = sigmoid(final_input)
    print(f"{X[i]} -> {1 if prediction >= 0.5 else 0} (Gerçek: {y[i][0]})")
