import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time

class LogisticRegressionBayes:
    
    def Sigmoid(self, z):
        return 1/(1+np.exp(-z)) 
    
    
    def fit(self, X, y, learning_rate, iterations, lambda_reg = 0.1):
        
        
        m, n = X.shape
        weights = np.zeros(n)
        bias = 0
        # başta ağırlıkları ve biası 0 olarak veriyoruz
        
        # Gradient Descent 
        for i in range(iterations):
            # lineer model
            z = np.dot(X, weights) + bias
            
            # sigmoid fonksiyonu uygulama
            y_pred = self.Sigmoid(z)
            
            # Loss function türevleri (gradyanlar)
            dw = (1 / m) * np.dot(X.T, (y_pred - y)) + (lambda_reg / m) * weights # regularizasyon
            db = (1/m) * np.sum(y_pred - y)
            
            # ağırlıkları güncelleme
            weights -= learning_rate * dw
            bias -= learning_rate * db
            
            # her 100 iterasyonda bir kayıp değerini yazdırır
            if i % 100 == 0:
                loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1-y_pred))
                print(f" iterasyon {i}: Loss = {loss:.4f}")
                
        return weights, bias
    
    def predict(self, X, weights, bias):
        z = np.dot(X, weights) + bias
        y_pred = self.Sigmoid(z)
        return np.where(y_pred >= 0.5, 1, 0)
    
    
data= pd.read_csv(r"C:\Users\Hp\Desktop\Makine-Ogrenmesi-Lablar\2.LogisticRegression\pcos_dataset.csv")
X = data.drop('PCOS_Diagnosis', axis=1)
y = data['PCOS_Diagnosis']

# Eğitim ve test verileri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##### algoritmayı optimize etmek için bunu ekliyorum
scaler = StandardScaler()

# Eğitim verisini ölçeklendirdim
X_train_scaled = scaler.fit_transform(X_train)

# Test verisini ölçeklendirdim (sadece X_train üzerinden öğrenilen parametrelerle)
X_test_scaled = scaler.transform(X_test)


# Modeli oluşturdum ve eğittim
learning_rate = 0.1
iterations = 1000

start_time = time.time()

model = LogisticRegressionBayes()
weights, bias = model.fit(X_train_scaled, y_train, learning_rate= learning_rate, iterations= iterations)

# Tahmin yapılıor
y_pred = model.predict(X_test_scaled,weights= weights, bias= bias )

end_time = time.time()

print(f"geçen süre: {end_time - start_time}")
# Modelin doğruluğu
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%\n")

# Sınıflandırma raporu ve karışıklık matrisi
print("Siniflandirma Raporu:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

# model doğruluk oranı %81

# modelin artık doğruluk oranı artık 88.50
# geçen süre 0.1600