import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

data= pd.read_csv(r"C:\Users\Hp\Desktop\Makine-Ogrenmesi-Lablar\2.LogisticRegression\pcos_dataset.csv")
X = data.drop('PCOS_Diagnosis', axis=1)
y = data['PCOS_Diagnosis']

# Eğitim ve test verileri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()

# Modeli oluşturdum ve eğittim
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahmin yapılıyor
y_pred = model.predict(X_test)

end_time = time.time()

print(f"geçen süre {end_time - start_time}")

# Modelin doğruluğu
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%\n")

# Sınıflandırma raporu ve karışıklık matrisi
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()

# doğruluk oranı yüzde 88.50

# geçen süre  0.0177

