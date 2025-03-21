import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, accuracy_score, classification_report
import time
import matplotlib.pyplot as plt

file_path = r"C:\Users\Hp\Desktop\2.2 lab1\naiveBayes\diabetes.csv"
df = pd.read_csv(file_path)  # data frame

# bağımsız Değişkenler X ve hedef değişken y
X = df.drop(columns=["Outcome"])  
y = df["Outcome"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modeli eğitiyorum
start_time = time.time()
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
end_time = time.time()

net_time = end_time - start_time
print(f"Geçen süre: {net_time}")
# 0.0015 kadar süre geçiyor

# accuracy ve sınıflandırma raporu
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Doğruluk Orani: {accuracy:.2f}")
print("\nSiniflandirma Raporu:")
print(report)

# karmaşıklık matrisini hesaplama 
cm = confusion_matrix(y_pred, y_test)

# matrisi görselleştirme
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
