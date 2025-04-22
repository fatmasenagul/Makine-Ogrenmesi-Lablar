1-)
Matris manipülasyonu nedir? 
Matematiksel matrislerin üzerinde yapılan işlemlerin genel adıdır. Bu işlemler toplama, çarpma, transpoz alma, tersini alma, determinant hesaplama, matrisin sırasını değiştirme veya yeniden boyutlandırma olabilir.

Özdeğer ve Özvektör nedir?
Matematikte, bir lineer transformasyon altında yönünü değiştirmeyen vektörlere özvektör denir.
v bir vektörse ve ϕ bir lineer dönüşümse, bir ⁁  skaleri için ϕ(v) = v⁁ eşitliğini sağlayan tüm  v vektörlerine ϕ'nin özvektörü, ve ⁁ skalerine de ϕ'nin özdeğeri denir.
Lineer dönüşümler vektörleri döndürerek, yansıtarak veya bükerek etkileyebilirler. Ancak v bu dönüşümün bir özvektörüyse, v bu dönüşümden sadece boyca uzayarak, kısalarak veya doğrultusunu değiştirmeden yönünü değiştirerek etkilenir. Boyca uzama ve kısalma miktarı ise bu dönüşümün özdeğerine tekabül eder.

Matris manipülasyonu, özdeğer ve özvektörler makine öğrenmesinin hangi yöntemler ve yaklaşımlarında kullanılmaktadır?

PCA (Principal Component Analysis):
PCA yüksek boyutlu verilerde en fazla bilgi içeren yönleri (bileşenleri) bulmak için kullanılan güçlü bir doğrusal dönüşüm tekniğidir.
PCA’nın temelinde matris manipülasyonu, özdeğer ve  özvektör kavramları yer alır
 
LDA (Linear Discriminant Analysis):
LDA, makine öğrenmesinde sınıflandırma için kullanılan hem boyut indirgeme hem de ayrım gücünü arttırma tekniğidir.
Scatter matrislerinin özdeğer ve özvektör analizleriyle çalışır.

Markov Zincirleri:
Markov zinciri, bir durumdan diğerine geçişin sadece mevcut duruma bağlı olduğu olasılık temelli bir sistemdir.
Geçiş olasılık matrislerinin özvektörlerini kullanılır.

Görüntü İşleme (Eigenfaces):
İnsan yüzü tanımada PCA’nın özvektör temelli hali.
Her yüz, bir özvektör kombinasyonu olarak temsil edilir.

##########

2-)
İlk kod hücresinde, linalg.eig fonksiyonunun (1,2,3) diagonal matrisinin özdeğer ve özvektörleri bulunmuş. Diagonal matrislerde özdeğerlerin köşegendeki değerlere eşit olduğu ve özvektörün birim matris olduğu gösterilmiş.

İkinci kod hücresinde, simetrik olmayan bir matrisin linalg.eig fonksiyonu ile özdeğerleri ve özvektörleri hesaplanmış. Bulunan özdeğelerin karmaşık sayı olduğu ve birbirlerinin eşleniği olduğu görülmüş. Özvektörlerin içerdiği değerlerin de karmaşık ve birbirinin eşleniği olduğu görülmüş.

Üçüncü hücrede kompleks sayılarla oluşturulmuş Hermit matrisine benzer bir matrisin özdeğer ve özvektörleri hesaplanmıştır. Özdeğerleri ve özvektörleri karmaşık sayılar çıkar.

Dördüncü hücrede numpy hassasiyet sınırı nedeni ile 1∓ 10⁻⁹ özdeğerinin bilgisayar tarafından tam olarak ayırt edilemediği ve sonucu 1 olarak gösterdiği gösterilmiştir. 

##########

3-)
linalg modülü kullanılan ve kullanılmayan kodları kıyaslarsak:

Özdeğerlerin verilişinde bir sıralama gözetilmediğinden dolayı iki sonuç aynıdır.
Cevap [5. 3. 7.] olarak bulunmuştur.

İki kodu kıyaslarsak linalg modülünü kullanmadığımız kodda 137 satırda sonuca ulaşabilirken linalg modülünü kullandığımız kodda 12 satırda aynı sonuca ulaşabiliyoruz.  

Linalg modülü kullanılmayan kodda, A - λI karakteristik matrisi oluşturulur.  
Daha sonra determinantı recursive hesaplayarak karakteristik polinom elde edilir.
Bu polinomun köklerini np.roots() ile bulur. bu buldukları özdeğerlere eşittir.
Linalg modülü kullandığımız kodda Numpy Ax = λx denkleminden λ'ları yani özdeğerleri ve x'leri yani özvektörleri bulur. 
Bu kodun altında BLAS ve LAPACK gibi güçlü kütüphaneler çalışır.

Linalg modülü kullanılan kod, linalg modülü kullanılmayan koda göre optimizasyonlar içerir.  
Yukarıda bahsettiğim BLAS ve LAPACK kütüphaneleri düşük seviyede c ya da Fortan ile yazıldığından ve bellek erişim optimizasyonu ve işlemci önbelleği kullanımına uygun çalıştıklarından dolayı çok hızlı çalışırlar.
Numpy, matris işlemlerinde küçük sayılardan oluşan farkları, sıfıra yakın determinantları vs. düzgün işler.
Linalg kullanılmamış kodda yuvarlama hatası görülme ihtimali çoktur.

Linalg kullanılan kod O(n!) gibi bir karmaşıklığa sahiptir.  
Kullanılmayan kod ise O(n³) karmaşıklığa sahiptir.


Kaynaklar:
"Özdeğerler ve özvektörler." Vikipedi, Wikimedia Foundation

Pattern Recognition and Machine Learning" — Christopher M. Bishop

Introduction to Machine Learning with Python  Andreas C. Müller & Sarah Guido


