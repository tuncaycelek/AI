# AI
        Aslan, K. (2025), "Yapay Zekâ, Makine Öğrenmesi ve Veri Bilimi Kursu", Sınıfta Yapılan Örnekler ve Özet Notlar, 
            C ve Sistem Programcıları Derneği, İstanbul.

# AI Eğitim Kodlarım

Bu repo, yapay zekâ, makine öğrenimi ve istatistik konularını öğretmek amacıyla hazırlanmış **Python tabanlı eğitim örneklerini** içermektedir. Kodlar; temel istatistikten başlayarak veri ön işleme, encoder yöntemleri, aktivasyon fonksiyonları, ölçekleme teknikleri ve Keras ile sinir ağı uygulamalarına kadar ilerleyen bir öğrenme yolunu takip eder.

---

## 📊 Temel İstatistik ve Olasılık

### 1_Standart_Deviation.py  
Bu dosya, bir veri kümesinin standart sapmasının Python kullanılarak nasıl hesaplandığını gösterir. Amaç, veri dağılımının ortalama etrafındaki yayılımını anlamaktır.

### 2_Variance.py  
Bu örnek, varyans kavramını ele alır ve bir veri setindeki değerlerin ne kadar dağıldığını ölçmenin matematiksel ve programatik yolunu öğretir.

### 3_Gaussian_Distribution_Cartesian.py  
Gaussian (Normal) dağılımının Kartezyen koordinat sisteminde grafiksel gösterimini içerir ve dağılımın temel özelliklerini görselleştirir.

### 3_1_Gaussian_Distribution.py  
Normal dağılımın teorik yapısını ve olasılık yoğunluk fonksiyonunu Python üzerinden örnekleyerek açıklar.

### 4_Normal_Distribution_Random_Numbers.py  
Normal dağılıma uygun rastgele sayı üretimini ve bu sayıların dağılım üzerindeki davranışını gösterir.

### 5_Normal_Distribution_P_120_130_Interval.py  
Normal dağılımda belirli bir aralıkta (örneğin 120–130) değerlerin gelme olasılığının nasıl hesaplandığını öğretir.

### 6_Continuous_Uniform_Distribution_Basic_Sample.py  
Sürekli uniform dağılımın temel mantığını ve bu dağılımdan rastgele sayı üretimini örnekler.

### 7_Standart_Normal_Dist_VS_t_Dist_Sample.py  
Standart normal dağılım ile Student t-dağılımı arasındaki farkları örnekler üzerinden karşılaştırır.

### 10_Binom_Dist_Sample.py  
Binom dağılımını, olasılık hesaplamalarını ve deneysel örneklemeyi gösterir.

### 11_Central_Limit_Theorem_Init_Sample.py  
Merkezi Limit Teoremi’nin temel mantığını simülasyonlar ile açıklayan bir örnektir.

### 12_KSTest_Normal_Distribution_Test.py  
Kolmogorov–Smirnov testi kullanılarak bir veri setinin normal dağılıma uyup uymadığını test eder.

### 13_SWTest_Normal_Distribution_Test.py  
Shapiro–Wilk testi ile normal dağılım varsayımının istatistiksel olarak kontrol edilmesini sağlar.

---

## 📐 Güven Aralığı Hesaplamaları

### 14_Confidence_Interval.py  
Temel güven aralığı hesaplamasını ve istatistiksel yorumlamayı gösterir.

### 15_Confidence_Interval_opt.py  
Güven aralığı hesaplamasının optimize edilmiş ve daha okunabilir bir versiyonunu sunar.

### 16_Confidence_Interval_3.py  
Farklı parametrelerle güven aralığı hesaplamalarını içeren ileri seviye bir örnektir.

---

## 🧹 Veri Ön İşleme – Eksik Veri ve Encoder’lar

### 17_melb_data_csv_Missing_Data_Analysis.py  
Melbourne veri seti üzerinde eksik verilerin analizini yapar ve hangi sütunlarda ne kadar eksik veri olduğunu gösterir.

### 18_melb_data_csv_DropNa_RowOrColumns.py  
Eksik verilerin satır veya sütun bazında veri setinden çıkarılmasını örnekler.

### 19_melb_data_csv_BasicImputation.py  
Eksik verilerin basit yöntemlerle (ortalama, medyan vb.) doldurulmasını gösterir.

### 20_melb_data_csv_SimpleImputer.py  
Scikit-learn SimpleImputer kullanılarak eksik veri tamamlama işlemini öğretir.

### 21_melb_data_csv_IterativeImputer.py  
İteratif imputasyon yöntemiyle eksik verilerin daha gelişmiş şekilde doldurulmasını sağlar.

### 22_melb_data_csv_ManuelCategoryEncoder.py  
Kategorik değişkenlerin manuel olarak sayısallaştırılmasını gösterir.

### 23_melb_data_csv_LabelEncoder.py  
LabelEncoder kullanarak kategorik verilerin sayısal forma dönüştürülmesini öğretir.

### 24_test_csv_LabelEncoderInverseTransform.py  
LabelEncoder ile dönüştürülen verilerin orijinal haline geri çevrilmesini gösterir.

### 25_test_csv_OrdinalEncoder.py  
OrdinalEncoder kullanarak sıralı kategorik verilerin kodlanmasını sağlar.

### 26_test_csv_OneHotEncoder.py  
One-Hot Encoding yöntemiyle kategorik değişkenlerin vektörleştirilmesini öğretir.

### 27_test_csv_OneHotEncoder_TensorflowToCategorical.py  
TensorFlow `to_categorical` fonksiyonu ile One-Hot Encoding örneği sunar.

### 28_test_csv_OneHotEncoder_ManuelwNumpyEye.py  
NumPy `eye` fonksiyonu kullanılarak manuel One-Hot Encoding yapılmasını gösterir.

### 28_test_csv_OneHotEncoder_ManuelwNumpyEyeFunctioned.py  
Manuel One-Hot Encoding işlemini fonksiyonel hale getiren bir örnektir.

### 29_test_csv_DummyVariableEncoding.py  
Dummy variable (kukla değişken) oluşturma mantığını açıklar.

### 30_test_csv_BinaryEncoding.py  
Binary encoding yöntemiyle kategorik verilerin daha kompakt şekilde kodlanmasını öğretir.

---

## 🤖 Makine Öğrenimi ve Sinir Ağları

### 31_Perceptron.py  
Tek katmanlı perceptron algoritmasının temel çalışma mantığını gösterir.

### 32_Keras_train_test_split.py  
Keras projelerinde eğitim ve test verisinin nasıl ayrıldığını öğretir.

### 33_Keras_Neural_Network.py  
Keras kullanılarak basit bir yapay sinir ağı modelinin kurulmasını gösterir.

---

## 🔌 Aktivasyon Fonksiyonları

### 34_ActivationFunc_Relu.py  
ReLU aktivasyon fonksiyonunun matematiksel ve grafiksel gösterimini içerir.

### 35_ActivationFunc_Sigmoid.py  
Sigmoid aktivasyon fonksiyonunun çalışma mantığını açıklar.

### 36_ActivationFunc_Sigmoid_First_Derivative_Graph.py  
Sigmoid fonksiyonunun birinci türevini ve geri yayılım ilişkisini görselleştirir.

### 37_ActivationFunc_HiperbolicTanjant.py  
Tanh aktivasyon fonksiyonunun özelliklerini grafiksel olarak açıklar.

### 38_ActivationFunc_Linear.py  
Lineer aktivasyon fonksiyonunun kullanım alanlarını gösterir.

---

## 💾 Callback, Model Kayıt ve Ölçekleme

### 39_Keras_Neural_Network_Layer_Saving_Loading_HistCallback.py  
Model katmanlarının kaydedilmesi, yüklenmesi ve histogram callback kullanımını öğretir.

### 40_Keras_Neural_Network_Layer_CSVLogger_Callback.py  
Eğitim sürecinin CSV dosyasına loglanmasını sağlar.

### 41_Keras_Neural_Network_Layer_Custom_Callback.py  
Özelleştirilmiş Keras callback yazımını gösterir.

### 42_Keras_Neural_Network_Layer_Lambda_Callback_And_MyLambdaCallback.py  
Lambda callback ve kullanıcı tanımlı callback örneklerini içerir.

---

## 📏 Ölçekleme ve Veri Seti Bazlı Uygulamalar

### 43_Standard_Scaler.py  
StandardScaler kullanarak verilerin normalize edilmesini öğretir.

### 44_Keras_Neural_Network_diabetes_csv_Standard_Scaler.py  
Diyabet veri seti üzerinde StandardScaler ile sinir ağı eğitimi yapar.

### 45_MinMax_Scaler.py  
Min-Max ölçekleme yönteminin temel kullanımını gösterir.

### 46_Keras_Neural_Network_diabetes_csv_MinMax_Scaler.py  
Min-Max ölçekleme ile diyabet veri seti eğitimi örneği sunar.

### 47_Maxabs__Scaler.py  
MaxAbsScaler ile veri ölçekleme mantığını öğretir.

### 48_Keras_Neural_Network_diabetes_csv_Maxabs_Scaler.py  
MaxAbsScaler kullanılarak eğitilen sinir ağı örneğidir.

### 49_Keras_Neural_Network_diabetes_csv_Standard_Scaler_Save.py  
Eğitilmiş model ve scaler’ın diske kaydedilmesini gösterir.

### 50_Keras_Neural_Network_diabetes_csv_Standard_Scaler_Load.py  
Kaydedilmiş model ve scaler’ın tekrar yüklenmesini öğretir.

### 51_Keras_Neural_Network_diabetes_csv_KerasNormalizationLayer_Scaler.py  
Keras Normalization katmanı ile veri ölçeklemeyi gösterir.

### 52_Keras_Neural_Network_diabetes_csv_KerasNormalizationMinMaxLayer_Scaler.py  
Min-Max normalizasyonunun Keras katmanıyla uygulanmasını içerir.

### 53_Keras_Neural_Network_auto-mpg_data_Standard_Scaler_Prediction.py  
Auto MPG veri seti üzerinde regresyon tahmini yapar.

### 54_Keras_Neural_Network_auto-mpg_data_Standard_Scaler_OHE_Prediction.py  
Auto MPG veri setinde One-Hot Encoding ile tahmin uygular.

### 54_Keras_Neural_Network_auto-mpg_data_Standard_Scaler_OHE_Prediction_2LoadH5.py  
Kaydedilmiş H5 model dosyasını yükleyerek tahmin yapar.

### 55_Keras_NN_housing_csv_Standard_Scaler_OHE_Prediction.py  
Konut fiyat tahmini için StandardScaler ve OHE kullanan bir sinir ağı örneğidir.

### 56_Keras_NN_iris_csv_Single_Label_Multiclass_Prediction.py  
Iris veri seti üzerinde tek etiketli çok sınıflı sınıflandırma yapar.

---

## 🧠 Doğal Dil İşleme (NLP)

### 57_Keras_NN_IMDB_csv_Sentiment_Analysis.py  
IMDB film yorumları üzerinde duygu analizi yapan temel bir sinir ağı örneğidir.

### 57_Keras_NN_IMDB_csv_Sentiment_Analysis_CountVectorizer.py  
CountVectorizer kullanılarak metinlerin sayısal vektörlere dönüştürülmesini ve duygu analizini gösterir.

### 57_Keras_NN_keras_IMDB_csv_Sentiment_Analysis_CountVectorizer_2.py  
CountVectorizer ve Keras entegrasyonu ile daha gelişmiş bir NLP örneği sunar.

---

## 🎯 Amaç

Bu repo, **istatistikten derin öğrenmeye geçiş yapanlar** için uçtan uca bir eğitim materyali olarak hazırlanmıştır.
            
