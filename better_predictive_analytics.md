# Laporan Proyek Machine Learning - Guntur Aji Pratama

## Domain Proyek

Proyek ini merupakan proyek untuk memprediksi kualitas susu berdasarkan parameter - paremeter tertentu dengan 3 klasifikasi kualitas susu, yaitu : rendah, sedang, dan tinggi.

### Latar Belakang

Cara untuk mengidentifikasi kualitas susu memerlukan banyak observasi secara konvensional dengan turun langsung ke lapangan untuk mengecek susu tersebut dan mencatat setiap indikator dengan nilai tertentu untuk mengetahui seberapa baik kualitas susu. Dengan adanya Machine Learning dan dataset berisi hasil observasi kualitas susu yang telah dinilai berdasarkan indikator - indikator tertentu, diharapkan dapat memudahkan para pengecek kualitas susu untuk memprediksi kualitas susu hanya dengan memasukkan nilai - nilai sesuai indikator yang ada.

## Business Understanding

### Problem Statements

- Memudahkan proses identifikasi kualitas susu dengan bantuan hasil prediksi dari model Machine Learning berjenis klasifikasi

- Memudahkan pemahaman alur pengerjaan proyek terkait kepada orang lain

### Goals

- Membangun satu atau lebih model Machine Learning yang dapat memprediksi kualitas susu dengan baik

- Memastikan dokumentasi kode program untuk membuat Machine Learning terkait dapat dipahami orang lain

### Solution statements

- Menggunakan algoritma Logistic Regression, K-Nearest Neighbor, Decision Tree, Random Forest, Support Vector Machine, dan Gradient Boosting untuk membuat satu atau lebih model Machine Learning guna memprediksi kualitas susu dengan baik, melakukan improvement pada baseline model dengan hyperparameter tuning, dan menerapkan metrik evaluasi pada model Machine Learning yang telah dibuat untuk mengetahui kualitas model terkait.
- Menyusun langkah - langkah pengerjaan kode program dari awal hingga akhir secara runtut, menggunakan istilah Bahasa Indonesia yang baik dan benar, dan memberikan komentar terkait penjelasan singkat dari setiap baris kode program untuk mempermudah pemahaman orang lain dalam memahami setiap baris kode program terkait.

## Data Understanding

Dataset ini secara manual dikumpulkan dari observasi. Dataset ini membantu kita untuk membangun model - model Machine Learning guna memprediksi kualitas susu. Dataset ini terdiri dari 7 fitur independen, seperti : pH, Temprature (suhu), Taste (rasa), Odor (bau), Fat (kadar lemak), Turbidity (kekeruhan), dan Color (warna), serta 1 fitur target, yaitu : Grade (kadar kualitas). Dataset telah dilakukan encoding pada fitur kategorik (nilai-nilai pada kolom Taste, Odor, Fat, dan Turbidity) dan tidak ada nilai null pada Dataset. Dataset terdiri dari 1059 baris dan 8 kolom. Berikut link dataset terkait : [2].

### Fitur-fitur pada Milk Quality Prediction dataset adalah sebagai berikut :

- pH : merupakan fitur numerik tentang kadar pH dari susu dengan rentang nilai fitur mulai dari 3 hingga 9.5
- Temprature : merupakan fitur numerik tentang suhu dari susu dengan rentang nilai fitur mulai dari 34 Celcius hingga 90 Celcius
- Taste : merupakan fitur kategorik tentang rasa dari susu dengan nilai 0 dan 1 dimana 0 mewakili rasa yang tidak enak dan 1 mewakili rasa yang enak
- Odor : merupakan fitur kategorik tentang bau dari susu dengan nilai 0 dan 1 dimana 0 mewakili bau tidak enak dan 1 mewakili bau yang enak
- Fat : merupakan fitur kategorik tentang kadar lemak dari susu dengan nilai 0 dan 1 dimana 0 mewakili kadar lemak yang rendah dan 1 mewakili kadar lemak yang tinggi
- Turbidity : merupakan fitur kategorik tentang kekeruhan dari susu dengan nilai 0 dan 1 dimana 0 mewakili kekeruhan yang rendah dan 1 mewakili kekeruhan yang tiggi
- Color : merupakan fitur numerik tentang warna dari susu dengan rentang nilai fitur mulai dari 240 hingga 255
- Grade : merupakan fitur kategorik tentang tingkat kualitas dari susu dengan nilai "low", "medium", dan "high" dimana low mewakili kualitas yang rendah, "medium" mewakili kualitas yang sedang, dan "high" mewakili kualitas yang tinggi dari susu

### Tahapan Data Understanding :

- Memuat Data
- EDA : Deskripsi fitur
    Bentuk dataset : (1059, 8)
    Overview dataset :
    |   |  pH | Temprature | Taste | Odor | Fat | Turbidity | Colour |  Grade |
    |--:|----:|-----------:|------:|-----:|----:|----------:|-------:|-------:|
    | 0 | 6.6 |         35 |     1 |    0 |   1 |         0 |    254 |   high |
    | 1 | 6.6 |         36 |     0 |    1 |   0 |         1 |    253 |   high |
    | 2 | 8.5 |         70 |     1 |    1 |   1 |         1 |    246 |    low |
    | 3 | 9.5 |         34 |     1 |    1 |   0 |         1 |    255 |    low |
    | 4 | 6.6 |         37 |     0 |    0 |   0 |         0 |    255 | medium |
    
    Deskripsi statistik dataset :
    |       |          pH |  Temprature |       Taste |        Odor |         Fat |   Turbidity |      Colour |
    |------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|
    | count | 1059.000000 | 1059.000000 | 1059.000000 | 1059.000000 | 1059.000000 | 1059.000000 | 1059.000000 |
    |  mean |    6.630123 |   44.226629 |    0.546742 |    0.432483 |    0.671388 |    0.491029 |  251.840415 |
    |  std  |    1.399679 |   10.098364 |    0.498046 |    0.495655 |    0.469930 |    0.500156 |    4.307424 |
    |  min  |    3.000000 |   34.000000 |    0.000000 |    0.000000 |    0.000000 |    0.000000 |  240.000000 |
    |  25%  |    6.500000 |   38.000000 |    0.000000 |    0.000000 |    0.000000 |    0.000000 |  250.000000 |
    |  50%  |    6.700000 |   41.000000 |    1.000000 |    0.000000 |    1.000000 |    0.000000 |  255.000000 |
    |  75%  |    6.800000 |   45.000000 |    1.000000 |    1.000000 |    1.000000 |    1.000000 |  255.000000 |
    |  max  |    9.500000 |   90.000000 |    1.000000 |    1.000000 |    1.000000 |    1.000000 |  255.000000 |
  - Klasifikasi Fitur Kategorik, Numerik, dan Target
    Penjelasan : Proses ini dilakukan dengan mengklasifikasikan nama - nama fitur yang bersifat kategorik seperti : 'Taste', 'Odor', 'Fat', dan 'Turbidity', numerik seperti : 'pH', 'Temprature', dan 'Colour', serta target seperti : 'Grade' untuk kemudian dirubah ke dalam istilah Bahasa Indonesia agar lebih mudah dimengerti.
- EDA : Menangani Missing Values dan Outliers Pada Dataset
  - Menangani Missing Values
    - Penjelasan : Missing values merupakan nilai - nilai kosong dalam sebuah fitur, baris, ataupun kolom. Adanya missing values ini membuat lubang serangkaian data dalam suatu fitur dalam dataset sehingga perlu adanya penanganan khusus terkait missing values ini.
    
    |             pH | 0 |
    |---------------:|--:|
    |      Suhu      | 0 |
    |      Rasa      | 0 |
    |       Bau      | 0 |
    |   Kadar Lemak  | 0 |
    |    Kekeruhan   | 0 |
    |      Warna     | 0 |
    | Kadar Kualitas | 0 |
    |  dtype: int64  |   |
  - Menangani Outliers
    - Penjelasan : Outliers merupakan titik - titik data yang terpaut jauh dari titik data lainnya. Adanya outliers ini membuat berpotensi menyebabkan performa model menjadi overfitting sehingga perlu adanya penanganan khusus terkait outliers ini. Adapun metode yang digunakan untuk mengatasi outliers pada dataset proyek ini ialah IQR (Inter Quartile Range) dimana IQR ini selisih antara persentil ke-75 (kuartil atas) dan persentil ke-25 (kuartil bawah). IQR ini dapat digunakan untuk membersihkan data - data pada dataset yang rentang nilainya kurang atau lebih dari rentang nilai pada IQR sehingga mengurangi overfitting pada model Machine Learning.
    
    Bentuk dataset setelah pembersihan outliers : (648, 8)
- EDA : Univariate Analysis (Analisis Terhadap 1 Fitur Dataset Dalam 1 Gambar Visualisasi Data)
  - Fitur Kategorik
    Terlihat bahwa terdapat 2 jenis data identik pada seluruh fitur kategorik, yaitu 0 (mewakili kualitas buruk, rendah, kecil dari susu) dan 1 (mewakili kualitas baik, tinggi, besar dari susu) dengan rincian :

    -   Pada fitur Rasa, persentase 0 ialah 47.7% dan 1 ialah 52.3% (jumlah nilai 0 < 1)
    -   Pada fitur Bau, persentase 0 ialah 61.9% dan 1 ialah 38.1% (jumlah nilai 0 > 1)
    -   Pada fitur Kadar Lemak, persentase 0 ialah 34.7% dan 1 ialah 65.3% (jumlah nilai 0 < 1)
    -   Pada fitur Kekeruhan, persentase 0 ialah 67.3% dan 1 ialah 32.7% (jumlah nilai 0 > 1)
  - Fitur Numerik
    Distribusi pada fitur pH dan Warna terlihat memiliki dominasi di beberapa daerah saja, sedangkan pada fitur Suhu terdapat pola 'right-side'.
- EDA : Multivariate Analysis (Analisis Terhadap 2 atau Lebih Fitur Dataset Dalam 1 Gambar Visualisasi Data)
  - Encoding Fitur Target (Seharusnya Dilakukan Pada Tahap Data Preparation, Namun Kesulitan Saat Implementasi Beberapa Gambar Visualisasi yang Harus Melibatkan Fitur Target Sehingga Proses Ini Dilakukan Pada Tahap Data Understanding)
    - Penjelasan : Encoding merupakan proses merubah nilai teks pada fitur target menjadi angka agar lebih mudah dimengerti komputer dan memudahkan univariate analysis.
    ```
    # Merubah nilai teks pada fitur target menjadi angka agar lebih mudah dimengerti komputer dan memudahkan univariate analysis
    df['Kadar Kualitas'].replace({
      'low':0,
      'medium':1,
      'high':2
    }, inplace=True)
    ```
  - Fitur Kategorik
    Terlihat bahwa mayoritas nilai pada fitur kategorik berada pada kadar kualitas susu yang medium dan high.
  - Fitur Numerik
    Terlihat bahwa korelasi fitur - fitur dalam dataset terhadap fitur Kadar Kualitas cukup terlihat dengan rincian :
    - pH : 0.23
    - Suhu : -0.3
    - Rasa : 0.24
    - Bau : 0.54
    - Kadar Lemak : 0.51
    - Kekeruhan : 0.41
    - Warna : 0.16

**catatan : EDA adalah Exploratory Data Analysis**.

## Data Preparation

- Pembagian Data Latih & Validasi
  - Penjelasan : Proses membagi dataset menjadi data latih dan data validasi dimana data latih akan digunakan oleh model Machine Learning untuk latihan dan data validasi sebagai validator bagi performa model Machine Learning yang dihitung melalui metrik evaluasi.
  - Alasan Penggunaan : Pembagian Dataset menjadi data latih yang berguna untuk menjadi masukan bagi model Machine Learning saat latihan dan data validasi yang berguna untuk menjadi validator model Machine Learning, dan menghitung metrik evaluasi, serta memudahkan standarisasi pada data latih dan data validasi.
  ```
  # Mengimpor modul train_test_split dari library sklearn untuk pendukung pembagian data
  from sklearn.model_selection import train_test_split
    
  # Mendefinisikan gabungan fitur kategorik & numerik sebagai masukan bagi model Machine Learning
  X = df.iloc[:, :df.columns.get_loc(fitur_target)]
    
  # Mendefinisikan target sebagai hasil masukan bagi model Machine Learning
  y = df[fitur_target]
    
  # Membagi masukan (X) dan hasil masukan (y) dengan persentase data validasi 20%
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
  ```
- Menyeimbangkan Jumlah Data Target Menggunakan Metode SMOTE
  - Penjelasan : Metode Synthetic Minority Over-sampling Technique (SMOTE) merupakan metode yang populer diterapkan dalam rangka menangani ketidak seimbangan kelas. Teknik ini mensintesis sampel baru dari kelas minoritas untuk menyeimbangkan dataset dengan cara sampling ulang sampel kelas minoritas.
  - Alasan Penggunaan : Metode SMOTE berguna untuk menyeimbangkan jumlah data target yang mana berpotensi untuk meningkatkan performa model Machine Learning.
  ```
  from imblearn.over_sampling import SMOTE
  print(y_train.value_counts())#Previous original class distribution
  X_train, y_train = SMOTE().fit_resample(X_train, y_train)
  print(y_train.value_counts()) #Preview synthetic sample class distribution
  ```
- Standarisasi pada data latih
  - Penjelasan : StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. [3]
  - Alasan Penggunaan : Standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma sehingga kemungkinan besar dapat meningkatkan performa model Machine Learning saat latihan maupun validasi.
  ```
  # Mengimpor modul StandardScaler dari library sklearn untuk pendukung standarisasi data
  from sklearn.preprocessing import StandardScaler

  # Mendefinisikan fungsi standarisasi dari StandardScaler
  scaler = StandardScaler()

  # Melakukan standarisasi pada data latih (X_train) pada fitur numerik
  X_train[fitur_numerik] = scaler.fit_transform(X_train[fitur_numerik])

  # Menampilkan overview data latih (X_train)
  X_train.head()
  ```
  |   |        pH |      Suhu | Rasa | Bau | Kadar Lemak | Kekeruhan |     Warna |
  |--:|----------:|----------:|-----:|----:|------------:|----------:|----------:|
  | 0 | -1.305146 | -1.131415 |    0 |   0 |           0 |         0 |  0.837644 |
  | 1 | -0.445060 | -0.954652 |    0 |   0 |           0 |         0 |  0.837644 |
  | 2 | -0.445060 | -1.131415 |    1 |   0 |           1 |         0 |  0.837644 |
  | 3 | -1.305146 | -0.954652 |    1 |   0 |           1 |         0 |  0.837644 |
  | 4 | -1.305146 | -1.308178 |    0 |   0 |           0 |         0 | -1.089793 |
## Modeling

Pada tahap modelling ini, kita akan menggunakan enam macam algoritma yang berbeda yang dapat diimplementasikan untuk permasalahan klasifikasi. Mereka adalah :
- Logistic Regression [4]
  - Penjelasan : Logistic Regression merupakan algoritma Machine Learning untuk menemukan hubungan antara dua faktor data dan menggunakan hubungan ini untuk memprediksi nilai dari salah satu faktor tersebut berdasarkan faktor yang lain.
  - Parameter pada Hyperparameter Tuning :
    1. C merupakan kebalikan dari regularisasi dimana parameter ini bertipe bilangan desimal positif dan semakin kecil nilainya maka semakin kuat regularisasinya. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [10^-2, 10^-1, 10^0, 10^1, 10^2].
    2. penalty merupakan parameter terkait dengan nilai koefisien ke kesalahan hipotesis. Semakin akurat model dengan nilai koefisien yang ekstrim akan dikenakan sanksi lebih banyak, semakin kurang akurat model dengan nilai yang lebih konservatif akan dikenakan sanksi lebih sedikit. Nilai parameter ini pada Hyperparameter Tuning ialah 'l2'.
  - Kelebihan :
    1. Dapat diinterpretasikan dan dijelaskan
    2. Mengurangi overfitting saat menggunakan fitur regularisasi
    3. Dapat digunakan untuk memprediksi beberapa kelas (klasifikasi)
  - Kekurangan :
    1. Mengasumsikan hubungan linear antara fitur masukan dan target dari model
    2. Bisa mengalami overfitting terhadap data yang berdimensi besar
- K-Nearest Neighbor [4]
  - Penjelasan : K-Nearest Neighbor merupakan algoritma Machine Learning bertipe klasifikasi yang paling sederhana dalam mengklasifikasikan data kedalam sebuah label. Metode ini mudah dipahami dibandingkan metode lain karena mengklasifikasikan berdasarkan jarak terdekat dengan objek lain (tetangga).
  - Parameter pada Hyperparameter Tuning :
    1. n_neighbors merupakan nilai berupa jumlah data tetangga terdekat dengan suatu data. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [2, 5, 10, 25, 50].
  - Kelebihan :
    1. Dapat adaptasi dengan dataset yang besar
    2. Dapat diinterpretasikan dan dijelaskan
    3. Hasil prediksi digolongkan ke dalam klaster - klaster yang rapat
  - Kekurangan :
    1. Membutuhkan ketepatan masukan jumlah klaster dalam dataset
    2. Memiliki masalah dengan keragaman klaster dan kerapatan nilai pada dataset
- Decision Tree [4]
  - Penjelasan : Decision Tree merupakan algoritma Machine Learning untuk membantu proses pengambilan keputusan. Disebut sebagai “tree” karena struktur ini menyerupai sebuah pohon lengkap dengan akar, batang, dan percabangannya.
  - Parameter pada Hyperparameter Tuning :
    1. max_depth merupakan nilai kedalaman maksimal dari algoritma. Semakin besar nilai max_depth, semakin banyak cabang dari arsitektur algoritma dan semakin panjang arsitektur algoritma. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [5, 10, 25].
    2. min_samples_split merupakan nilai minimal yang dibutuhkan untuk membagi sampel - sampel dalam node internal dari arsitektur algoritma. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [5, 10, 25].
  - Kelebihan :
    1. Dapat diinterpretasikan dan dijelaskan
    2. Bisa menangani nilai - nilai null
  - Kekurangan :
    1. Berpotensi mudah overfitting
    2. Sensitif terhadap outliers
- Random Forest [4]
  - Penjelasan : Random Forest merupakan algoritma Machine Learning versi Bagging dari Decision Tree dimana arsitektur algoritma ini terdiri dari beberapa Decision Tree yang saling melakukan komputasinya masing - masing dengan pembagian data dan fitur secara acak dan kemudian mengumpulkan seluruh hasil komputasi akhirnya untuk dipilih hasil akhir dengan skor secara tertinggi.
  - Parameter pada Hyperparameter Tuning :
    1. n_estimators merupakan nilai berupa jumlah pohon atau 'tree' yang ingin dibuat untuk menjalankan komputasi independen sebelum dilakukan voting berupa hasil komputasi akhir dengan skor tertinggi. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [10, 50, 100, 250].
    2. max_depth merupakan nilai kedalaman maksimal dari algoritma. Semakin besar nilai max_depth, semakin banyak cabang dari arsitektur algoritma dan semakin panjang arsitektur algoritma. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [5, 10, 20].
  - Kelebihan :
    1. Mengurangi overfitting
    2. Memiliki nilai akurasi model yang lebih tinggi dari algoritma - algoritma yang lain
  - Kekurangan :
    1. Kompleksitas training model bisa saja sangat tinggi
    2. Agak susah diinterpretasikan dan dijelaskan
- Support Vector Machine [5]
  - Penjelasan : Support Vector Machine merupakan algoritma Machine Learning yang menggunakan ruang hipotesis yang berupa fungsi-fungsi linear didalam sebuah fitur yang memiliki dimensi tinggi dan dilatih dengan menggunakan algoritma pembelajaran berdasarkan teori optimasi.
  - Parameter pada Hyperparameter Tuning :
    1. C merupakan kebalikan dari regularisasi dimana parameter ini bertipe bilangan desimal positif dan semakin kecil nilainya maka semakin kuat regularisasinya. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [10^-2, 10^-1, 10^0, 10^1, 10^2].
  - Kelebihan :
    1. Pengklasifikasi SVM menawarkan akurasi tinggi dan bekerja dengan baik dengan ruang dimensi tinggi.
  - Kekurangan :
    1. Mereka memiliki waktu pelatihan yang tinggi sehingga dalam praktiknya tidak cocok untuk kumpulan data yang besar.
    2. Pengklasifikasi SVM tidak berfungsi dengan baik dengan kelas yang tumpang tindih.
- Gradient Boosting [4]
  - Penjelasan : Gradient Boosting merupakan algoritma Machine Learning versi Boosting dari Decision Tree dimana arsitektur algoritma ini terdiri dari beberapa Decision Tree yang saling melakukan komputasinya masing - masing dengan pembagian data dan fitur secara terfokus serta terus menerus meningkatkan akurasi dari Decision Tree pendahulunya untuk membuat beberapa Decision Tree baru yang telah membawa pengetahuan dari pendahulunya.
  - Parameter pada Hyperparameter Tuning :
    1. n_estimators merupakan nilai berupa jumlah pohon atau 'tree' yang ingin dibuat untuk menjalankan komputasi independen sebelum dilakukan voting berupa hasil komputasi akhir dengan skor tertinggi. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [10, 50, 100, 250].
    2. max_depth merupakan nilai kedalaman maksimal dari algoritma. Semakin besar nilai max_depth, semakin banyak cabang dari arsitektur algoritma dan semakin panjang arsitektur algoritma. Rentang nilai parameter ini pada Hyperparameter Tuning ialah [5, 10, 20].
  - Kelebihan :
    1. Memiliki nilai akurasi yang lebih tinggi dari algoritma - algoritma yang lain
    2. Bisa menangani hubungan antar fitur dalam dataset yang bersifat kolinearitas
    3. Bisa menangani hubungan tak linear antar fitur dalam dataset
  - Kekurangan :
    1. Sensitif terhadap outliers sehingga berpotensi mudah mengalami overfitting
    2. Biaya komputasi mahal dan kompleksitas algoritma yang tinggi

Selanjutnya akan dilakukan hyperparameter tuning dan training terhadap model - model berbasis keenam algoritma tersebut dengan rincian tahap :
- Mengimport keenam algoritma terkait
- Mengimpor fungsi Pipeline untuk melakukan urutan transformasi yang berbeda dari dataset untuk mendapat model dan parameter akhir dan GridSearchCV untuk mengakomodasi sarana guna training beberapa model dan parameter yang berbeda dengan bantuan fungsi Pipeline
- Mendefinisikan parameter - parameter yang relevan sesuai untuk setiap model dari keenam algoritma terkait
- Menampung seluruh definisi model dan parameter ke dalam fungsi Pipeline
- Training model - model dengan bantuan fungsi Pipeline
```
# Mengimpor beberapa modul algoritma Machine Learning library sklearn untuk basis model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Mengimpor model Pipeline & GridSearchCV dari library sklearn untuk pendukung pembuatan pipeline hyperparameter & models
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Initialze the estimators
model1 = RandomForestClassifier(random_state=123)
model2 = SVC(random_state=123)
model3 = LogisticRegression(random_state=123)
model4 = DecisionTreeClassifier(random_state=123)
model5 = KNeighborsClassifier()
model6 = GradientBoostingClassifier(random_state=123)

# Mendefinisikan beberapa hyperparameter yang relevan untuk setiap model
param1 = {
    'classifier__n_estimators':[10,50,100,250],
    'classifier__max_depth':[5,10,20],
    'classifier':[model1]
}
param2 = {
    'classifier__C':[10**-2, 10**-1, 10**0, 10**1, 10**2],
    'classifier':[model2]
}
param3 = {
    'classifier__C':[10**-2, 10**-1, 10**0, 10**1, 10**2],
    'classifier__penalty':['l2'],
    'classifier':[model3]
}
param4 = {
    'classifier__max_depth':[5,10,25],
    'classifier__min_samples_split':[2,5,10],
    'classifier':[model4]
}
param5 = {
    'classifier__n_neighbors':[2,5,10,25,50],
    'classifier':[model5]
}
param6 = {
    'classifier__n_estimators':[10,50,100,250],
    'classifier__max_depth':[5,10,20],
    'classifier':[model6]
}

# Membuat pipeline untuk mendata seluruh model dan parameter terkait
pipeline = Pipeline([('classifier', model1)])
params = [param1, param2, param3, param4, param5, param6]

# Melakukan training pada model - model & hyperparameter tuning dengan fungsi gridsearchcv
# Menggunakan 'accuracy' sebagai metrik evaluasi training model
gs = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='accuracy').fit(X_train, y_train)
```
```
# Mendefinisikan laporan hasil training & hyperparameter tuning terhadap beberapa model
gs_results = pd.DataFrame(gs.cv_results_)[[
    'param_classifier',
    'params',
    'mean_test_score',
    'rank_test_score'
]].sort_values(by='rank_test_score')

# Menampilkan overview data dari laporan hasil training & hyperparameter tuning terhadap beberapa model
gs_results.head(10)
```
|    |                                  param_classifier |                                            params | mean_test_score | rank_test_score |
|---:|--------------------------------------------------:|--------------------------------------------------:|----------------:|----------------:|
| 38 | GradientBoostingClassifier(max_depth=5, random... | {'classifier': GradientBoostingClassifier(max_... |        1.000000 |               1 |
| 10 |          RandomForestClassifier(random_state=123) | {'classifier': RandomForestClassifier(random_s... |        0.998775 |               2 |
| 37 | GradientBoostingClassifier(max_depth=5, random... | {'classifier': GradientBoostingClassifier(max_... |        0.998775 |               2 |
| 39 | GradientBoostingClassifier(max_depth=5, random... | {'classifier': GradientBoostingClassifier(max_... |        0.998775 |               2 |
|  4 |          RandomForestClassifier(random_state=123) | {'classifier': RandomForestClassifier(random_s... |        0.998775 |               2 |
|  5 |          RandomForestClassifier(random_state=123) | {'classifier': RandomForestClassifier(random_s... |        0.998775 |               2 |
|  6 |          RandomForestClassifier(random_state=123) | {'classifier': RandomForestClassifier(random_s... |        0.998775 |               2 |
|  7 |          RandomForestClassifier(random_state=123) | {'classifier': RandomForestClassifier(random_s... |        0.998775 |               2 |
|  8 |          RandomForestClassifier(random_state=123) | {'classifier': RandomForestClassifier(random_s... |        0.998775 |               2 |
|  9 |          RandomForestClassifier(random_state=123) | {'classifier': RandomForestClassifier(random_s... |        0.998775 |               2 |
## Evaluation

Pada tahap evaluasi ini, kita menggunakan fungsi classification_report dan confusion_matrix dari library sklearn yang memiliki beberapa metrik evaluasi, antara lain :
- Precision [6]
  - Penjelasan : Persentase jumlah prediksi label yang tepat dengan nilai aktualnya.
  - Formula : TP / (TP + FP)
  - Cara Kerja : Metrik ini bekerja dengan menghitung banyaknya jumlah prediksi yang sesuai dengan nilai aktualnya pada dataset. Presisi di setiap label didefinisikan sebagai rasio dari nilai True Positive dengan jumlah total True Positive & False Positive.
- Recall [6]
  - Penjelasan : Persentase dari nilai bersifat positif (True Positive & False Negative) dari prediksi model.
  - Formula : TP / (TP + FN)
  - Cara Kerja : Metrik ini bekerja dengan menghitung semua nilai yang bersifat positif dari prediksi model. Recall di setiap label didefinisikan sebagai rasio dari nilai True Positive dengan jumlah total True Positive & False Negative.
- F1-Score [7]
  - Penjelasan : Rerata harmonic dari precision dan recall.
  - Formula : 2 * (Recall * Precision) / (Recall + Precision)
  - Cara Kerja : Metrik ini bekerja dengan menghitung perbandingan nilai presisi dan recall dengan minimal skor 0 dan maksimal skor 1. Secara umum, metrik ini lebih kecil dibanding metrik akurasi karena metrik F1-Score mengaitkan presisi dan recall ke dalam komputasinya. 
- Accuracy [8]
  - Penjelasan : Persentase jumlah data yang diprediksi secara benar terhadap jumlah keseluruhan data.
  - Formula : (TP + TN) / (TP + TN + FP + FN)
  - Cara Kerja : Metrik ini bekerja dengan menghitung jumlah prediksi di setiap label secara tepat kemudian dibagi dengan jumlah total prediksi yang dilakukan.
```
# Mengimpor seluruh model dari library sklearn
from sklearn.metrics import *

# Menampilkan parameter terbaik dari hasil training & hyperparameter tuning terhadap beberapa model
print(gs.best_params_)

# Menampilkan skor akurasi tertinggi dari training model
print(gs.best_score_)

# Melakukan standarisasi pada data validasi (X_val) pada fitur numerik
X_val[fitur_numerik] = scaler.transform(X_val[fitur_numerik])

# Menampilkan skor akurasi pada data validasi dari model terbaik
print("Validation Score:",gs.score(X_val, y_val))
```
Output :
```
{'classifier': GradientBoostingClassifier(max_depth=5, random_state=123), 'classifier__max_depth': 5, 'classifier__n_estimators': 100}
1.0
Validation Score: 0.9923076923076923
```
```
# Memanggil model terbaik hasil training & hyperparameter tuning
algorithm = gs.best_params_['classifier']

# Training model terbaik
algorithm.fit(X_train, y_train)

# Memprediksi data validasi
val_pred = algorithm.predict(X_val)

# Melakukan visualisasi performa model dengan fungsi vis_eval
vis_eval(y_val, val_pred)
```
| Classification Report |           |        |          |         |
|----------------------:|----------:|-------:|---------:|--------:|
|                       | precision | recall | f1-score | support |
|           0           |      1.00 |   1.00 |     1.00 |      10 |
|           1           |      0.99 |   1.00 |     0.99 |      70 |
|           2           |      1.00 |   0.98 |     0.99 |      50 |
|        accuracy       |           |        |     0.99 |     130 |
|       macro avg       |      1.00 |   0.99 |     0.99 |     130 |
|      weighted avg     |      0.99 |   0.99 |     0.99 |     130 |

| Confusion Matrix |        |       |                 |        |
|------------------|--------|-------|-----------------|--------|
|                  | Buruk  | 10    | 0               | 0      |
| True label       | Sedang | 0     | 70              | 0      |
|                  | Tinggi | 0     | 1               | 49     |
|                  |        | Buruk | Sedang          | Tinggi |
|                  |        |       | Predicted label |        |

Kesimpulan :
- Berikut 10 besar model terbaik hasil hyperparameter tuning dengan urutan pertama diraih oleh model dengan algoritma Gradient Boosting. Adapun penyebab model berbasis Gradient Boosting menjadi model terbaik karena algoritmanya yang telah terbukti lebih unggul dibanding algoritma - algoritma, terutama dalam kasus klasifikasi kualitas susu pada proyek ini. Penyebab lain ialah karena Gradient Boosting menggunakan lebih banyak sumber daya untuk melakukan komputasi yang kompleks sehingga dapat meraih hasil akurasi yang sangat tinggi, bahkan sempurna
- Akurasi prediksi model terbaik mencapai 99.2% dengan hanya mengalami 1 kali prediksi salah yang menandakan pengerjaan proyek Machine Learning ini telah dilakukan dengan baik dan benar
- Perlu peningkatan jumlah dataset agar dapat meraih nilai korelasi antar fitur dengan fitur target yang lebih tinggi karena nilai korelasi tertinggi antara fitur masukan dengan fitur target hanya mencapai 0.54 saja
- Penggunaan keenam algoritma Machine Learning cukup relevan karena dari hasil hyperparameter tuning, mayoritas model dari keenam algoritma tersebut bersaing cukup ketat untuk mendapat nilai prediksi terbaik

Referensi :

[1]   M. Frizzarin, I.C. Gormley, D.P. Berry, T.B. Murphy, A. Casa, A. Lynch, and S. McParland, "Predicting cow milk quality traits from routinely available milk spectra using statistical machine learning methods", _Journal of Dairy Science_, vol. 104, no. 7, pp. 7438-7447, 2021.

[2]   https://www.kaggle.com/datasets/cpluzshrijayan/milkquality

[3]   https://www.dicoding.com

[4]   https://www.datacamp.com

[5]   https://dqlab.id

[6]   https://www.muthu.co

[7]   https://stevkarta.medium.com/membicarakan-precision-recall-dan-f1-score-e96d81910354

[8]   https://medium.com/@pararawendy19/memahami-metrik-pada-pemodelan-klasifikasi-29cd5b738ee7
