# Laporan Proyek Machine Learning - Guntur Aji Pratama

## Project Overview

Proyek ini merupakan proyek untuk memberikan rekomendasi buku berdasarkan preferensi - preferensi tertentu yang berkaitan dengan pengguna dan buku.

### Latar Belakang

Selama beberapa tahun terakhir, banyak layanan website yang menyediakan sistem rekomendasi, baik dari sisi kesehatan, menu masakan, perangkat elektronik, alat rumah tangga hingga pelajaran. Di sisi pelajaran, salah satu komponen terpenting ialah buku bacaan. Untuk lebih cepat dalam melakukan proses pembelajaran, diperlukan buku bacaaan yang relevan dan sesuai dengan preferensi pengguna, entah dilihat dari sisi kategori buku, penerbit buku, maupun testimoni orang lain terhadap buku tersebut.

Dengan adanya sistem rekomendasi ini, diharapkan dapat membantu mempermudah pengguna mendapat pengetahuan berupa rekomendasi buku - buku yang relevan, berjalan secara otomatis, dalam waktu yang cepat, dan mudah diakses di mana saja.

## Business Understanding

### Problem Statements

- Memudahkan pengetahuan terhadap rekomendasi buku sesuai preferensi pengguna maupun rating buku dengan bantuan model Machine Learning

### Goals

- Membangun satu atau lebih model Machine Learning yang dapat memberikan pengetahuan berupa rekomendasi buku yang sesuai dan relevan terhadap pengguna

### Solution statements

- Menggunakan algoritma Content-Based Filtering dan Collaborative Filtering untuk membuat satu atau lebih model Machine Learning guna mendapat rekomendasi buku yang sesuai dan relevan terhadap preferensi pengguna, melatih dan mengevaluasi model menggunakan metrik evaluasi yang relevan untuk mengetahui kualitas model tersebut

## Data Understanding

Dataset ini dikumpulkan oleh Cai-Nicolas Ziegler dalam waktu 4 minggu (mulai Agustus - September 2004) dari [Komunitas Book-Crossing](https://www.bookcrossing.com) dengan izin dari Ron Hornbaker (CTO of Humankind Systems). Dataset berisi 278.858 informasi mengenai pengguna yang telah dianonimisasi yang memberikan 1.149.780 rating terhadap 271.379 buku. [Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

Lebih detail, terdapat 3 dataset utama, yaitu : dataset Users, Books, dan Ratings. Kita akan jelaskan fitur dari setiap dataset satu per satu.

### Fitur-fitur pada 3 dataset tersebut adalah sebagai berikut :

Dataset Users terdiri dari 278858 baris dan 3 kolom. 3 kolom yang dimaksud antara lain :

- User-ID (identitas pengguna)

- Location (lokasi pengguna)

- Age (umur pengguna)

Dataset Books terdiri dari 271360 baris dan 8 kolom. 8 kolom yang dimaksud antara lain :

- ISBN (identitas buku)

- Book-Title (judul buku)

- Book-Author (penulis buku)

- Year-Of-Publication (tahun publikasi buku)

- Publisher	(penerbit buku)

- Image-URL-S (url foto buku ukuran kecil)

- Image-URL-M (url foto buku ukuran sedang)

- Image-URL-L (url foto buku ukuran besar)

Dataset Ratings terdiri dari 1149780 baris dan 3 kolom. 3 kolom yang dimaksud antara lain :

- User-ID (identitas pengguna)

- ISBN (identitas buku)

- Book-Rating (rating buku)

### Tahapan Data Understanding :

- Memuat Data

- EDA : Deskripsi Data

- EDA : Univariate Analysis (Analisis Terhadap 1 Fitur Dataset Dalam 1 Gambar Visualisasi Data)

  - Data User

  - Data Book

  - Data Rating

## Data Preparation

Secara umum, pada tahap ini akan dilakukan terhadap 2 data untuk tahap Modelling berbasis Content-Based Filtering dan Collaborative Filtering dengan rincian sebagai berikut :

- Merging Data User, Book, dan Rating

  - Penjelasan : Merging merupakan proses penggabungan dua atau lebih data terpisah menjadi satu data baru berdasarkan 1 fitur yang di antara dua data.

  - Alasan Penggunaan : Merging berguna untuk menggabungkan dua atau lebih data terpisah menjadi satu data baru sehingga dapat lebih mudah dalam melakukan persiapan data tanpa khawatir ada data yang tidak tersentuh.

- Menangani Missing Values

  - Penjelasan : Missing values merupakan nilai - nilai kosong dalam sebuah fitur, baris, ataupun kolom. Adanya missing values ini membuat lubang serangkaian data dalam suatu fitur dalam dataset sehingga perlu adanya penanganan khusus terkait missing values ini.

  - Alasan Penggunaan : Penanganan terhadap Missing Values berguna untuk mengurangi ketidakjelasan konteks dari data yang ada dan membantu meningkatkan kualitas model berbasis data tersebut.

- Persiapan Data Untuk Content-Based Filtering

  - Menghapus Data Duplikat dan Mengurutkan Data

    - Penjelasan : Data duplikat merupakan data yang bernilai sama persis dengan data lainnya serta berdampak pada ketidakefektifan model Machine Learning dalam belajar sesuatu yang berulang. Pengurutan data merupakan proses untuk menempatkan data dari urutan terkecil sampai terbesar ataupun sebaliknya.

    - Alasan Penggunaan : Menghapus data duplikat berguna untuk meningkatkan keefektifan kalkulasi data maupun latihan dan evaluasi bagi model Machine Learning. Mengurutkan data berguna untuk meningkatkan akurasi rekomendasi buku karena dataset memiliki beberapa buku dengan data yang sama persis, hanya rating buku tersebut yang berbeda.

- Persiapan Data Untuk Collaborative Filtering

  - Pembagian Data Latih & Validasi

    - Penjelasan : Proses membagi dataset menjadi data latih dan data validasi dimana data latih akan digunakan oleh model Machine Learning untuk latihan dan data validasi sebagai validator bagi performa model Machine Learning yang dihitung melalui metrik evaluasi.

    - Alasan Penggunaan : Pembagian Dataset menjadi data latih yang berguna untuk menjadi masukan bagi model Machine Learning saat latihan dan data validasi yang berguna untuk menjadi validator model Machine Learning, dan menghitung metrik evaluasi, serta memudahkan standarisasi pada data latih dan data validasi.

## Modelling

Pada tahap modelling ini, kita akan menggunakan dua macam algoritma yang berbeda yang dapat diimplementasikan untuk permasalahan rekomendasi buku. Mereka adalah :

- Content-Based Filtering

  - Penjelasan : Content-Based Filtering merupakan algoritma Machine Learning yang cara kerjanya merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu.

  - Library & Fungsi Pendukung :

    1. TF-IDF Vectorizer

       - Penjelasan : TF-IDF Vectorizer merupakan salah satu algoritma untuk menemukan representasi fitur penting dari kategori pada dataset.

       - Alasan Penggunaan : TF-IDF Vectorizer berguna untuk menemukan representasi fitur penting dari setiap penerbit buku (sesuai dataset yang digunakan).

    2. Cosine Similarity

       - Penjelasan : Cosine Similarity merupakan salah satu algoritma untuk menghitung derajat kesamaan antar data berdasarkan kategori pada data tersebut.

       - Alasan Penggunaan : Cosine Similarity berguna untuk menghitung derajat kesamaan (similarity degree) antar judul buku.

  - Kelebihan :

    1. Dapat diinterpretasikan dan dijelaskan

    2. Implementasi algoritma sederhana

    3. Waktu latihan model relatif lebih cepat

    4. Spesifikasi hardware yang relatif lebih rendah untuk komputasi dibanding Deep Learning

  - Kekurangan :

    1. Memerlukan data yang rapi dan urut serta homogenitas yang tinggi pada setiap data yang berbeda untuk mendapat akurasi rekomendasi yang tinggi

    2. Bergantung pada preferensi kategori dari data untuk menjadi label guna memberikan rekomendasi yang sesuai

- Collaborative Filtering

  - Penjelasan : Collaborative Filtering merupakan algoritma Machine Learning yang bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten. Collaborative filtering dibagi lagi menjadi dua kategori, yaitu: model based (metode berbasis model machine learning) dan memory based (metode berbasis memori).

  - Library & Fungsi Pendukung :

    1. TensorFlow

       - Penjelasan : TensorFlow merupakan platform pendukung pengembangan Machine Learning dan Deep Learning yang dapat diintegrasikan dengan bahasa pemrograman Python (bidang Machine Learning murni), Javascript (bidang Web Development), dan Java & Kotlin (bidang Android Development).

       - Alasan Penggunaan : TensorFlow berguna untuk mengembangkan pemodelan Machine Learning & Deep Learning dengan kualitas fungsi - fungsi bawaan TensorFlow yang sangat baik, cepat, dan memiliki dokumentasi kode yang baik.

    2. Keras

       - Penjelasan : Keras merupakan interface library yang bertujuan menyederhanakan implementasi algoritma-algoritma Deep Learning di atas TensorFlow.

       - Alasan Penggunaan : Keras berguna untuk memudahkan pengembangan Machine Learning dan Deep Learning dengan dukungan TensorFlow.

    3. RecommenderNet

       - Penjelasan : RecommenderNet merupakan kelas yang menjadi arsitektur Deep Learning untuk masalah sistem rekomendasi yang dikembangkan oleh Tim Dicoding Indonesia.

       - Alasan Penggunaan : RecommenderNet berguna untuk membangun model Deep Learning terkait masalah sistem rekomendasi yang sudah terbukti memiliki akurasi dan kestabilan prediksi rekomendasi yang tinggi.

      - Optimizer, Loss, dan Metrik : Pada pemodelan berbasis RecommenderNet, Optimizer yang digunakan ialah 'adam' karena Optimizer tersebut merupakan salah satu yang terbaik dan bisa beradaptasi dengan hampir semua studi kasus Deep Learning. Loss Function yang digunakan ialah 'binary_crossentropy' karena Loss Function tersebut sejalan dengan output dari model RecommenderNet yang mengklasifikasikan suatu buku untuk direkomendasikan pada skala 0 hingga 1 dengan 0 berarti tidak direkomendasikan dan 1 berarti direkomendasikan. Loss yang digunakan ialah 'adam'. Metrik yang digunakan ialah 'root_mean_squared_error' karena dapat mengukur tingkat error pada proses latihan model dimana semakin mendekati angka 1 atau lebih menandakan tingkat error semakin tinggi dan mendekati angka 0 menandakan tingkat error semakin rendah dan model dapat berlatih.

  - Kelebihan :

    1. Mudah diinterpretasikan dan dijelaskan

    2. Cocok untuk masalah sistem rekomendasi dan dataset yang besar

    3. Terbukti memiliki akurasi dan kestabilan prediksi rekomendasi yang tinggi

  - Kekurangan :

    1. Membutuhkaan dataset yang besar dan parameter yang jelas

    2. Membutuhkan dukungan TensorFlow dan Keras dalam menjalankannya

    3. Waktu latihan model yang relatif lebih lama

    4. Membutuhkan hardware berspesifikasi tinggi untuk melakukan komputasi dengan lancar dan aman

## Evaluation

Pada tahap evaluasi ini, kita menggunakan pembandingan rekomendasi dan acuan buku pada evaluasi berbasis Content-Based Filtering dan Collaborative Filtering dengan tambahan metrik evaluasi berupa Root Mean Squared Error. Berikut uraiannya :

- Pembadingan rekomendasi dan acuan buku

  - Penjelasan : Ini merupakan proses sederhana untuk mengetahui hubungan nilai rekomendasi dan acuan buku. Karena pada Content-Based Filtering, kita berfokus pada fitur penerbit_buku, maka kita bisa langsung membandingkan hasil rekomendasi buku apakah memiliki nilai penerbit_buku yang sama dengan buku yang dijadikan sebagai acuan. Semisal acuan buku ialah buku yang berjudul 'Cold New Dawn' dan penerbitnya 'harpercollinspublishers', kita bisa mengetahui kualitas model berbasis Content-Based Filtering yang telah dibuat dengan melihat juga penerbit dari buku - buku yang direkomendasikan. Semakin banyak rekomendasi buku yang penerbitnya 'harpercollinspublishers', semakin tinggi kualitas dari model tersebut. Begitu pula dengan model berbasis Collaborative Filtering.

  - Formula : Jumlah nilai penerbit_buku pada rekomendasi buku yang sama dengan nilai penerbit_buku pada acuan buku / Jumlah rekomendasi buku

  - Cara Kerja : Metrik ini bekerja dengan menghitung jumlah nilai penerbit_buku pada rekomendasi buku yang sama dengan nilai penerbit_buku pada acuan buku. Untuk setiap rekomendasi buku, diteliti apakah nilai penerbit_buku di dalam rekomendasi buku tersebut sudah sama dengan nilai penerbit_buku pada acuan buku. Jika sama diberi nilai 1 dan jika tidak diberi nilai 0. Nilai - nilai tersebut kemudian dijumlahkan dan dibagi dengan jumlah rekomendasi buku yang diberikan.

- Root Mean Squared Error

  - Penjelasan : Root Mean Squared Error merupakan metrik untuk mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi terhadap nilai yang diobservasi.

  - Formula di Python : math.sqrt(np.square(np.subtract(y_actual,y_predicted)).mean())

  - Cara Kerja : Metrik ini bekerja dengan menghitung akar dari rata - rata dari selisih setiap nilai rekomendasi terhadap nilai aktual buku.

Kesimpulan :

- Hasil rekomendsai buku pada model berbasis Content-Based Filtering dan Collaborative Filtering menunjukkan hasil yang sangat baik, ditunjukkan dengan nilai penerbit_buku pada rekomendasi buku berbasis Content-Based Filtering bernilai sama semua dengan nilai penerbit_buku pada acuan buku dan 10 besar rekomendasi buku berbasis Collaborative Filtering memiliki nilai penerbit_buku di semua buku yang disukai oleh pengguna.

- Kualitas hasil rekomendasi buku berbasis Content-Based Filtering dan Collaborative Filtering yang sangat baik menandakan pengerjaan proyek Machine Learning ini telah dilakukan dengan baik dan benar

- Perlu peningkatan spesifikasi hardware untuk mendukung pelatihan model berbasis Content-Based Filtering Collaborative Filtering dengan lebih luas dan dalam. Hal ini diharapkan dapat meningkatkan kualitas rekomendasi buku dalam menghadapi beragam permasalahan yang belum diujikan pada penelitian ini.

- Perlu perluasan penerapan algoritma sistem rekomendasi lain untuk mendapat lebih banyak rekomendasi dan perbandingan antar rekomendasi buku.

**catatan : EDA adalah Exploratory Data Analysis**.

Referensi :

[1]   S. Zhang, L. Yao, A. Sun, and Y. Tay, "Deep Learning based Recommender System: A Survey and New Perspectives", _ACM Computing Surveys_, vol. 1, no. 1, 2018.

[2]   https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada

[3]   https://www.dicoding.com

[4]   https://www.datacamp.com

[5]   https://dqlab.id

[6]   https://www.muthu.co
