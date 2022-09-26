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

Dataset ini dikumpulkan oleh Cai-Nicolas Ziegler dalam waktu 4 minggu (mulai Agustus - September 2004) dari [Komunitas Book-Crossing](https://www.bookcrossing.com) dengan izin dari Ron Hornbaker (CTO of Humankind Systems). Dataset berisi 278.858 informasi mengenai pengguna yang telah dianonimisasi yang memberikan 1.149.780 rating terhadap 271.379 buku. Berikut link dataset terkait : [book recommendation dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

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

  Bentuk Data User   : (278858, 3)
  
  Bentuk Data Book   : (271360, 8)
  
  Bentuk Data Rating : (1149780, 3)
  
  Informasi Data User :
  
  ```
   #   Column    Non-Null Count   Dtype  
  ---  ------    --------------   -----  
   0   User-ID   278858 non-null  int64  
   1   Location  278858 non-null  object 
   2   Age       168096 non-null  float64
  dtypes: float64(1), int64(1), object(1)
  memory usage: 6.4+ MB
  ```
  
  Terlihat bahwa data User terdiri dari 278858 baris dan 3 kolom. 3 kolom yang dimaksud antara lain : User-ID (identitas pengguna), Location (lokasi pengguna), dan Age (umur pengguna).
  
  Informasi Data Book :
  
  ```
   #   Column               Non-Null Count   Dtype 
  ---  ------               --------------   ----- 
   0   ISBN                 271360 non-null  object
   1   Book-Title           271360 non-null  object
   2   Book-Author          271359 non-null  object
   3   Year-Of-Publication  271360 non-null  object
   4   Publisher            271358 non-null  object
   5   Image-URL-S          271360 non-null  object
   6   Image-URL-M          271360 non-null  object
   7   Image-URL-L          271357 non-null  object
  dtypes: object(8)
  memory usage: 16.6+ MB
  ```
  
  Terlihat bahwa data Book terdiri dari 271360 baris dan 8 kolom. 8 kolom yang dimaksud antara lain : ISBN (identitas buku),	Book-Title (judul buku), Book-Author (penulis buku), Year-Of-Publication (tahun publikasi buku),	Publisher	(penerbit buku), Image-URL-S (url foto buku ukuran kecil),	Image-URL-M (url foto buku ukuran sedang), dan Image-URL-L (url foto buku ukuran besar).
  
  Informasi Data Rating :
  
  ```
   #   Column       Non-Null Count    Dtype 
  ---  ------       --------------    ----- 
   0   User-ID      1149780 non-null  int64 
   1   ISBN         1149780 non-null  object
   2   Book-Rating  1149780 non-null  int64 
  dtypes: int64(2), object(1)
  memory usage: 26.3+ MB
  ```
  
  Terlihat bahwa data Rating terdiri dari 1149780 baris dan 3 kolom. 3 kolom yang dimaksud antara lain : User-ID (identitas pengguna), ISBN (identitas buku), Book-Rating (rating buku).

- EDA : Univariate Analysis (Analisis Terhadap 1 Fitur Dataset Dalam 1 Gambar Visualisasi Data)

  - Data User

    <div align="center">
  
    |   | User-ID |                           Location |  Age |
    |--:|--------:|-----------------------------------:|-----:|
    | 0 |       1 |                 nyc, new york, usa |  NaN |
    | 1 |       2 |          stockton, california, usa | 18.0 |
    | 2 |       3 |    moscow, yukon territory, russia |  NaN |
    | 3 |       4 |          porto, v.n.gaia, portugal | 17.0 |
    | 4 |       5 | farnborough, hants, united kingdom |  NaN |
  
    </div>
    
    Terlihat bahwa dari 3 kolom yang ada, kolom Location dan Age tidak terlalu mempengaruhi tingkat presisi sistem rekomendasi yang akan dibuat mengingat mereka hanya merupakan distribusi demografi dari setiap pengguna. Maka dari itu, kita hanya akan fokus pada kolom User-ID.
    
    Jumlah User-ID unik : 278858
    
    Terlihat bahwa jumlah pengguna unik sama dengan jumlah baris pada data User yaitu 278858 sehingga dapat dipastikan tidak ada duplikat pada data User.

  - Data Book

    <div align="center">
  
    |   |       ISBN |                                        Book-Title |          Book-Author | Year-Of-Publication |                  Publisher |                                       Image-URL-S |                                       Image-URL-M |                                       Image-URL-L |
    |--:|-----------:|--------------------------------------------------:|---------------------:|--------------------:|---------------------------:|--------------------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|
    | 0 | 0195153448 |                               Classical Mythology |   Mark P. O. Morford |                2002 |    Oxford University Press | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... |
    | 1 | 0002005018 |                                      Clara Callan | Richard Bruce Wright |                2001 |      HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... |
    | 2 | 0060973129 |                              Decision in Normandy |         Carlo D'Este |                1991 |            HarperPerennial | http://images.amazon.com/images/P/0060973129.0... | http://images.amazon.com/images/P/0060973129.0... | http://images.amazon.com/images/P/0060973129.0... |
    | 3 | 0374157065 | Flu: The Story of the Great Influenza Pandemic... |     Gina Bari Kolata |                1999 |       Farrar Straus Giroux | http://images.amazon.com/images/P/0374157065.0... | http://images.amazon.com/images/P/0374157065.0... | http://images.amazon.com/images/P/0374157065.0... |
    | 4 | 0393045218 |                            The Mummies of Urumchi |      E. J. W. Barber |                1999 | W. W. Norton &amp; Company | http://images.amazon.com/images/P/0393045218.0... | http://images.amazon.com/images/P/0393045218.0... | http://images.amazon.com/images/P/0393045218.0... |
  
    </div>
    
    Terlihat bahwa dari 8 kolom yang ada, kolom Book-Title, Year-Of-Publication, Image-URL-S, Image-URL-M, dan	Image-URL-L tidak terlalu mempengaruhi tingkat presisi sistem rekomendasi yang akan dibuat mengingat mereka hanya merupakan informasi pelengkap dari setiap buku. Maka dari itu, kita hanya akan fokus pada kolom ISBN sebagai penanda identitas buku, Book-Author sebagai penanda kategori buku dari sisi penulis, dan Publisher sebagai penanda kategori buku dari sisi penerbit.
    
    Jumlah ISBN unik        : 271360
    
    Jumlah Book-Author unik : 271360
    
    Jumlah Publisher unik   : 16808
    
    Ternyata, baik jumlah buku dan penulis unik sama dengan jumlah baris pada data Book yaitu 271360 sehingga asumsi bahwa kolom Book-Author merupakan penanda kategori buku dari sisi penulis merupakan hal yang salah dan tidak berpengaruh terhadap tingkat presisi sistem rekomendasi yang akan dibuat. Adapun jumlah penerbit unik adalah 16808 sehingga kita bisa menggunakan kolom Publisher sebagai salah satu komponen yang akan memberi pengaruh terhadap tingkat presisi sistem rekomendasi yang akan dibuat.

  - Data Rating

    <div align="center">
  
    |   | User-ID |       ISBN | Book-Rating |
    |--:|--------:|-----------:|------------:|
    | 0 |  276725 | 034545104X |           0 |
    | 1 |  276726 | 0155061224 |           5 |
    | 2 |  276727 | 0446520802 |           0 |
    | 3 |  276729 | 052165615X |           3 |
    | 4 |  276729 | 0521795028 |           6 |
  
    </div>
    
    Terlihat bahwa dari 3 kolom yang ada, semua kolom berpengaruh terhadap tingkat presisi dari sistem rekomendasi yang akan dibuat, sama seperti kolom - kolom yang ada pada data User dan Book.
    
    Jumlah User-ID unik     : 105283
    
    Jumlah ISBN unik        : 340556
    
    Jumlah Book-Rating unik : 11
    
    Terlihat bahwa jumlah pengguna unik yang memberikan rating terhadap buku yaitu 105283 lebih kecil dari jumlah total pengguna unik yaitu 278858. Terlihat juga bahwa jumlah buku unik yang diberikan rating yaitu 340556 lebih besar dari jumlah total buku unik yaitu 271360, sementara untuk jumlah rating unik seperti yang telah diperkirakan. Ini berarti tidak semua pengguna memberikan rating pada buku dan ada beberapa buku tak dikenal yang diberi rating oleh pengguna.

## Data Preparation

Secara umum, pada tahap ini akan dilakukan terhadap 2 data untuk tahap Modelling berbasis Content-Based Filtering dan Collaborative Filtering dengan rincian sebagai berikut :

- Merging Data User, Book, dan Rating

  - Penjelasan : Merging merupakan proses penggabungan dua atau lebih data terpisah menjadi satu data baru berdasarkan 1 fitur yang di antara dua data.

  - Alasan Penggunaan : Merging berguna untuk menggabungkan dua atau lebih data terpisah menjadi satu data baru sehingga dapat lebih mudah dalam melakukan persiapan data tanpa khawatir ada data yang tidak tersentuh.

  Terkait merging data User, Book, dan Rating terfokus pada data Rating karena data Rating memiliki rating dari pengguna dan buku yang menjadi tujuan utama kita dalam membuat sistem rekomendasi. Merging ini bisa dilakukan dengan 2 tahap :
  
  - Merging data User dan Rating

    Merging antara data User dan Rating dengan metode 'inner join' dimana data Rating sebagai data utama. Didapatkan jumlah pengguna unik pada data hasil merging sama dengan jumlah pengguna unik pada data Rating. Kita lanjut merging dengan data Book.
    
    Jumlah User-ID unik : 105283

  - Merging data hasil merging sebelumnya dengan data Book

    Merging antara data sebelumnya dengan data Book dengan metode 'left join' dan 'right join' dimana ISBN sebagai penghubung kedua data. Didapatkan hasil seperti ini :
    
    1. 'left join'
    
       Jumlah User-ID unik : 105283
       
       Jumlah ISBN unik    : 340556

       Jumlah data null di setiap fitur data :

       ```
       User-ID                     0
       ISBN                        0
       Book-Rating                 0
       Location                    0
       Age                    309492
       Book-Title             118644
       Book-Author            118645
       Year-Of-Publication    118644
       Publisher              118646
       Image-URL-S            118644
       Image-URL-M            118644
       Image-URL-L            118648
       dtype: int64
       ```
       
    2. 'right join'
       
       Jumlah User-ID unik : 92107
       
       Jumlah ISBN unik    : 271360
       
       Jumlah data null di setiap fitur data :
       
       ```
       User-ID                  1209
       ISBN                        0
       Book-Rating              1209
       Location                 1209
       Age                    279044
       Book-Title                  0
       Book-Author                 1
       Year-Of-Publication         0
       Publisher                   2
       Image-URL-S                 0
       Image-URL-M                 0
       Image-URL-L                 4
       dtype: int64
       ```
       
    Terlihat bahwa jika kita menggunakan metode 'left join' menimbulkan banyak data buku yang tidak teridentifikasi. Sedangkan jika menggunakan 'right join' akan menimbulkan banyak data pengguna yang tidak teridentifikasi. Maka dari itu, kita perlu menggunakan metode 'inner join' untuk merging, sama seperti merging sebelumnya.
       
    Jumlah User-ID unik : 92106
    
    Jumlah ISBN unik    : 270151
    
    Jumlah data null di setiap fitur data :
    
    ```
    User-ID                     0
    ISBN                        0
    Book-Rating                 0
    Location                    0
    Age                    277835
    Book-Title                  0
    Book-Author                 1
    Year-Of-Publication         0
    Publisher                   2
    Image-URL-S                 0
    Image-URL-M                 0
    Image-URL-L                 4
    dtype: int64
    ```
    
    Didapatkan hasil data hasil merging dengan seluruh identitas pengguna dan buku teridentifiksi dan telah terintegrasi dengan nilai rating.

- Menangani Missing Values

  - Penjelasan : Missing values merupakan nilai - nilai kosong dalam sebuah fitur, baris, ataupun kolom. Adanya missing values ini membuat lubang serangkaian data dalam suatu fitur dalam dataset sehingga perlu adanya penanganan khusus terkait missing values ini.

  - Alasan Penggunaan : Penanganan terhadap Missing Values berguna untuk mengurangi ketidakjelasan konteks dari data yang ada dan membantu meningkatkan kualitas model berbasis data tersebut.

  Terlihat bahwa missing values terdapat pada kolom Age, Book-Author, Publisher, dan Image-URL-L. Di awal kita telah memahami bahwa kolom Location dan Age tidak berpengaruh terhadap tingkat presisi sistem rekomendasi yang akan dibuat sehingga kita bisa drop 2 kolom tersebut. Kita juga memahami bahwa kolom Book-Title, Book-Author, Year-Of-Publication, Image-URL-S, Image-URL-M	Image-URL-L tidak berpengaruh pada tingkat presisi sistem rekomendasi yang akan dibuat. Namun, kita hanya akan drop 2 kolom, yaitu : kolom Image-URL-M dan Image-URL-L karena sudah kolom Image-URL-M. Untuk missing values pada kolom Book-Author, kita bisa menambahkan string tertentu agar tidak null. Untuk missing values pada kolom Publisher, kita bisa drop baris yang memiliki nilai null mengingat nilai pada kolom ini tidak boleh null dan tidak boleh digantikan dengan nilai lain, terlebih lagi jumlah missing values-nya hanya 2 buah.
  
  Jumlah data null di setiap fitur data :
  
  ```
  User-ID                0
  ISBN                   0
  Book-Rating            0
  Book-Title             0
  Book-Author            0
  Year-Of-Publication    0
  Publisher              0
  Image-URL-M            0
  dtype: int64
  ```
  
  Kita juga akan merubah nama fitur dari data menjadi istilah - istilah dalam Bahasa Indonesia agar lebih mudah dimengerti orang Indonesia.

- Persiapan Data Untuk Content-Based Filtering

  - Menghapus Data Duplikat dan Mengurutkan Data

    - Penjelasan : Data duplikat merupakan data yang bernilai sama persis dengan data lainnya serta berdampak pada ketidakefektifan model Machine Learning dalam belajar sesuatu yang berulang. Pengurutan data merupakan proses untuk menempatkan data dari urutan terkecil sampai terbesar ataupun sebaliknya.

    - Alasan Penggunaan : Menghapus data duplikat berguna untuk meningkatkan keefektifan kalkulasi data maupun latihan dan evaluasi bagi model Machine Learning. Mengurutkan data berguna untuk meningkatkan akurasi rekomendasi buku karena dataset memiliki beberapa buku dengan data yang sama persis, hanya rating buku tersebut yang berbeda.

    Terkait persiapan data untuk Content-Based Filtering, kita hanya akan menggunakan sebagian data dari data asli. Hal ini dikarenakan keterbatasan hardware untuk melakukan komputasi terhadap pemodelan Machine Learning. Jumlah data yang digunakan adalah seperseribu dari jumlah data asli.

    Kita telah berhasil menangani missing values. Namun, masih ada satu hal lagi yang perlu dilakukan terkait data tersebut, yaitu menghapus data duplikat. Data duplikat dapat dengan mudah diketahui dari kolom id_buku. Hal ini dikarenakan id_buku merupakan berisi identitas buku dimana satu buku berpotensi memiliki beberapa macam rating yang berbeda dari pengguna yang berbeda - beda. Kita bisa menghapus seluruh data yang sama (terduplikat) dan menyisakan satu data dari setiap data duplikat tersebut.

    Kita juga akan melakukan pengurutan data berdasarkan fitur id_buku yang nilainya berbasis numerik guna meningkatkan akurasi rekomendasi buku sesuai penerbit buku.
    
    ```
    array(['simon&amp;schuster', 'collins', 'harpercollinsuk',
       'harpercollinspublishers', 'trafalgarsquarepublishing',
       'trafalgarsquarebooks', 'atlantic', 'fairmountbooksltdremainders',
       'trafalgarsquare', 'collin', 'williamcollinssonscoltd',
       'harperflamingocanada', 'harpercollinscanada', 'harpercollins',
       'harper&amp;collins', 'harpercollinspublisher',
       'smithmarkpublishing', 'cuisinarts', 'collinsharvill',
       'collinspubsanfrancisco', 'harpercollinsjuvenilebooks',
       'smithmarkpub', 'harpercollinswillow', 'bbcbooks', 'totembooks',
       'flamingo', 'harperperennial', 'harpercollinspubrs',
       'collinspublishers', 'perennialcurrents', 'harperentertainment',
       'indus'], dtype=object)
    ```
    
    Dengan ini, data untuk pemodelan Machine berbasis Content-Based Filtering siap digunakan.

- Persiapan Data Untuk Collaborative Filtering

  - Pembagian Data Latih & Validasi

    - Penjelasan : Proses membagi dataset menjadi data latih dan data validasi dimana data latih akan digunakan oleh model Machine Learning untuk latihan dan data validasi sebagai validator bagi performa model Machine Learning yang dihitung melalui metrik evaluasi.

    - Alasan Penggunaan : Pembagian Dataset menjadi data latih yang berguna untuk menjadi masukan bagi model Machine Learning saat latihan dan data validasi yang berguna untuk menjadi validator model Machine Learning, dan menghitung metrik evaluasi, serta memudahkan standarisasi pada data latih dan data validasi.

    Terkait persiapan data untuk Collaborative Filtering, kita hanya akan menggunakan sebagian data dari data asli. Hal ini dikarenakan keterbatasan hardware untuk melakukan komputasi terhadap pemodelan Machine Learning. Jumlah data yang digunakna adalah seperseratus dari jumlah data asli.

    Pada Collaborative Filtering, perlu adanya pembagian data menjadi data latih dan data validasi. Hal ini dilakukan supaya tidak terjadi ketidakmampuan model untuk berlatih maupun melakukan validasi, serta mengalami underfitting dan overfitting.
    
    Bentuk fitur masukan data latih : (82490, 2)
    
    Bentuk fitur target data latih  : (82490,)
    
    Dengan ini, data untuk pemodelan Machine berbasis Collaborative Filtering siap digunakan.

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

  Tahapan pemodelan berbasis Content-Based Filtering :
  
  1. Modelling dengan Content-Based Filtering diawali dengan menampilkan nilai pada penerbit_buku

  2. Proses selanjutnya ialah melakukan fitting dan transformasi nilai pada penerbit_buku menemukan representasi fitur penting berdasarkan penerbit_buku

  3. Proses selanjutnya ialah melakukan perubahan bentuk data dari dataframe menjadi matriks

  4. Melakukan overview terhadap hasil komputasi tfidfvectorizer terhadap data tersebut

  5. Menghitung derajat kesamaan antar judul_buku pada data tersebut

  6. Melakukan overview terhadap hasil komputasi fungsi cosine_similarity terhadap data tersebut

  Fitur hasil TF-IDF Vectorizer :
  
  ```
  ['amp',
   'atlantic',
   'bbcbooks',
   'collin',
   'collins',
   'collinsharvill',
   'collinspublishers',
   'collinspubsanfrancisco',
   'cuisinarts',
   'fairmountbooksltdremainders',
   'flamingo',
   'harper',
   'harpercollins',
   'harpercollinscanada',
   'harpercollinsjuvenilebooks',
   'harpercollinspublisher',
   'harpercollinspublishers',
   'harpercollinspubrs',
   'harpercollinsuk',
   'harpercollinswillow',
   'harperentertainment',
   'harperflamingocanada',
   'harperperennial',
   'indus',
   'perennialcurrents',
   'schuster',
   'simon',
   'smithmarkpub',
   'smithmarkpublishing',
   'totembooks',
   'trafalgarsquare',
   'trafalgarsquarebooks',
   'trafalgarsquarepublishing',
   'williamcollinssonscoltd']
  ```
  
  Matriks TF-IDF :
  
  ```
  matrix([[0.44943143, 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        ...,
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ],
        [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
         0.        ]])
  ```
  
  Overview hubungan judul buku dan penerbit buku dari matriks TF-IDF :
  

  |                                                      | indus | bbcbooks | collinspubsanfrancisco | atlantic |  collins | harpercollins | harpercollinswillow |  harper | trafalgarsquare | flamingo |
  |-----------------------------------------------------:|------:|---------:|-----------------------:|---------:|---------:|--------------:|--------------------:|--------:|----------------:|---------:|
  |                                           judul_buku |       |          |                        |          |          |               |                     |         |                 |          |
  |                         Bess                         |   0.0 |      0.0 |                    0.0 |      0.0 | 1.000000 |           0.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |
  |                    Agent In Place                    |   0.0 |      0.0 |                    0.0 |      0.0 | 1.000000 |           0.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |
  |                The Colours of Her Day                |   0.0 |      0.0 |                    0.0 |      0.0 | 0.330015 |           0.0 |                 0.0 | 0.67493 |             0.0 |      0.0 |
  | The Dixon Cornbelt League and other baseball stories |   0.0 |      0.0 |                    0.0 |      0.0 | 0.000000 |           1.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |
  |         British Bats (Collins New Naturalist)        |   0.0 |      0.0 |                    0.0 |      0.0 | 0.000000 |           0.0 |                 0.0 | 0.00000 |             1.0 |      0.0 |
  |                Sue Crowther's Marriage               |   0.0 |      0.0 |                    0.0 |      0.0 | 0.000000 |           0.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |
  |             Twopence to Cross the Mersey             |   0.0 |      0.0 |                    0.0 |      0.0 | 0.000000 |           0.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |
  |         Wedding cakes, rats, and rodeo queens        |   0.0 |      0.0 |                    0.0 |      0.0 | 0.000000 |           1.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |
  |           The Revenge of Murray the Mantis           |   0.0 |      0.0 |                    0.0 |      0.0 | 0.000000 |           0.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |
  |                         Hugh                         |   0.0 |      0.0 |                    0.0 |      0.0 | 1.000000 |           0.0 |                 0.0 | 0.00000 |             0.0 |      0.0 |

  
  Dataframe Cosine Similarity :
  
  ```
  array([[1., 0., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.],
       [0., 0., 0., ..., 0., 0., 1.]])
  ```
  
  Overview hubungan antar judul buku dari dataframe Cosine Similarity :
  

  |                                                                       judul_buku | The Dixon Cornbelt League and other baseball stories | AMPHIBIANS AND REPTILES | Which Colour? | The Golden Gate | Little Grey Rabbit makes lace (The Little Grey Rabbit library) | STEVE MCQUEEN | The Echoing Grove | The Financial post selects the 100 best companies to work for in Canada | The perfect carrier | The stationary ark |
  |---------------------------------------------------------------------------------:|-----------------------------------------------------:|------------------------:|--------------:|----------------:|---------------------------------------------------------------:|--------------:|------------------:|------------------------------------------------------------------------:|--------------------:|-------------------:|
  |                                                                       judul_buku |                                                      |                         |               |                 |                                                                |               |                   |                                                                         |                     |                    |
  |                                A bridge of magpies                               |                                                  0.0 |                     0.0 |           0.0 |             1.0 |                                                            1.0 |           0.0 |               0.0 |                                                                     1.0 |                 1.0 |                1.0 |
  |                                Bedside Guardian 29                               |                                                  0.0 |                     0.0 |           0.0 |             0.0 |                                                            0.0 |           0.0 |               0.0 |                                                                     0.0 |                 0.0 |                0.0 |
  |                              Lord of the Far Island                              |                                                  0.0 |                     0.0 |           0.0 |             1.0 |                                                            1.0 |           0.0 |               0.0 |                                                                     1.0 |                 1.0 |                1.0 |
  |                                 The Echoing Grove                                |                                                  0.0 |                     0.0 |           1.0 |             0.0 |                                                            0.0 |           1.0 |               1.0 |                                                                     0.0 |                 0.0 |                0.0 |
  |               Little Grey Rabbit's Christmas (Collins Colour Cubs)               |                                                  0.0 |                     0.0 |           1.0 |             0.0 |                                                            0.0 |           1.0 |               1.0 |                                                                     0.0 |                 0.0 |                0.0 |
  |                                   Babe Dressing                                  |                                                  0.0 |                     0.0 |           1.0 |             0.0 |                                                            0.0 |           1.0 |               1.0 |                                                                     0.0 |                 0.0 |                0.0 |
  |                                   STEVE MCQUEEN                                  |                                                  0.0 |                     0.0 |           1.0 |             0.0 |                                                            0.0 |           1.0 |               1.0 |                                                                     0.0 |                 0.0 |                0.0 |
  |                          Sir Gawain and the Green Knight                         |                                                  0.0 |                     0.0 |           0.0 |             0.0 |                                                            0.0 |           0.0 |               0.0 |                                                                     0.0 |                 0.0 |                0.0 |
  | The diaries of Lord Louis Mountbatten, 1920-1922: Tours with the Prince of Wales |                                                  0.0 |                     0.0 |           0.0 |             1.0 |                                                            1.0 |           0.0 |               0.0 |                                                                     1.0 |                 1.0 |                1.0 |
  |                                   A good woman                                   |                                                  0.0 |                     0.0 |           1.0 |             0.0 |                                                            0.0 |           1.0 |               1.0 |                                                                     0.0 |                 0.0 |                0.0 |

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

  Tahapan pemodelan berbasis Collaborative Filtering :
  
  1. Modelling dengan Collaborative Filtering diawali dengan membuat kelas yang akan digunakan menjadi arsitektur model Deep Learning

  2. Memanggil kelas dengan menyertakan parameter - parameter yang relevan untuk membuat sebuah model Deep Learning sistem rekomendasi dan melakukan kompilasi model tersebut dengan memasukkan nilai Optimizer, Loss Function, dan Metric yang relevan

  3. Melakukan latihan pada model Deep Learning tersebut dengan menyertakan data latih dan data validasi dan beberapa parameter lain untuk dapat melatih model dengan lancar dan aman

  Riwayat latihan model berbasis Collaborative Filtering :
  
  ```
  Epoch 1/10
  10312/10312 [==============================] - 197s 19ms/step - loss: 0.6447 - root_mean_squared_error: 0.4140 - val_loss: 0.6048 - val_root_mean_squared_error: 0.3908
  Epoch 2/10
  10312/10312 [==============================] - 194s 19ms/step - loss: 0.5468 - root_mean_squared_error: 0.3569 - val_loss: 0.5671 - val_root_mean_squared_error: 0.3697
  Epoch 3/10
  10312/10312 [==============================] - 194s 19ms/step - loss: 0.5152 - root_mean_squared_error: 0.3398 - val_loss: 0.5532 - val_root_mean_squared_error: 0.3629
  Epoch 4/10
  10312/10312 [==============================] - 193s 19ms/step - loss: 0.4937 - root_mean_squared_error: 0.3271 - val_loss: 0.5441 - val_root_mean_squared_error: 0.3585
  Epoch 5/10
  10312/10312 [==============================] - 194s 19ms/step - loss: 0.4771 - root_mean_squared_error: 0.3170 - val_loss: 0.5384 - val_root_mean_squared_error: 0.3557
  Epoch 6/10
  10312/10312 [==============================] - 195s 19ms/step - loss: 0.4630 - root_mean_squared_error: 0.3083 - val_loss: 0.5357 - val_root_mean_squared_error: 0.3545
  Epoch 7/10
  10312/10312 [==============================] - 195s 19ms/step - loss: 0.4512 - root_mean_squared_error: 0.3010 - val_loss: 0.5338 - val_root_mean_squared_error: 0.3535
  Epoch 8/10
  10312/10312 [==============================] - 200s 19ms/step - loss: 0.4404 - root_mean_squared_error: 0.2940 - val_loss: 0.5332 - val_root_mean_squared_error: 0.3531
  Epoch 9/10
  10312/10312 [==============================] - 198s 19ms/step - loss: 0.4314 - root_mean_squared_error: 0.2882 - val_loss: 0.5334 - val_root_mean_squared_error: 0.3529
  Epoch 10/10
  10312/10312 [==============================] - 205s 20ms/step - loss: 0.4234 - root_mean_squared_error: 0.2830 - val_loss: 0.5340 - val_root_mean_squared_error: 0.3530
  ```

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

Evaluation model berbasis Content-Based Filtering :

1. Acuan buku :

   <div align="center">
     
   |        |    id_buku | rating_buku |    judul_buku | penulis_buku | tahun_publikasi_buku |           penerbit_buku |                                       gambar_buku |
   |-------:|-----------:|------------:|--------------:|-------------:|---------------------:|------------------------:|--------------------------------------------------:|
   | 517583 | 0002229544 |           0 | Cold New Dawn | Ian St James |                 1987 | harpercollinspublishers | http://images.amazon.com/images/P/0002229544.0... |
   
   </div>
      
2. Rekomendasi buku :
      
   <div align="center">
      
   |   |                                        judul_buku |           penerbit_buku |
   |--:|--------------------------------------------------:|------------------------:|
   | 0 |                     Hilda Boswell's Blue Treasury | harpercollinspublishers |
   | 1 |                                The Family Mashber | harpercollinspublishers |
   | 2 |                 Glue (First Facts - First Skills) | harpercollinspublishers |
   | 3 |                        The Brambly Hedge Treasury | harpercollinspublishers |
   | 4 | Little Grey Rabbit's Birthday (Little Grey Rab... | harpercollinspublishers |
      
   </div>
     
Evaluation model berbasis Collaborative Filtering :
   
1. Visualisasi riwayat latihan model :
      
   ![1](https://user-images.githubusercontent.com/40670734/192267863-21a5621f-bb78-4bf9-8388-102508fe74ad.jpg)
      
2. Rekomendasi buku :
   
   ```
   Rekomendasi Buku Untuk Pengguna 60244
   ========================================
   Buku Favorit Pengguna
   ----------------------------------------
   A Wrinkle in Time - Laure Leaf
   Zen and the Art of Motorcycle Maintenance : An Inquiry into Values - Audio Renaissance
   The Little House - Houghton Mifflin
   Heckedy Peg - Voyager Books
   Ed Emberley's Drawing Book: Make a World - Little, Brown
   Little Bear's Friend - HarperTrophy
   The Hours : A Novel - Farrar, Straus and Giroux
   Everyday Sacred: A Women's Journey Home - HarperSanFrancisco
   Who Will Run the Frog Hospital? - Warner Books
   The Shelter of Each Other: Rebuilding Our Families - Putnam Publishing Group
   ----------------------------------------
   10 Besar Rekomendasi Buku
   ----------------------------------------
   Hard Revolution: A Novel - Little, Brown
   Second Wind - Putnam Publishing Group
   The Bridges of Madison County - Warner Books
   The Fifth Profession - Warner Books
   Shooting at Loons (Deborah Knott Mysteries (Paperback)) - Warner Books
   The Third Eye (Laurel Leaf Books) - Laure Leaf
   The Lovely Bones: A Novel - Little, Brown
   Short &amp; Tall Tales: Moose County Legends Collected by James Mackintosh Qwilleran - Putnam Publishing Group
   The Slave Dancer (Laurel-Leaf Historical Fiction) - Laure Leaf
   Counting Coup: A True Story of Basketball and Honor on the Little Big Horn - Warner Books
   ```

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
