### 2023-08-13
1- Düzenli hata düzeltmesi
- Minimum toplam epoch sayısını 1 olarak değiştirin ve minimum toplam epoch sayısını 2 olarak değiştirin
- Önceden eğitilmiş modellerin kullanılmamasına ilişkin eğitim hatalarını düzeltin
- Eşlik eden vokallerin ayrılmasından sonra grafik belleğini temizleyin
- Faiss kaydetme yolunu mutlak yoldan göreceli yola değiştirin
- Boşluk içeren yol destekleniyor (hem eğitim seti yolu hem de deney adı desteklenir ve artık hatalar bildirilmeyecektir)
- Dosya listesi zorunlu utf8 kodlamasını iptal eder
- Gerçek zamanlı ses değişiklikleri sırasında faiss arama tarafından neden olan CPU tüketim sorununu çözün

2- Anahtar güncellemeler
- Mevcut en güçlü açık kaynaklı ses perdesi çıkarma modeli RMVPE eğitin ve RVC eğitimi için kullanın, çevrimdışı/gerçek zamanlı çıkarım, PyTorch/Onnx/DirectML desteklemek üzere kullanın
- Pytorch_DML aracılığıyla AMD ve Intel grafik kartlarını destekleyin

(1) Gerçek zamanlı ses değişimi (2) Çıkarım (3) Vokal eşlikçisinin ayrılması (4) Şu anda desteklenmeyen eğitim, CPU eğitimine geçiş yapacaktır; RMVPE'nin gpu ile Onnx_Dml tarafından çıkarımını destekler


### 2023-06-18
- Yeni önceden eğitilmiş v2 modeller: 32k ve 48k
- F0 olmayan model çıkarım hatalarını düzeltin
- Eğitim seti 1 saati aşarsa, özellik şeklini azaltmak için otomatik minibatch-kmeans yaparak indeks eğitimini, eklemeyi ve aramayı çok daha hızlı hale getirin.
- Bir oyuncak vocal2guitar huggingface alanı sağlayın
- Aykırı değer kısa yol eğitim seti seslerini otomatik olarak silin
- Onnx dışa aktarma sekmesi

Başarısız deneyler:
- ~~Özellik alımı: zamansal özellik alımı ekle: etkili değil~~
- ~~Özellik alımı: PCAR boyut indirgeme ekleyin: arama daha yavaş~~
- ~~Eğitim sırasında rastgele veri artırma: etkili değil~~

Yapılacaklar listesi:
- ~~Vocos-RVC (minik vokoder): etkili değil~~
- ~~Eğitim için Crepe desteği: RMVPE ile değiştirildi~~
- ~~Yarı hassas Crepe çıkarımı: RMVPE ile değiştirildi. Ve zor elde edilebilir.~~
- F0 düzenleyici desteği

### 2023-05-28
- v2 jupyter notebook, korece değişiklik günlüğü, bazı ortam gereksinimlerini düzeltme
- Sessiz ünsüz ve nefes koruma modu ekleme
- Crepe-full pitch tespitini destekleme
- UVR5 vokal ayrımı: dereverb modellerini ve yankı giderme modellerini destekleme
- İndeks adının üzerine deney adı ve sürümünü eklemek
- Toplu ses dönüşüm işlemi ve UVR5 vokal ayrımı sırasında çıktı seslerinin manuel olarak ihracat formatını seçme desteği
- v1 32k model eğitimi artık desteklenmiyor

### 2023-05-13
- Tek tıklamalı paketin eski sürümünde gereksiz kodları temizleme: lib.infer_pack ve uvr5_pack
- Eğitim seti ön işlemede sahte çoklu işlem hatasını düzeltme
- Harvest pitch tanıma algoritması için median filtreleme yarıçap ayarı ekleme
- Ses dışı örnekleme için ihracat sonrası örnekleme desteği
- Eğitim için "n_cpu" ayarı, "f0 çıkarımı" ndan "veri ön işleme ve f0 çıkarımına" değiştirildi
- Günlükler klasörü altındaki indeks yollarını otomatik olarak algılama ve açılır liste işlevi sağlama
- Sekme sayfasında "Sıkça Sorulan Sorular ve Cevaplar" ekleyin (github RVC wiki'ye de bakabilirsiniz)
- Çıkarım sırasında aynı giriş ses yolu kullanıldığında harvest pitch önbelleği (amaç: harvest pitch çıkarımı kullanırken, tüm iş akışı uzun ve tekrarlayan bir pitch çıkarım işlemi üzerinden geçer. Önbellekleme kullanılmazsa, farklı timbre, indeks ve pitch median filtreleme yarıçapı ayarlarıyla deney yapan kullanıcılar ilk çıkarımdan sonra çok acılı bir bekleme süreci yaşayacaklardır)

### 2023-05-14
- Girişin sesin ses zarfını çıkarım veya değiştirme seçeneği ekleme ("girişin sessizleştirilmesi ve çıktının düşük amplitüdlü gürültü oluşturması" sorununu hafifletebilir. Giriş ses arka plan gürültüsü yüksekse, açmanız önerilmez ve varsayılan olarak açılmaz (1 açık olarak düşünülebilir))
- Özelliğin belirli bir sıklıkta küçük modelleri kaydetmeyi destekleme

 (farklı epoch'ların performansını görmek istiyorsanız, ancak her seferinde tüm büyük kontrolleri kaydetmek istemiyorsanız ve manuel olarak her seferinde ckpt işlemi ile küçük modelleri çıkarmak istemiyorsanız, bu özellik çok pratik olacaktır)
- Sunucunun global proxy'sinin neden olduğu "bağlantı hataları" sorununu ortam değişkenleri ayarlayarak çözme
- Önceden eğitilmiş v2 modelleri destekleme (şu anda sadece 40k versiyonları halka açık olarak test için mevcuttur ve diğer iki örnekleme hızı henüz tam olarak eğitilmemiştir)
- Girişimi aşan aşırı ses sınırlamasını önleme
- Eğitim seti ön işleme ayarlarını hafifçe ayarlama

#######################

Geçmiş değişiklik günlükleri:

### 2023-04-09
- GPU kullanım oranını artırmak için eğitim parametrelerini düzeltme: A100 %25'ten %90'a kadar, V100: %50'den %90'a kadar, 2060S: %60'tan %85'e kadar, P40: %25'ten %95'e kadar; eğitim hızını önemli ölçüde artırıldı
- Parametre değişikliği: toplam batch_size artık her GPU için batch_size
- Toplam_epoch değişti: maksimum sınır 100'den 1000'e çıkarıldı; varsayılan 10'dan 20'ye çıkarıldı
- Ckpt çıkarma sorununu düzelterek pitch tanıma hatasını düzeltme, anormal çıkarıma neden olma
- Dağıtılmış eğitimde her sıra için ckpt kaydetme sorununu düzeltme
- Özellik çıkarımı için nan özellik filtrelemesini uygulama
- Giriş/çıkış sessiz ise rastgele ünsüzler veya gürültü üretme sorununu düzeltme (eski modeller yeni bir veri kümesiyle yeniden eğitilmelidir)

### 2023-04-16 Güncellemesi
- Yerel gerçek zamanlı ses değiştirme mini-GUI eklenmiş, çift tıklama ile go-realtime-gui.bat başlatılır
- Eğitim ve çıkarım sırasında 50Hz'nin altındaki frekans bantları için filtreleme uygulama
- Eğitim ve çıkarım için pyworld'ün minimum pitch çıkarımını 80'den 50'ye indirme, erkek düşük perdeli seslerin 50-80Hz arasında sessizleştirilmeden kalmasına izin verme, varsayılan olarak 80'den 50'ye indirildi
- WebUI, sistem yereli (şu anda en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW desteklenir) uyarınca dil değiştirme desteklemektedir; desteklenmezse varsayılan olarak en_US olarak ayarlanır
- Bazı GPU'ların tanınmasını düzeltme (örneğin, V100-16G tanıma hatası, P4 tanıma hatası)

### 2023-04-28 Güncellemesi
- Daha hızlı hız ve daha yüksek kalite için faiss indeks ayarlarını yükseltme
- Gelecekteki model paylaşımları için total_npy bağımlılığını kaldırma
- Kısıtlamaları kaldırarak 16 serisi GPU'lar için 4GB çıkarım ayarları sağlama
- Belirli ses formatları için UVR5 vokal eşlikçisi ayrımındaki hata sorununu düzeltme
- Gerçek zamanlı ses değiştirme mini-GUI artık 40k ve tembel pitch modellerini destekler

### Gelecek Planlar:
Özellikler:
- Her epoch kaydetmek için küçük modeller çıkarma seçeneği eklemek
- Çıkarım sırasında belirli bir yola ekstra mp3 kaydetme seçeneği eklemek
- Çoklu kişi eğitim sekmesini desteklemek (en fazla 4 kişiye kadar)
