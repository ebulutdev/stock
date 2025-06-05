# Borsa İstanbul Hisse Analizi

Bu Django uygulaması, Borsa İstanbul hisse senetlerinin canlı fiyat ve hedef fiyat bilgilerini gösterir.

## Özellikler

- Borsa İstanbul hisse senetleri için canlı fiyat takibi
- Hedef fiyat analizi
- Redis önbelleği ile performans optimizasyonu
- Kullanıcı dostu arayüz

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Redis sunucusunu başlatın (Windows için Redis'i indirip kurmanız gerekiyor)

3. Django uygulamasını başlatın:
```bash
python manage.py migrate
python manage.py runserver
```

## Kullanım

1. Tarayıcınızda `http://localhost:8000` adresine gidin
2. Hisse senedi sembolünü girin (örn: THYAO, GARAN)
3. "Analiz Et" butonuna tıklayın
4. Sonuçları görüntüleyin

## Notlar

- Hisse senedi sembolleri büyük harfle yazılmalıdır
- Veriler 5 dakikada bir güncellenir
- Redis önbelleği sayesinde gereksiz API çağrıları önlenir 