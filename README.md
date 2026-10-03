# Akıllı Yara Asistanı

Akıllı Yara Asistanı yara bakım destek sistemi için Vercel'e taşınabilir web prototipi.

## Özellikler

- Profesyonel sağlık arayüzü
- Yara görüntüsü yükleme ve önizleme
- Tarayıcı içinde ONNX ile yara evresi tahmini
- Gerektiğinde manuel evre seçimi
- Kural tabanlı bakım önerisi
- Kısa RAG/rehber notları
- OpenRouter üzerinden asistana soru sorma

## Model Yapısı

Web/Vercel sürümünde görüntü işleme modeli frontend tarafında çalışır:

```text
public/models/mobilenet.web.onnx
```

Bu dosya tarayıcı tarafından indirilir ve `onnxruntime-web` ile çalıştırılır. Böylece Vercel'de GPU veya Python sunucu gerekmez.

Full ensemble ONNX'e çevrilmiştir ancak web için ağırdır:

```text
resnet   ~98 MB
alexnet  ~234 MB
vit      ~327 MB
toplam   ~670 MB
```

Bu yüzden GitHub/Vercel sürümünde sadece MobileNet ONNX dosyaları kullanılacak şekilde ayarlandı. Full ensemble dosyaları yerel test için saklanabilir; `.gitignore` büyük dosyaların yanlışlıkla repoya eklenmesini engeller.

## Kurulum

```bash
cd /Users/osmanural/Desktop/AYA/akilliyaraasistani
npm install
cp .env.example .env
```

`.env` içine gerçek OpenRouter key ekle:

```text
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=qwen/qwen3.7-flash
```

## Local Çalıştırma

OpenRouter soru-cevap endpointini yerelde test etmek için:

```bash
npm run local
```

Adres:

```text
http://localhost:3000
```

3000 portu doluysa:

```bash
PORT=3001 npm run local
```

```text
http://localhost:3001
```

## Vercel

Vercel ile deploy ederken GitHub reposunu bağla. Build ayarı gerekmez; bu proje statik dosyaları `public/` klasöründen, API endpointini `api/chat.js` dosyasından çalıştırır.



Vercel project settings içinde environment variable olarak ekle:

```text
OPENROUTER_API_KEY
OPENROUTER_MODEL
```

API key frontend'e yazılmaz. `/api/chat` serverless endpoint içinde güvenli şekilde kullanılır.

## Not

Bu sistem tanı koymaz ve tedavi belirlemez. Görüntü modeli yalnızca ön değerlendirme desteği verir; sağlık profesyoneli gerekli gördüğünde evreyi manuel düzeltmeli ve klinik değerlendirme esas alınmalıdır.
