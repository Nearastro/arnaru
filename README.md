# Arnaru OpenAI Proxy v3

Proxy OpenAI-compatible untuk Arnaru AI API.

Proxy ini otomatis melakukan retry (dengan backoff singkat) kalau Arnaru
membalas kosong atau cuma placeholder seperti `.` saja — terutama setelah
eksekusi tool/command — sehingga client (RikkaHub/dll) selalu mendapat
jawaban teks yang utuh.

## Deploy ke Vercel

1. Extract zip, push ke GitHub
2. Import ke Vercel (Framework: Other)
3. Deploy

## Endpoint

- `POST /v1/chat/completions` — Chat (OpenAI-compatible)
- `GET /v1/models` — List models
- `GET /health` — Health check
- `GET /` — Info

## Test

```bash
curl https://your-app.vercel.app/health
curl https://your-app.vercel.app/v1/models
curl -X POST https://your-app.vercel.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-fable-5","messages":[{"role":"user","content":"Halo"}]}'
```

## RikkaHub Setup

- Provider: OpenAI Compatible
- Base URL: `https://your-app.vercel.app/v1`
- API Key: (kosong, atau isi kalau PROXY_API_KEY diaktifkan)
- Model: `claude-fable-5` (atau model lain)

## Upload File / Gambar

Proxy ini sekarang mendukung upload file dan gambar seperti fitur attachment di
[RikkaHub](https://github.com/rikkahub/rikkahub). RikkaHub (dan client OpenAI-compatible lain)
akan mengirim gambar/file lewat `content` berbentuk array bagian (`text` + `image_url`/`file`),
bukan cuma string biasa. Proxy akan otomatis:

1. Mengekstrak semua teks jadi `question`.
2. Mengekstrak semua `image_url` (base64 data URI atau URL http/https) dan `file` (base64 data URI)
   jadi lampiran biner.
3. Kalau ada lampiran, request ke Arnaru dikirim sebagai `multipart/form-data` (field `files`,
   sesuai dokumentasi endpoint `/api/chat` milik Arnaru), bukan JSON biasa.

Contoh body request (format OpenAI vision standar, ini yang dikirim RikkaHub):

```json
{
  "model": "claude-sonnet-5",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Ini gambar apa?" },
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,iVBORw0KGgoAAAANS..." }
        }
      ]
    }
  ]
}
```

**Catatan penting soal ukuran file:**

- Vercel Serverless Functions punya batas keras ~4.5MB per request body (tidak bisa
  ditingkatkan lewat konfigurasi apa pun). Kompres gambar dulu di sisi client (RikkaHub biasanya
  sudah melakukan ini) kalau file besar.
- Maksimum 9 file per request, mengikuti batas API Arnaru (`/api/chat`).
- Kalau `image_url` berupa URL http/https (bukan base64), proxy akan mengunduhnya dulu di server
  sebelum diteruskan ke Arnaru.
