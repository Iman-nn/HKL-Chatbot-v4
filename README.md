# HKL Chatbot

Chatbot maklumat Hospital Kuala Lumpur: carian FAQ, cadangan soalan dan panduan lokasi daripada CSV tempatan. Refactor ini mengekalkan Flask dan menggantikan latihan model klasifikasi yang tidak digunakan dengan carian TF-IDF. Jawapan daripada korpus perlu disahkan oleh pemilik kandungan HKL; kod tidak menjamin ketepatan maklumat hospital.

## Jalankan

Python 3.10-3.12 dan Node.js 18+ untuk ujian JavaScript. Gunakan Python yang menerima kemas kini keselamatan; pengesahan tempatan menggunakan Python 3.10.6, bukan pengesahan keselamatan runtime tersebut. Untuk produksi, bina semula pada Python 3.12 bertampal dan jalankan semula ujian.

PowerShell, dari direktori projek:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements.lock
.venv/Scripts/python.exe -m waitress --listen=127.0.0.1:5000 --threads=8 app:app
```

requirements.txt menyenaraikan dependency utama; requirements.lock mengunci semua 19 versi runtime yang diuji.

Buka http://127.0.0.1:5000/. Fail HTML sahaja dan Netlify static hosting tidak menjalankan backend Python. Konfigurasi Netlify/SPA lama serta package.json kosong dibuang. Perintah python app.py hanya untuk pembangunan tempatan dan debugger dimatikan.

## Struktur

- app.py: titik masuk WSGI.
- hkl_chatbot/__init__.py: application factory, konfigurasi, routes, pengesahan input dan respons HTTP.
- hkl_chatbot/knowledge.py: validasi CSV, indeks FAQ dan padanan lokasi.
- hkl_chatbot/storage.py: transaksi SQLite, token perbualan, sejarah terhad, TTL dan rate limiting.
- hkl_chatbot/provider.py: fallback OpenRouter pilihan dengan timeout dan had respons.
- templates/index.html, static/chat.js, static/style.css: antara muka tempatan tanpa skrip CDN.
- FAQbot.py: klien terminal yang berkongsi enjin carian.
- corpus_gen.py: eksport JSON kepada CSV dengan validasi dan penggantian fail atomik.
- Fail .joblib lama tidak digunakan atau dimuatkan. Jangan deserialisasi artifak yang tidak dipercayai.

Matriks FAQ dibina sekali bagi setiap worker, bukan setiap permintaan. Tiada keperluan pandas, NLTK corpus, latihan LogisticRegression/SVM atau muat turun model semasa startup.

## Kontrak API

| Endpoint | Kontrak |
| --- | --- |
| POST /start_conversation | JSON kosong {}; pulangkan 201 dengan conversation_id, conversation_token, expires_in |
| POST /ask/:id | Header Authorization: Bearer [conversation_token]; JSON {"message":"..."} |
| GET /history/:id | Token yang sama diperlukan; pulangkan sejarah mengikut urutan simpanan |
| DELETE /history/:id | Token diperlukan; padam perbualan dan mesej; 204 |
| GET /faq_classes | Senarai kategori |
| GET /faqs/:category | FAQ untuk kategori tersebut |
| GET /navigation/:location | Padanan perkataan literal, maksimum 20 rekod |
| GET /health | Status kesediaan indeks dan sambungan storan; tiada butiran dalaman |

Ini ialah perubahan kontrak keselamatan: ID integer lama diganti dengan ID rawak dan token akses berasingan. Klien perlu menggunakan JSON untuk permintaan POST. Token disimpan sebagai hash dalam DB dan hanya dalam memori halaman pada klien; refresh memulakan sesi baharu apabila soalan seterusnya dihantar. Sejarah sesi lama tidak dipindahkan daripada memori proses asal.

Had lalai: body 16 KiB, mesej 2,000 aksara, 60 permintaan/minit/IP, 10 sesi baharu/minit/IP, 1,000 sesi aktif, 50 pasangan mesej/sesi, TTL mutlak satu jam. Permintaan berlebihan menerima 429 dan Retry-After; storan sibuk/penuh menerima 503. Permintaan ask memerlukan token sah sebelum carian dan panggilan luar. Token sesi lain menghasilkan respons 404 yang sama seperti ID tidak wujud.

## Konfigurasi dan operasi

.env.example ialah contoh; fail itu tidak dimuatkan secara automatik. Suntik environment melalui service manager. Nilai pilihan dan had boleh diubah melalui create_app(config) untuk deployment tersuai.

| Environment | Kegunaan |
| --- | --- |
| HKL_DATABASE | Laluan fail SQLite tempatan. Lalai instance/chatbot.sqlite3, di luar direktori static |
| HKL_TRUSTED_HOSTS | Senarai host dipisahkan koma. Lalai localhost, 127.0.0.1, [::1]; tambah domain produksi |
| HKL_HSTS_ENABLED | true selepas HTTPS diwajibkan di reverse proxy |
| OPENROUTER_ENABLED | Lalai false; tiada trafik keluar apabila dimatikan |
| OPENROUTER_API_KEY | Rahsia runtime; jangan masukkan ke fail sumber |
| OPENROUTER_MODEL | Model yang tersedia dalam akaun penyedia; diperlukan jika fallback diaktifkan |

**Kunci API dalam versi asal mesti dibatalkan dan diganti di pihak penyedia.** Pembuangan daripada working tree tidak memadamkannya daripada sejarah Git, klon atau log terdahulu. Tiada kunci lama digunakan atau dihantar semasa pengujian ini. Urus pembersihan sejarah dan klon selepas pembatalan mengikut proses organisasi.

Fallback AI dimatikan secara lalai. Mengaktifkannya menghantar soalan yang tiada padanan tempatan kepada pihak luar. Ia mempunyai had 20 panggilan/minit bagi keseluruhan hos, empat panggilan serentak/worker, had output token, connect/read timeout, had saiz respons dan pemeriksaan deadline. Prompt tidak menjamin pencegahan prompt injection atau ketepatan fakta. Jangan aktifkannya bagi trafik sensitif tanpa keputusan pemilik sistem tentang privasi, penyedia dan kualiti jawapan. UI memberi notis apabila fallback diaktifkan. Input suara boleh menggunakan perkhidmatan pengecaman browser.

SQLite WAL sesuai untuk satu hos dengan beberapa proses/threads berkongsi fail pada cakera tempatan. Letakkan DB di luar webroot untuk deployment sebenar, di bawah akaun servis dengan ACL minimum dan storan/backup tersulit. Jangan letakkan DB di network share atau kongsi fail SQLite antara beberapa hos. Untuk penskalaan berbilang hos, ganti adapter Store dengan PostgreSQL dan stor rate limit teragih; jalankan ujian concurrency dan load semula. Had beban produksi belum ditentukan melalui load test representatif.

Expiry menyekat akses serta-merta. Untuk membuang rekod tamat tempoh ketika aplikasi melahu, jadualkan setiap minit:

```powershell
.venv/Scripts/python.exe -m flask --app app prune-data
```

Pembersihan rekod bukan jaminan pemadaman forensik daripada SQLite, WAL, snapshot atau backup. Padam backup mengikut dasar retensi. Butang Clear chat memadam rekod semasa secara logik.

.htaccess menolak capaian terus ke repositori melalui Apache apabila AllowOverride membenarkannya, untuk mengurangkan pendedahan sumber dan fail DB dalam direktori Laragon. Ia tidak menggantikan konfigurasi web server; Nginx mengabaikan fail ini. Jangan gunakan repositori ini sebagai document root awam.

Letakkan Waitress di belakang reverse proxy HTTPS. Hadkan saiz URL/body, masa sambungan dan jumlah sambungan di proxy; ikat backend ke loopback atau rangkaian persendirian. Aplikasi mengabaikan X-Forwarded-For daripada klien. Di belakang proxy, had IP aplikasi berkongsi alamat proxy; tetapkan had IP di proxy. Jika mengintegrasikan forwarded headers dalam aplikasi kemudian, hanya percaya hop proxy yang diketahui dan sekat akses terus ke backend. CORS tidak dibuka; UI dan API menggunakan origin yang sama. Authentication API menggunakan bearer header, tanpa cookie sesi.

Pantau 429/503, masa respons, ruang cakera, backup, serta readiness /health. Ralat aplikasi mencatat jenis peristiwa dan request ID tanpa teks pengguna, token atau badan respons penyedia. Semak konfigurasi log reverse proxy supaya tiada Authorization header atau body direkodkan.

## Pengesahan

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
node --test tests/test_chat.cjs
node --check static/chat.js
.venv/Scripts/python.exe -m pip check
# Pasang pip-audit dalam persekitaran pembangunan, kemudian audit runtime dan transitif:
.venv/Scripts/python.exe -m pip_audit -r requirements.lock --no-deps --disable-pip
```

Ujian HTTP tempatan disediakan dalam tests/http_smoke.py dan boleh dijalankan dengan .venv/Scripts/python.exe tests/http_smoke.py. Ujian ini lulus pada 19 Oktober 2026. Semakan visual kini dilakukan menggunakan Chromium terasing: desktop 1440x1000, tema gelap serta emulasi telefon 390x844 dan 320x740.

Ujian merangkumi akses silang sesi, input rosak/terlalu besar, literal XSS, TTL, penghapusan sejarah, concurrency, rate limiting merentas worker, kategori/navigasi, caching korpus dan kegagalan penyedia. Ujian JavaScript menggunakan DOM test double; ia bukan ujian visual atau end-to-end pelayar.

Rujukan reka bentuk: [OWASP IDOR Prevention](https://cheatsheetseries.owasp.org/cheatsheets/Insecure_Direct_Object_Reference_Prevention_Cheat_Sheet.html), [Flask Security Considerations](https://flask.palletsprojects.com/en/stable/web-security/), [advisori Windows Werkzeug](https://github.com/pallets/werkzeug/security/advisories/GHSA-g6x2-hccm-hh4m).

## UI remake - 19 Oktober 2026

Antara muka HKL Guide menggunakan panel navigasi biru gelap, kanvas krim, ilustrasi SVG tempatan dan susun atur responsif. Halaman pembukaan menyediakan pintasan ke FAQ dan carian lokasi. Dialog FAQ mengambil kategori/soalan daripada API sebenar; dialog lokasi membolehkan carian nama fasiliti dan menghantar lokasi dipilih kepada sembang.

Butang New conversation dan Start fresh memadam sesi semasa melalui API sebelum memulangkan halaman pembukaan. Jika pemadaman gagal, mesej dan token sesi dikekalkan supaya pengguna boleh mencuba semula. Tema gelap, pengiraan aksara, input suara, navigasi keyboard dan reduced motion disokong.

Ujian selepas remake: 33 ujian backend sedia ada, 3 semakan HTML/CSP/control bindings, dan 13 ujian JavaScript lulus. Ujian HTTP sebenar melalui Waitress kini lulus termasuk aset dan fungsi sembang/pemadaman. Semakan visual menggunakan Chromium terasing lulus untuk desktop, tema gelap, serta emulasi telefon 390 px dan 320 px. Tiada overflow mendatar atau ralat JavaScript dikesan. Aliran FAQ ke sembang, reset dan carian lokasi juga disahkan dalam browser.

Kod baharu tidak memerlukan dependency atau CDN tambahan. Arkib versi UI baharu disediakan dalam artifacts/hkl-chatbot-ui-remake.zip; arkib remaster asal dikekalkan sebagai versi terdahulu.

Pratonton tempatan untuk sesi reka bentuk ini berjalan di http://127.0.0.1:5050/. Untuk menjalankannya semula:

```powershell
.venv/Scripts/python.exe -m waitress --listen=127.0.0.1:5050 --threads=8 app:app
```

Bukti visual disimpan dalam artifacts/ui-desktop.png, ui-dark.png, ui-mobile.png, ui-mobile-small.png, ui-conversation.png dan ui-locations.png. Semakan browser pilihan boleh diulang melalui tests/ui_browser_check.py menggunakan Python yang mempunyai Playwright dan binari Chromium; dependency ini hanya diperlukan untuk QA, bukan untuk menjalankan aplikasi.

Semakan Chromium tidak menggantikan ujian Safari/Firefox atau peranti telefon fizikal.
