# Changelog remaster HKL Chatbot

## Jawapan sembang - 19 Oktober 2026

- `static/chat.js` dan `public/static/chat.js`: renderer DOM untuk bold, perenggan,
  senarai, kod dan pautan; HTML mentah kekal literal, protokol berbahaya disekat,
  dan notis AI dipaparkan sebagai label berasingan daripada kandungan jawapan.
- `static/style.css` dan `public/static/style.css`: jarak teks lebih padat,
  pautan aksesibel, label AI dan footer sumber yang menyokong tema gelap/telefon.
- `hkl_chatbot/contact.py`: jawapan hubungan umum BM/English berdasarkan laman
  rasmi HKL, dengan pautan sumber dan tarikh semakan; tidak memanggil OpenRouter.
- `hkl_chatbot/knowledge.py`: tapis cadangan yang lemah, silang topik, berulang
  atau sudah dijawab; jangan cadangkan semula lokasi yang baru dipaparkan.
- `hkl_chatbot/provider.py`: arahan jawapan lebih ringkas, mengikut bahasa pengguna,
  dan larangan mereka nombor telefon, URL, waktu operasi atau rujukan.
- Ujian: pemformatan/pautan selamat, XSS, label AI, sumber kurasi, pilihan jawapan
  hubungan, relevansi cadangan dan kesamaan aset Vercel. Semakan Chromium terasing
  menguji jawapan hubungan sebenar serta respons AI simulasi tanpa panggilan luar.

## Skop dan penemuan

Sistem asal ialah aplikasi web Python/Flask monolitik dengan Jinja, JavaScript/jQuery, pandas, NLTK dan scikit-learn. Ia memadankan soalan FAQ daripada CSV, menyediakan navigasi hospital, menyimpan sejarah dalam dictionary proses dan menggunakan OpenRouter apabila padanan rendah. Skrip terminal juga melatih SVM semasa import. Antara muka HTML wujud dalam beberapa salinan.

Isu yang disahkan:
- Rahsia OpenRouter terbenam dalam kod; pembatalan kunci masih perlu dilakukan di pihak penyedia.
- ID perbualan berurutan tanpa pengesahan pemilikan membolehkan sejarah sesi lain dibaca.
- Mesej/cadangan dimasukkan ke HTML tanpa escaping, membuka ruang XSS.
- Tiada had masa panggilan luar, saiz input, kadar permintaan, kapasiti sesi atau retensi sejarah.
- Matriks korpus dikira semula setiap soalan, dan model klasifikasi dilatih tetapi tidak digunakan dalam laluan jawapan web.
- Path CSV relatif dan corpus NLTK menjadikan startup bergantung kepada direktori kerja/data tambahan.
- Satu rekod lokasi RHU tiada arahan perjalanan. Kod kini memaparkan arahan belum tersedia; kandungan asal tidak direka atau diubah.
- Konfigurasi Netlify statik dan package.json kosong tidak menyediakan backend Flask.
- Versi Werkzeug global asal mempunyai advisori keselamatan Windows; runtime projek menggunakan versi bertampal yang dikunci.

## Perubahan fail dan baris

- app.py:3 - Menggantikan monolit dengan titik masuk WSGI dan mematikan debug.
- hkl_chatbot/__init__.py:22 - Memusatkan konfigurasi, validasi JSON/Unicode/body, pengesahan bearer token, rate limiting, security headers dan ralat generik.
- hkl_chatbot/knowledge.py:32 - Membina matriks TF-IDF sekali, membuang latihan klasifikasi yang tidak digunakan, mendahulukan FAQ tepat dan menggantikan carian regex dengan padanan perkataan literal.
- hkl_chatbot/storage.py:25 - Menambah SQLite WAL, transaksi atomik, token rawak dengan hash, TTL, had kapasiti/sejarah dan penghapusan perbualan.
- hkl_chatbot/provider.py:14 - Mengeluarkan kunci terbenam; fallback pilihan dimatikan secara lalai, dengan timeout, had respons/concurrency dan validasi UTF-8.
- static/chat.js:40 - Menggantikan interpolasi HTML dengan textContent; menghapuskan event handler inline, mencegah penghantaran berganda dan memulihkan soalan selepas kegagalan.
- templates/index.html:1 - Menggunakan aset tempatan, struktur aksesibel, notis privasi dan butang Clear chat; skrip/style diasingkan untuk CSP.
- static/style.css:1 - Menyediakan gaya responsif dan dark mode tanpa CDN.
- FAQbot.py:11 - Menggunakan enjin carian yang sama dan main guard supaya import tidak menjalankan latihan atau membuka prompt.
- corpus_gen.py:13 - Memvalidasi struktur JSON dan menulis CSV secara atomik supaya eksport gagal tidak merosakkan fail sedia ada.
- requirements.lock:1 - Mengunci 19 dependency runtime termasuk Flask 3.1.3 dan Werkzeug 3.1.9; menyertakan requirements.txt, requirements-dev.txt dan contoh environment.
- .htaccess:3 - Menghalang capaian terus repositori melalui Apache apabila arahan ini dibenarkan; backend perlu disediakan melalui WSGI/reverse proxy.
- tests/test_app.py:22 - Menambah ujian tingkah laku, kawalan akses, input tidak sah, penyimpanan, concurrency, rate limit dan kegagalan penyedia.
- tests/test_chat.cjs:7 - Menguji skrip klien sebenar menggunakan DOM test double dan test runner terbina dalam Node.
- tests/http_smoke.py:13 - Menyediakan smoke test HTTP Waitress yang belum dapat dijalankan kerana kegagalan semakan kelulusan.
- tests/benchmark_retrieval.py:12 - Menyediakan mikrobenchmark yang boleh diulang untuk membandingkan penggunaan semula matriks dengan transform korpus berulang.

- netlify.toml, package.json kosong dan templates/_redirects dibuang kerana aliran deployment statik tidak menjalankan Flask.
- index.html diganti dengan arahan menjalankan backend; templates/test_index.html berkongsi template utama supaya tiada salinan klien tidak selamat.
- README.md menerangkan startup, kontrak API baharu, konfigurasi, retensi, reverse proxy, logging dan batas penskalaan.
- .gitignore mengecualikan rahsia, DB, virtualenv dan artifak; .gitattributes menetapkan LF untuk fail sumber.
- CSV, JSON pengetahuan asal dan model .joblib lama tidak diubah. Runtime tidak memuatkan model .joblib.

## Keputusan pengesahan

- 33 ujian Python: lulus pada persekitaran virtualenv terasing dengan dependency runtime yang dikunci.
- 5 ujian JavaScript: lulus. Ini ujian dengan DOM test double, bukan pelayar sebenar.
- Python compileall dan node --check: lulus.
- pip check: tiada dependency rosak.
- pip-audit terhadap semua 19 versi runtime: tiada kerentanan diketahui dilaporkan oleh pangkalan data yang disemak. Ini bukan jaminan ketiadaan semua kelemahan.
- Korpus: 164 baris FAQ dan 51 lokasi.
- Mikrobenchmark 100 sampel: median carian dengan matriks tersimpan 0.557 ms, berbanding 5.06 ms dengan transform korpus berulang (nisbah 9.08). Pengukuran tempatan ini hanya meliputi komponen carian yang setara, bukan keseluruhan pipeline asal, latensi HTTP atau kapasiti produksi.

## Batas dan tindakan deployment

1. Batalkan dan gantikan kunci API yang terdedah. Working tree yang bersih daripada kunci tidak membersihkan sejarah Git/klon.
2. Jalankan di Python bertampal dan belakang HTTPS/reverse proxy. Tetapkan domain dipercayai, ACL DB, backup/retensi dan had trafik proxy.
3. SQLite digunakan untuk satu hos. Skala berbilang hos memerlukan storan dan rate limiter teragih serta load test representatif.
4. Fallback AI dimatikan secara lalai. Mengaktifkannya memerlukan konfigurasi kunci/model serta keputusan operasi tentang privasi dan ketepatan jawapan.
5. Smoke test HTTP belum dijalankan: automatic approval review gagal kerana kredit workspace habis. Arahan tidak dilaksanakan; ini kegagalan semakan, bukan keputusan bahawa ujian tersebut tidak selamat.
6. Tiada browser tersedia untuk pengesahan visual/end-to-end. Perlindungan Apache .htaccess dan konfigurasi reverse proxy juga belum diuji dalam deployment.

## UI remake - 19 Oktober 2026

- templates/index.html: susun atur HKL Guide, sidebar desktop/navigasi telefon, ilustrasi hospital, pintasan, dialog FAQ/lokasi dan penerangan privasi.
- static/style.css: sistem warna, tipografi, tema gelap, focus states, saiz kawalan sentuh, breakpoint responsif dan reduced motion.
- static/chat.js: sambungan panel kepada API, pemilihan soalan/lokasi, pengendalian respons lewat apabila panel ditukar/ditutup, pengiraan aksara serta peralihan welcome/sembang.
- tests/test_chat.cjs: 13 ujian klien termasuk reset yang berjaya/gagal, race conditions, hasil carian literal dan label tema aksesibel.
- tests/test_ui.py: 3 ujian HTML untuk binding kawalan, ID unik, pintasan FAQ sah, CSP/aset tempatan dan notis AI bersyarat.
- README.md: arahan UI dan status pengesahan dikemas kini.
- 33 ujian backend, 3 ujian HTML dan 13 ujian JavaScript lulus. Ujian HTTP Waitress kini lulus; kegagalan semakan kelulusan yang direkodkan dalam versi 17 Oktober tidak lagi menghalang ujian ini.
- Semakan visual melalui Chromium terasing kini lulus untuk desktop, tema gelap dan emulasi telefon (390 px / 320 px), termasuk dialog FAQ/lokasi, sembang, reset dan pemeriksaan overflow mendatar. Screenshot disertakan dalam artifacts. Tiada perubahan dependency runtime atau kontrak API backend.

## Vercel deployment - 19 Oktober 2026

- `vercel.json`: menambah konfigurasi Flask Function 30 saat dan mengecualikan ujian, artifak, virtualenv, SQLite tempatan serta model `.joblib` lama daripada bundle.
- `.python-version`: menetapkan Python 3.12 untuk wheel `numpy`, `scipy` dan `scikit-learn` yang konsisten.
- `public/static/`: menyalin aset UI supaya Vercel boleh melayan fail statik melalui CDN.
- `hkl_chatbot/__init__.py`: menggunakan `/tmp` hanya sebagai fallback apabila `VERCEL=1`, dan mengenali hostname Vercel automatik; domain tersuai masih perlu `HKL_TRUSTED_HOSTS`.
- README.md: menambah arahan CLI, environment variables, OpenRouter dan had SQLite stateless.
