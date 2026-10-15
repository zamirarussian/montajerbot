# Shadowing video bot

Telegram bot: audio + kun raqami → tayyor shadowing video (1920×1080, B uslubi: personaj + kartochka).

## Qanday ishlaydi
1. Botga audio yuborasiz (m4a, mp3, ogg, voice), izohga kun raqamini yozasiz: `31` yoki `31 a1`.
2. Bot audioni Groq Whisper orqali matnga o'giradi (har bir so'z vaqti bilan).
3. `data/phrases.json` dan o'sha kunning 5 ta iborasini olib, audio ichidan topadi va ro'yxatini ko'rsatadi.
4. "✅ Yasash" bosilsa, video yig'iladi va chatga yuboriladi.

## Railway'ga joylash
1. Bu papkani GitHub'da yangi repo qilib yuklang (masalan `zamirarussian/shadowing-bot`).
2. Railway → New Project → Deploy from GitHub repo → shu repo. Railway `Dockerfile` ni o'zi topadi.
3. Variables bo'limiga qo'shing:
   - `BOT_TOKEN`: @BotFather'dan yangi bot tokeni
   - `GROQ_API_KEY`: Groq kaliti (Suhbatdosh botdagi kalitni ishlatsa ham bo'ladi)
   - `ALLOWED_IDS`: botdan foydalanadiganlarning Telegram ID'lari, vergul bilan (@userinfobot dan bilib olinadi)
   - `AUTO_RENDER`: `0` (har safar tasdiq so'raydi) yoki `1` (hammasi topilsa, darhol yasaydi)
4. Deploy. Loglarda `bot started` chiqsa, tayyor.

Resurs: 1 vCPU / 1 GB RAM yetadi. 2 daqiqalik video taxminan 3–6 daqiqada yig'iladi. Bir vaqtda faqat bitta video yig'iladi, qolganlari navbatda turadi.

## Iboralar jadvali
`data/phrases.json`: `{ "b1": { "30": { "lesson": 9, "topic": "...", "phrases": [ {stem, stem_uz, sent, uz} ×5 ] } }, "a1": {...} }`

Kunlar hisobi: har 7-kun dam olish (7, 14, 21, 28…), har bir dars 3 kun, har kunga 5 ta ibora.
Jadvalni o'zgartirsangiz, faylni tahrirlab, GitHub'ga qayta yuklang. Railway o'zi qayta deploy qiladi.

Ekranda so'zlar orasidagi uzun tire (—) avtomatik vergulga almashtiriladi.

## Audio yozish qoidasi (aniq ishlashi uchun)
- Ibora qolipi, keyin misol gap 3–4 marta, gaplar orasida ~1 soniya pauza.
- Iboralar jadvaldagi tartibda (1 → 5).
- Gapni jadvaldagidek o'qing. Kichik farqlar (изменил/изменила) muammo emas.

## Fayllar
- `app/bot.py`: Telegram bot
- `app/align.py`: Groq transkripsiya va iboralarni moslash
- `app/render.py`: video yig'ish (Pillow + ffmpeg)
- `app/character.py`: personaj chizmasi
- `fonts/`: Inter shrifti (SIL Open Font License), Docker build vaqtida avtomatik yuklanadi
