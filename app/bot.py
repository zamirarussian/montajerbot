"""Shadowing video bot: audio + kun raqami -> tayyor video."""
import asyncio, json, logging, os, re, tempfile, uuid

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from .align import build_plan, summary, transcribe_groq, utterances_from_words
from .render import render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("shadowing")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHRASES = json.load(open(os.path.join(BASE, "data", "phrases.json"), encoding="utf-8"))
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_IDS", "").replace(" ", "").split(",") if x}
AUTO = os.environ.get("AUTO_RENDER", "0") == "1"
RENDER_LOCK = asyncio.Lock()
JOBS = {}  # job_id -> dict(path, plan, chat_id)
WORK = tempfile.mkdtemp(prefix="shadow_")

HELP = (
    "Salom! Menga shadowing audiosini yuboring, izohiga kun raqamini yozing.\n\n"
    "Misol: audio + izoh <b>31</b>  (B1–B2 daraja)\n"
    "A1–A2 uchun: <b>31 a1</b>\n\n"
    "Audio tuzilishi: ibora qolipi, keyin misol gap bir necha marta. "
    "Men iboralarni topib, ro'yxatini ko'rsataman, siz tasdiqlaysiz, video 3–6 daqiqada tayyor bo'ladi."
)


def allowed(update: Update):
    return not ALLOWED or (update.effective_user and update.effective_user.id in ALLOWED)


def parse_caption(text):
    if not text:
        return None, None
    m = re.search(r"\d{1,3}", text)
    if not m:
        return None, None
    level = "a1" if re.search(r"\ba[12]\b", text.lower()) else "b1"
    return int(m.group()), level


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_html(HELP)


async def on_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    msg = update.message
    media = msg.audio or msg.voice or msg.document
    if msg.document and not (msg.document.mime_type or "").startswith(("audio", "video")):
        await msg.reply_text("Bu audio fayl emasga o'xshaydi. m4a, mp3, ogg yoki voice yuboring.")
        return
    if media.file_size and media.file_size > 20 * 1024 * 1024:
        await msg.reply_text("Fayl 20 MB dan katta, Telegram botlar uni yuklay olmaydi. Siqib yoki mp3 qilib yuboring.")
        return
    day, level = parse_caption(msg.caption)
    ctx.user_data["pending"] = dict(file_id=media.file_id, name=getattr(media, "file_name", None) or "audio.m4a")
    if day is None:
        await msg.reply_text("Qaysi kun? Raqam yozing, masalan: 31  (A1–A2 bo'lsa: 31 a1)")
        return
    await process(update, ctx, day, level)


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    if "pending" not in ctx.user_data:
        await update.message.reply_html(HELP)
        return
    day, level = parse_caption(update.message.text)
    if day is None:
        await update.message.reply_text("Kun raqamini yozing, masalan: 31")
        return
    await process(update, ctx, day, level)


async def process(update: Update, ctx: ContextTypes.DEFAULT_TYPE, day, level):
    chat = update.effective_chat
    pend = ctx.user_data.pop("pending")
    data = PHRASES.get(level, {}).get(str(day))
    if not data:
        await chat.send_message(f"{day}-kun uchun iboralar jadvalda yo'q ({level.upper()}). "
                                f"Dam olish kuni bo'lishi mumkin (7, 14, 21, 28… kunlar).")
        return
    status = await chat.send_message(f"⏳ {day}-kun: audio tahlil qilinyapti…")
    job = uuid.uuid4().hex[:10]
    ext = os.path.splitext(pend["name"])[1] or ".m4a"
    path = os.path.join(WORK, f"{job}{ext}")
    f = await ctx.bot.get_file(pend["file_id"])
    await f.download_to_drive(path)
    try:
        words = await asyncio.to_thread(transcribe_groq, path)
        dur = await asyncio.to_thread(audio_duration, path)
        plan = build_plan(utterances_from_words(words), data, day, dur)
    except Exception as e:
        log.exception("align")
        await status.edit_text(f"❌ Audioni tahlil qilib bo'lmadi: {e}")
        return
    ok, txt = summary(plan)
    JOBS[job] = dict(path=path, plan=plan, chat_id=chat.id)
    head = f"<b>{day}-kun · {plan['topic']}</b>\n\n"
    if ok and AUTO:
        await status.edit_text(head + txt, parse_mode="HTML")
        await run_render(ctx, job, chat.id)
        return
    tail = "\n\nVideoni yasaymi?" if ok else "\n\n⚠️ Ba'zi iboralar topilmadi. Baribir yasaymi?"
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yasash", callback_data=f"go:{job}"),
                                InlineKeyboardButton("❌ Bekor", callback_data=f"no:{job}")]])
    await status.edit_text(head + txt + tail, parse_mode="HTML", reply_markup=kb)


def audio_duration(path):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    act, job = q.data.split(":", 1)
    if job not in JOBS:
        await q.edit_message_reply_markup(None)
        await q.message.reply_text("Bu so'rov eskirgan, audioni qayta yuboring.")
        return
    await q.edit_message_reply_markup(None)
    if act == "no":
        j = JOBS.pop(job)
        try:
            os.remove(j["path"])
        except OSError:
            pass
        await q.message.reply_text("Bekor qilindi.")
        return
    await run_render(ctx, job, q.message.chat_id)


async def run_render(ctx, job, chat_id):
    j = JOBS[job]
    note = await ctx.bot.send_message(chat_id, "🎬 Navbatga qo'yildi…" if RENDER_LOCK.locked() else "🎬 Video yig'ilyapti…")
    async with RENDER_LOCK:
        out = os.path.join(WORK, f"{job}.mp4")
        loop = asyncio.get_running_loop()
        last = {"p": -1}

        def progress(p):
            pct = int(p * 100) // 20 * 20
            if pct != last["p"]:
                last["p"] = pct
                asyncio.run_coroutine_threadsafe(safe_edit(note, f"🎬 Video yig'ilyapti… {pct}%"), loop)

        try:
            await asyncio.to_thread(render, j["path"], j["plan"], out, progress)
            plan = j["plan"]
            cap = f"{plan['day']}-kun · {plan['topic']}"
            with open(out, "rb") as fh:
                if os.path.getsize(out) < 49 * 1024 * 1024:
                    await ctx.bot.send_video(chat_id, fh, caption=cap, supports_streaming=True,
                                             width=1920, height=1080, read_timeout=300, write_timeout=300)
                else:
                    await ctx.bot.send_document(chat_id, fh, caption=cap, read_timeout=300, write_timeout=300)
            await safe_edit(note, "✅ Tayyor")
        except Exception as e:
            log.exception("render")
            await safe_edit(note, f"❌ Xatolik: {e}")
        finally:
            for p in (j["path"], out):
                try:
                    os.remove(p)
                except OSError:
                    pass
            JOBS.pop(job, None)


async def safe_edit(msg, text):
    try:
        await msg.edit_text(text)
    except Exception:
        pass


def main():
    app = (Application.builder().token(os.environ["BOT_TOKEN"])
           .read_timeout(60).write_timeout(300).concurrent_updates(True).build())
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE | filters.Document.ALL, on_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(CallbackQueryHandler(on_button))
    log.info("bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
