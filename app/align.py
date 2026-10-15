"""Audio -> so'zlar (Groq Whisper) -> kunning 5 ta iborasiga moslash -> video rejasi."""
import os, re, json, subprocess, tempfile
from difflib import SequenceMatcher

import requests

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GAP = 0.45  # shu soniyadan uzun pauza = yangi gap


# ---------------- matn tozalash ----------------
def clean_display(text):
    """Ekranda ko'rsatiladigan matn: so'zlar orasidagi uzun tire o'rniga vergul."""
    t = re.sub(r"\s+[—–]\s+", ", ", text)
    t = re.sub(r",\s*,", ",", t)
    return t.strip()


def norm(s):
    s = s.lower().replace("ё", "е")
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ntok(w):
    return norm(w)


# ---------------- transkripsiya ----------------
def to_flac16k(src, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", "16000", dst], check=True)


def transcribe_groq(audio_path, api_key=None, model="whisper-large-v3"):
    api_key = api_key or os.environ["GROQ_API_KEY"]
    with tempfile.TemporaryDirectory() as td:
        fl = os.path.join(td, "a.flac")
        to_flac16k(audio_path, fl)
        with open(fl, "rb") as f:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("a.flac", f, "audio/flac")},
                data=[("model", model), ("language", "ru"), ("response_format", "verbose_json"),
                      ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment"),
                      ("temperature", "0")],
                timeout=300,
            )
    r.raise_for_status()
    j = r.json()
    words = [dict(w=w["word"], s=float(w["start"]), e=float(w["end"])) for w in j.get("words", [])]
    return words


def utterances_from_words(words, gap=GAP):
    utts, cur = [], []
    for w in words:
        if cur and w["s"] - cur[-1]["e"] > gap:
            utts.append(cur); cur = []
        if norm(w["w"]):
            cur.append(w)
    if cur:
        utts.append(cur)
    return [dict(s=u[0]["s"], e=u[-1]["e"], words=u, text=" ".join(x["w"] for x in u)) for u in utts]


def load_pcm16k(path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0


def speech_intervals(path, min_sil=0.4, min_speech=0.25):
    """Audio energiyasi bo'yicha gaplar chegarasi (Whisper vaqtlaridan ancha aniq)."""
    import numpy as np
    a = load_pcm16k(path)
    hop = 160  # 10 ms
    n = len(a) // hop
    fr = a[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 98)
    thr = max(floor + 12, peak - 32)
    sp = db > thr
    segs, i = [], 0
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]:
                j += 1
            segs.append([i * 0.01, j * 0.01]); i = j
        else:
            i += 1
    merged = []
    for s_, e_ in segs:
        if merged and s_ - merged[-1][1] < min_sil:
            merged[-1][1] = e_
        else:
            merged.append([s_, e_])
    return [(round(s_, 3), round(e_, 3)) for s_, e_ in merged if e_ - s_ >= min_speech]


def utterances_from_intervals(words, intervals):
    """Har bir audio-gapga Whisper so'zlarini (o'rta nuqtasi bo'yicha) biriktirish."""
    utts = [dict(s=s_, e=e_, words=[]) for s_, e_ in intervals]
    for w in words:
        if not norm(w["w"]) or not utts:
            continue
        m = (w["s"] + w["e"]) / 2
        best = min(utts, key=lambda u: 0 if u["s"] - 0.15 <= m <= u["e"] + 0.15 else min(abs(m - u["s"]), abs(m - u["e"])))
        if best["s"] - 0.6 <= m <= best["e"] + 0.6:
            best["words"].append(w)
    for u in utts:
        u["text"] = " ".join(x["w"] for x in u["words"])
    return utts


# ---------------- moslash ----------------
def _ratio(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _tok_match(utt_toks, sent_toks):
    """utterance so'zlarini gap so'zlariga bog'lash (birinchi 5 harf bo'yicha)."""
    a = [t[:4] for t in utt_toks]
    b = [t[:4] for t in sent_toks]
    pairs = []
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            pairs.append((blk.a + k, blk.b + k))
    return pairs


VOWELS = set("аеёиоуыэюяaeiou")


def _syl(w):
    """so'z og'irligi: unli harflar soni (bo'g'in) + tinish belgisidan keyingi pauza."""
    n = sum(1 for c in w.lower() if c in VOWELS)
    return max(1, n) + (0.6 if w[-1:] in ",.!?…;:" else 0)


def prop_times(words, lo, hi, s, e):
    """[s, e] oralig'ini so'zlarga bo'g'in soniga qarab taqsimlash."""
    wt = [None] * len(words)
    ws = [_syl(words[i]) for i in range(lo, hi)]
    tot = sum(ws); t = s
    for k, i in enumerate(range(lo, hi)):
        d = (e - s) * ws[k] / tot
        wt[i] = [round(t, 3), round(t + d, 3)]
        t += d
    return wt


def build_plan(utts, day_data, day, audio_dur, min_score=0.5):
    phrases = []
    for p in day_data["phrases"]:
        sent = clean_display(p["sent"])
        words = sent.split()
        phrases.append(dict(stem=clean_display(p["stem"]), stem_uz=clean_display(p["stem_uz"]),
                            sent=sent, uz=clean_display(p["uz"]), words=words,
                            toks=[ntok(w) or w for w in words], ev=[]))
    cur = 0
    last_ex = None  # (k, davomiylik) oxirgi to'liq misol
    for u in utts:
        un = norm(u["text"])
        dur = u["e"] - u["s"]
        best = None
        if un:
            for k in range(cur, min(cur + 2, len(phrases))):
                ph = phrases[k]
                rs = _ratio(un, norm(ph["stem"]))
                rf = _ratio(un, norm(ph["sent"]))
                sc, kind = (rs, "stem") if rs >= rf else (rf, "ex")
                if k > cur:
                    sc -= 0.05
                if best is None or sc > best[0]:
                    best = (sc, k, kind)
        if (best is None or best[0] < min_score) and un:
            # gapning bir bo'lagi ("на автобусе"): so'zlarning ko'pi joriy misolda bor
            ut = [ntok(w["w"]) for w in u["words"] if ntok(w["w"])]
            ph = phrases[cur]
            if ut and len(_tok_match(ut, ph["toks"])) / len(ut) >= 0.75:
                best = (1.0, cur, "ex")
        if best is None or best[0] < min_score:
            # Whisper takrorni tashlab ketgan bo'lishi mumkin: oldingi misolga
            # davomiyligi o'xshash bo'lsa, uni o'sha misolning takrori deb olamiz.
            if last_ex and last_ex[0] == cur and 0.6 <= dur / last_ex[1] <= 1.7:
                best = (1.0, cur, "ex")
            else:
                continue
        _, k, kind = best
        cur = k
        ph = phrases[k]
        n = len(ph["words"])
        ev = dict(s=round(u["s"], 3), e=round(u["e"], 3), kind=kind, span=None, wt=None)
        utoks = [ntok(w["w"]) for w in u["words"]]
        if kind == "stem":
            # oldidagi ortiqcha so'zlarni (mavzu nomi, "второе") kesib tashlash
            stoks = [ntok(w) for w in ph["stem"].split() if ntok(w)]
            pairs = _tok_match(utoks, stoks)
            if pairs:
                ui = min(pp[0] for pp in pairs)
                if ui > 0:
                    ev["s"] = round(max(u["s"], u["words"][ui]["s"] - 0.1), 3)
        else:
            lo, hi = 0, n
            pairs = _tok_match(utoks, ph["toks"]) if utoks else []
            if pairs:
                a = min(pp[1] for pp in pairs); b = max(pp[1] for pp in pairs) + 1
                if b - a < 0.85 * n:
                    lo, hi = a, b
                    ev["span"] = [lo, hi]
            if ev["span"] is None:
                last_ex = (k, dur)
            ev["wt"] = prop_times(ph["words"], lo, hi, ev["s"], ev["e"])
        ph["ev"].append(ev)

    # sahnalar boshlanishi
    found = [i for i, p in enumerate(phrases) if p["ev"]]
    first_t = phrases[found[0]]["ev"][0]["s"] if found else 0
    intro_end = round(first_t - 0.4, 2) if first_t > 1.6 else 0.0
    scenes, prev_end = [], intro_end
    for i, ph in enumerate(phrases):
        if not ph["ev"]:
            continue
        start = max(prev_end, ph["ev"][0]["s"] - 1.2) if scenes else intro_end
        if scenes:
            start = max(scenes[-1]["ev"][-1]["e"] + 0.35, min(start, ph["ev"][0]["s"] - 0.4))
        scenes.append(dict(n=i + 1, start=round(start, 2), stem=ph["stem"], stem_uz=ph["stem_uz"],
                           sent=ph["sent"], uz=ph["uz"], ev=ph["ev"]))
        prev_end = ph["ev"][-1]["e"]
    last_e = scenes[-1]["ev"][-1]["e"] if scenes else audio_dur
    plan = dict(day=day, topic=clean_display(day_data["topic"]), intro_end=round(intro_end, 2),
                end_start=round(min(audio_dur, last_e + 0.6), 2), audio_dur=audio_dur, scenes=scenes)
    return plan


def summary(plan):
    lines = []
    got = {s["n"]: s for s in plan["scenes"]}
    ok = True
    for n in range(1, 6):
        s = got.get(n)
        if not s:
            lines.append(f"❌ {n}-ibora topilmadi"); ok = False; continue
        st = sum(1 for e in s["ev"] if e["kind"] == "stem")
        ex = sum(1 for e in s["ev"] if e["kind"] == "ex")
        mark = "✅" if ex >= 1 else "⚠️"
        if ex < 1:
            ok = False
        lines.append(f"{mark} {n}. {s['sent']}  (qolip: {st}, misol: {ex} marta, {s['ev'][0]['s']:.0f}s dan)")
    return ok, "\n".join(lines)
