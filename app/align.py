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


def build_plan(utts, day_data, day, audio_dur, min_score=0.5):
    phrases = []
    for p in day_data["phrases"]:
        sent = clean_display(p["sent"])
        words = sent.split()
        phrases.append(dict(stem=clean_display(p["stem"]), stem_uz=clean_display(p["stem_uz"]),
                            sent=sent, uz=clean_display(p["uz"]), words=words,
                            toks=[ntok(w) or w for w in words], ev=[]))
    cur = 0
    for u in utts:
        un = norm(u["text"])
        if not un:
            continue
        best = None
        for k in range(cur, min(cur + 2, len(phrases))):
            ph = phrases[k]
            rs = _ratio(un, norm(ph["stem"]))
            rf = _ratio(un, norm(ph["sent"]))
            sc, kind = (rs, "stem") if rs >= rf else (rf, "ex")
            # keyingi iboraga o'tish uchun biroz yuqoriroq ishonch talab qilamiz
            if k > cur:
                sc -= 0.05
            if best is None or sc > best[0]:
                best = (sc, k, kind)
        if best is None or best[0] < min_score:
            continue
        _, k, kind = best
        cur = k
        ph = phrases[k]
        ev = dict(s=u["s"], e=u["e"], kind=kind, span=None, wt=None)
        if kind == "ex":
            utoks = [ntok(w["w"]) for w in u["words"]]
            pairs = _tok_match(utoks, ph["toks"])
            if pairs:
                lo = min(p[1] for p in pairs); hi = max(p[1] for p in pairs) + 1
                n = len(ph["toks"])
                if hi - lo < 0.85 * n:
                    ev["span"] = [lo, hi]
                else:
                    lo, hi = 0, n
                # har bir gap so'zining boshlanish/tugash vaqti
                wt = [None] * n
                for ui, si in pairs:
                    wt[si] = [u["words"][ui]["s"], u["words"][ui]["e"]]
                known = [i for i in range(lo, hi) if wt[i]]
                if len(known) >= 2:
                    for i in range(lo, hi):
                        if wt[i] is None:
                            before = [j for j in known if j < i]
                            after = [j for j in known if j > i]
                            a = wt[before[-1]][1] if before else u["s"]
                            b = wt[after[0]][0] if after else u["e"]
                            wt[i] = [a, b]
                    ev["wt"] = wt
        ph["ev"].append(ev)

    # sahnalar boshlanishi
    found = [i for i, p in enumerate(phrases) if p["ev"]]
    first_t = phrases[found[0]]["ev"][0]["s"] if found else 0
    intro_end = max(2.5, first_t - 0.7) if first_t > 2.0 else 0.0
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
