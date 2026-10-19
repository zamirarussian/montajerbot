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


def stem_prefix(stem):
    """Qolipning '…' dan oldingi birinchi 1-2 so'zi (gap boshlanishini tanish uchun)."""
    head = re.split(r"[…]|\.\.\.", stem)[0]
    toks = [ntok(w) for w in head.split() if ntok(w)]
    return toks[:2]


def find_prefix(utoks, pref):
    if not pref:
        return []
    n = len(pref)
    return [i for i in range(len(utoks) - n + 1)
            if all(utoks[i + j][:4] == pref[j][:4] for j in range(n))]


def _sub_utt(u, a, b):
    """u ning a..b so'zlaridan iborat bo'lak; vaqt bo'g'in soniga qarab."""
    ws = u["words"]
    syl = [_syl(w["w"]) for w in ws]
    tot = sum(syl) or 1
    dur = u["e"] - u["s"]
    s_ = u["s"] + dur * sum(syl[:a]) / tot
    e_ = u["s"] + dur * sum(syl[:b]) / tot
    part = ws[a:b]
    return dict(s=round(s_, 3), e=round(e_, 3), words=part, text=" ".join(w["w"] for w in part))


def _merge(a, b):
    return dict(s=a["s"], e=b["e"], words=a["words"] + b["words"],
                text=(a["text"] + " " + b["text"]).strip())


def merge_fragments(utts, score):
    """Bitta so'zli mayda bo'laklarni ("Мне" / "45" / "лет") qo'shni gapga qo'shish."""
    def frag(u):
        return len(u["words"]) <= 1 and u["e"] - u["s"] < 0.9
    # 1) ketma-ket bo'laklarni birlashtirish
    out = []
    for u in utts:
        if out and frag(u) and frag(out[-1]) and u["s"] - out[-1]["e"] < 0.8 and u["words"] and out[-1]["words"]:
            out[-1] = _merge(out[-1], u)
        else:
            out.append(dict(u))
    # 2) yolg'iz bo'lakni qaysi qo'shniga qo'shish foydaliroq bo'lsa, o'shanga
    i = 0
    while i < len(out):
        u = out[i]
        if frag(u) and u["words"]:
            best, gain = None, 0.02
            if i > 0 and u["s"] - out[i - 1]["e"] < 0.8:
                g = score(_merge(out[i - 1], u)["text"]) - score(out[i - 1]["text"])
                if g > gain:
                    best, gain = "prev", g
            if i + 1 < len(out) and out[i + 1]["s"] - u["e"] < 0.8:
                g = score(_merge(u, out[i + 1])["text"]) - score(out[i + 1]["text"])
                if g > gain:
                    best, gain = "next", g
            if best == "prev":
                out[i - 1] = _merge(out[i - 1], u); out.pop(i); continue
            if best == "next":
                out[i + 1] = _merge(u, out[i + 1]); out.pop(i); continue
        i += 1
    return out


def display_text(words):
    t = " ".join(w["w"].strip() for w in words).strip()
    t = re.sub(r"\s+([,.!?…])", r"\1", t)
    if t:
        t = t[0].upper() + t[1:]
        if t[-1] not in ".!?…":
            t += "."
    return t


def build_plan(utts, day_data, day, audio_dur, min_score=0.5):
    phrases = []
    for p in day_data["phrases"]:
        sent = clean_display(p["sent"])
        words = sent.split()
        phrases.append(dict(stem=clean_display(p["stem"]), stem_uz=clean_display(p["stem_uz"]),
                            sent=sent, uz=clean_display(p["uz"]), words=words,
                            toks=[ntok(w) or w for w in words], ev=[]))
    for ph in phrases:
        ph["pref"] = stem_prefix(ph["stem"])
        ph["stoks"] = [ntok(w) for w in ph["stem"].split() if ntok(w)]

    def score(text):
        t = norm(text)
        return max([0] + [max(_ratio(t, norm(ph["stem"])), _ratio(t, norm(ph["sent"]))) for ph in phrases])

    queue = merge_fragments(list(utts), score)
    cur = 0
    last_ex = None  # (k, davomiylik) oxirgi to'liq misol
    while queue:
        u = queue.pop(0)
        # bir nafasda aytilgan ikki gap ("Я работаю. Я работаю в школе.", "Мне 31 год Мне 22 года")
        utoks0 = [ntok(w["w"]) for w in u["words"]]
        cuts = []
        for k in range(cur, min(cur + 2, len(phrases))):
            occ = find_prefix(utoks0, phrases[k]["pref"])
            if len(occ) >= 2:
                cuts = occ[1:]
                break
        if cuts:
            bounds = [0] + cuts + [len(u["words"])]
            parts = [_sub_utt(u, a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
            queue = parts + queue
            continue
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
        variant = None
        if un and (best is None or best[0] < 0.8):
            # qolip bilan boshlanadigan gap: qolipning o'zi yoki jadvaldan boshqa misol
            utoks1 = [ntok(w["w"]) for w in u["words"]]
            for k in range(cur, min(cur + 2, len(phrases))):
                ph = phrases[k]
                occ = find_prefix(utoks1, ph["pref"])
                if not occ:
                    continue
                p0 = occ[0]
                sk = {t[:4] for t in ph["stoks"]}
                extra = [t for t in utoks1[p0:] if t[:4] not in sk]
                if not extra:
                    best = (1.0, k, "stem")
                else:
                    tail = " ".join(utoks1[p0:])
                    part = utoks1[p0:]
                    is_part = len(_tok_match(part, ph["toks"])) >= 0.9 * len(part) and not any(
                        t.isdigit() and t not in ph["toks"] for t in part)
                    if is_part or _ratio(tail, norm(ph["sent"])) >= 0.75:
                        best = (1.0, k, "ex")
                    else:
                        best = (1.0, k, "ex"); variant = p0
                break
        if best is not None and un:
            k0 = best[1]; ph0 = phrases[k0]
            utoks2 = [ntok(w["w"]) for w in u["words"] if ntok(w["w"])]
            sent4 = [t[:4] for t in ph0["toks"]]
            # so'zlari jadvaldagi gapniki, lekin tartibi boshqa ("Я в школе работаю")
            if best[2] == "stem" and len(utoks2) > len(ph0["stoks"]):
                cov = sum(1 for t in set(sent4) if t in {x[:4] for x in utoks2}) / max(1, len(set(sent4)))
                if cov >= 0.85:
                    best = (best[0], k0, "ex")
                    if len(_tok_match(utoks2, ph0["toks"])) < 0.85 * len(sent4):
                        variant = 0
            # raqami boshqa ("Мне 45 лет" jadvalda "Мне 25 лет")
            if best[2] == "ex" and variant is None:
                nums_u = [t for t in utoks2 if t.isdigit()]
                if nums_u and any(t not in ph0["toks"] for t in nums_u):
                    occ = find_prefix(utoks2, ph0["pref"])
                    variant = occ[0] if occ else 0
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
            # (Whisper vaqtiga ishonmaymiz: bo'g'in soni bo'yicha bo'lamiz)
            stem_words = [w for w in ph["stem"].split() if ntok(w)]
            stoks = [ntok(w) for w in stem_words]
            pairs = _tok_match(utoks, stoks)
            if pairs:
                ui = min(pp[0] for pp in pairs)
                if ui > 0:
                    allw = [w["w"] for w in u["words"]]
                    frac = sum(_syl(w) for w in allw[:ui]) / sum(_syl(w) for w in allw)
                    ev["s"] = round(u["s"] + dur * frac - 0.05, 3)
            else:
                # so'zlar tanilmadi: qolipning taxminiy davomiyligidan foydalanamiz
                est = 0.35 + 0.19 * sum(_syl(w) for w in stem_words)
                if dur > est * 1.6:
                    ev["s"] = round(u["e"] - est, 3)
        elif variant is not None:
            # diktor jadvaldagidan boshqa misol aytdi: ekranda aytilganini ko'rsatamiz
            if variant > 0:
                sub = _sub_utt(u, variant, len(u["words"]))
                ev["s"] = sub["s"]
            vw = u["words"][variant:]
            ev["text"] = display_text(vw)
            ev["uz"] = ph["stem_uz"]
            words_v = ev["text"].split()
            ev["wt"] = prop_times(words_v, 0, len(words_v), ev["s"], ev["e"])
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

    # qolip va misol bir xil bo'lsa ("Счёт, пожалуйста."): birinchisi qolip, qolganlari misol
    for ph in phrases:
        if ph["ev"] and norm(ph["stem"]) == norm(ph["sent"]):
            ph["ev"].sort(key=lambda e: e["s"])
            for i, ev in enumerate(ph["ev"]):
                if ev.get("text"):
                    continue
                if i == 0:
                    ev.update(kind="stem", span=None, wt=None)
                else:
                    ev.update(kind="ex", span=None)
                    ev["wt"] = prop_times(ph["words"], 0, len(ph["words"]), ev["s"], ev["e"])

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
        var = [e["text"] for e in s["ev"] if e.get("text")]
        if var:
            lines.append("      boshqa misollar: " + "; ".join(dict.fromkeys(var)))
    return ok, "\n".join(lines)
