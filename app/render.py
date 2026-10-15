"""B uslubi (kartochka): chapda gapiruvchi personaj, o'ngda oq kartochkada gap + tarjima."""
import math, os, subprocess, tempfile

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .character import head, body

W, H, FPS = 1920, 1080, 25
TAIL = 2.5
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")

CREAM = (247, 241, 232)
RED = (200, 50, 60)
TEAL = (30, 111, 104)
TEAL_L = (214, 234, 229)
DARK = (34, 37, 43)
GRAY = (150, 143, 134)
LIGHT = (200, 193, 184)
DIM = (226, 221, 214)
MARK = (255, 214, 102)
WHITE = (255, 255, 255)
CARD = (860, 250, 1840, 830)

_fonts = {}


def F(name, size):
    k = (name, size)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _fonts[k]


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def fade(img, a):
    if a >= 1:
        return img
    img.putalpha(img.getchannel("A").point(lambda v: int(v * max(0, a))))
    return img


# ---------------- assets ----------------
_assets = {}


def assets():
    if _assets:
        return _assets
    sc = 0.9
    _assets["heads"] = {(m, b): head(m, b).resize((int(520 * sc), int(620 * sc)), Image.LANCZOS)
                        for m in range(5) for b in (0, 0.4, 1)}
    _assets["body"] = body().resize((700, 479), Image.LANCZOS)
    bg = Image.new("RGBA", (W, H), CREAM + (255,))
    bl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bl)
    bd.ellipse([1500, -260, 2200, 380], fill=(250, 214, 200, 140))
    bd.ellipse([-200, 760, 380, 1300], fill=(255, 226, 160, 110))
    bg.alpha_composite(bl.filter(ImageFilter.GaussianBlur(60)))
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([CARD[0], CARD[1] + 18, CARD[2], CARD[3] + 18], 44, fill=(90, 60, 40, 55))
    bg.alpha_composite(sh.filter(ImageFilter.GaussianBlur(26)))
    d = ImageDraw.Draw(bg)
    d.ellipse([440 - 330, 700 - 330, 440 + 330, 700 + 330], fill=TEAL_L)
    d.rounded_rectangle(CARD, 44, fill=WHITE)
    _assets["bg"] = bg
    return _assets


# ---------------- text ----------------
def layout(words, font, maxw, lh):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    sp = d.textlength(" ", font=font)
    lines, cur, cw = [], [], 0
    for wd in words:
        wl = d.textlength(wd, font=font)
        if cur and cw + sp + wl > maxw:
            lines.append((cur, cw)); cur, cw = [], 0
        cw = cw + (sp if cur else 0) + wl
        cur.append((wd, wl))
    if cur:
        lines.append((cur, cw))
    out = []
    for li, (ws, lw) in enumerate(lines):
        x = -lw / 2
        for wd, wl in ws:
            out.append((wd, x, li * lh, wl))
            x += wl + sp
    return out, len(lines)


def fit(text, maxw, sizes, maxlines):
    words = text.split()
    for s in sizes:
        f = F("InterDisplay-Bold.otf", s)
        lay, nl = layout(words, f, maxw, int(s * 1.22))
        if nl <= maxlines:
            break
    return f, lay, nl, int(f.size * 1.22)


def fit_line(text, maxw, sizes=(38, 34)):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    for s in sizes:
        f = F("Inter-Medium.otf", s)
        if d.textlength(text, font=f) <= maxw:
            return f, [text]
    # ikki qatorga bo'lish
    f = F("Inter-Medium.otf", sizes[-1])
    words = text.split(); half = len(words) // 2
    return f, [" ".join(words[:half]), " ".join(words[half:])]


def word_state(t, ev, n):
    """(aytilgan so'zlar, faol so'z, faol so'z ulushi)"""
    lo, hi = ev["span"] if ev.get("span") else (0, n)
    if ev.get("wt"):
        spoken, act, frac = set(), None, 0.0
        for i in range(lo, hi):
            s, e = ev["wt"][i]
            if t >= s - 0.04:
                spoken.add(i); act = i; frac = min(1, (t - s) / max(0.08, e - s))
        return spoken, act, frac, lo, hi
    return None


def word_state_prop(t, ev, lens, n):
    lo, hi = ev["span"] if ev.get("span") else (0, n)
    L = lens[lo:hi]; tot = sum(L)
    p = max(0, min(1, (t - ev["s"]) / max(0.05, ev["e"] - ev["s"] - 0.15)))
    acc = 0; spoken, act, frac = set(), None, 0.0
    for i, l in enumerate(L):
        if p * tot >= acc:
            spoken.add(lo + i); act = lo + i; frac = min(1, (p * tot - acc) / l)
        acc += l
    return spoken, act, frac, lo, hi


def draw_words(layer, lay, font, cx, top, colors, mark=None):
    d = ImageDraw.Draw(layer)
    if mark and mark[0] is not None:
        i, frac = mark
        wd, x, y, wl = lay[i]
        y0 = top + y + font.size * 0.62
        d.rounded_rectangle([cx + x - 8, y0, cx + x - 8 + (wl + 16) * ease(frac * 1.6), y0 + font.size * 0.42], 10, fill=MARK)
    for i, (wd, x, y, wl) in enumerate(lay):
        d.text((cx + x, top + y), wd, font=font, fill=colors[i])


# ---------------- render ----------------
def render(audio_path, plan, out_path, progress=None, only_frames=None):
    A = assets()
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", audio_path, "-ac", "1", "-ar", "48000", wav], check=True)
        a, sr = sf.read(wav, dtype="float32")
        audio_dur = len(a) / sr
        dur = audio_dur + TAIL
        nfr = int(dur * FPS) + 1
        hop = sr // FPS
        rms = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(a) else 0.0 for i in range(nfr)])
        nz = rms[rms > 0]
        ref = np.percentile(nz, 95) if len(nz) else 1
        env = np.clip(rms / max(ref, 1e-6), 0, 1.2)
        sm = env.copy()
        for i in range(1, len(sm)):
            sm[i] = max(env[i], sm[i - 1] * 0.55)

        scenes = plan["scenes"]
        intro_end = plan["intro_end"]
        end_start = min(plan["end_start"], audio_dur)
        day = plan["day"]
        cx = (CARD[0] + CARD[2]) / 2
        pre = {}
        for sc in scenes:
            pre[sc["n"]] = dict(
                sent=fit(sc["sent"], 860, (82, 74, 66, 60), 3),
                stem=fit(sc["stem"], 860, (92, 82, 72), 3),
                uz=fit_line(sc["uz"], 880), stem_uz=fit_line(sc["stem_uz"], 880))
        blinks = [b + 0.6 * math.sin(b * 7) for b in np.arange(1.7, dur, 3.9)]

        def scene_at(t):
            cur = None
            for sc in scenes:
                if t >= sc["start"]:
                    cur = sc
            return cur

        def uz_lines(layer, f_lines, y):
            f, lines = f_lines
            d = ImageDraw.Draw(layer)
            for k, ln in enumerate(lines):
                d.text((cx, y + k * (f.size + 10)), ln, font=f, fill=GRAY, anchor="mt")

        def card(t, sc):
            L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(L)
            d.text((CARD[0] + 56, CARD[1] + 50), str(sc["n"]), font=F("InterDisplay-Black.otf", 44), fill=RED)
            d.text((CARD[0] + 56 + d.textlength(str(sc["n"]), font=F("InterDisplay-Black.otf", 44)) + 10, CARD[1] + 66),
                   "/ 5", font=F("Inter-SemiBold.otf", 28), fill=LIGHT)
            evs = sc["ev"]
            exs = [e for e in evs if e["kind"] == "ex"]
            first_ex = exs[0]["s"] if exs else 1e9
            speaking = [e for e in evs if e["s"] <= t <= e["e"]]
            mix = ease((t - (first_ex - 0.6)) / 0.45) if t >= first_ex - 0.6 else 0.0
            p = pre[sc["n"]]
            if mix < 1:
                S = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                f, lay, nl, lh = p["stem"]
                block = nl * lh + 30 + len(p["stem_uz"][1]) * 48
                top = 560 - block / 2
                cols = [DARK] * len(lay)
                sp = [e for e in speaking if e["kind"] == "stem"]
                if sp:
                    lens = [w[3] for w in lay]
                    spoken, *_ = word_state_prop(t, sp[0], lens, len(lay))
                    cols = [RED if i in spoken else DARK for i in range(len(lay))]
                draw_words(S, lay, f, cx, top, cols)
                uz_lines(S, p["stem_uz"], top + nl * lh + 30)
                L.alpha_composite(fade(S, 1 - mix), (0, int(-30 * mix)))
            if mix > 0:
                E = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                f, lay, nl, lh = p["sent"]
                block = nl * lh + 30 + len(p["uz"][1]) * 48
                top = 560 - block / 2
                sp = [e for e in speaking if e["kind"] == "ex"]
                mark = None
                if sp:
                    st = word_state(t, sp[0], len(lay)) or word_state_prop(t, sp[0], [w[3] for w in lay], len(lay))
                    spoken, act, frac, lo, hi = st
                    cols = [DARK if i in spoken else (LIGHT if lo <= i < hi else DIM) for i in range(len(lay))]
                    mark = (act, frac)
                else:
                    cols = [DARK] * len(lay)
                draw_words(E, lay, f, cx, top, cols, mark)
                uz_lines(E, p["uz"], top + nl * lh + 30)
                L.alpha_composite(fade(E, mix), (0, int(30 * (1 - mix))))
            return L

        def intro(t):
            L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(L)
            f, lay, nl, lh = fit(plan["topic"], 860, (84, 74, 64), 3)
            top = 575 - nl * lh / 2
            d.text((cx, top - 50), f"{day}-KUN", font=F("InterDisplay-Black.otf", 40), fill=RED, anchor="mm")
            draw_words(L, lay, f, cx, top, [DARK] * len(lay))
            return L

        def outro(t):
            L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(L)
            d.text((cx, 480), "Barakalla!", font=F("InterDisplay-Black.otf", 96), fill=RED, anchor="mm")
            d.text((cx, 600), f"Ertaga: {day + 1}-kun", font=F("Inter-Medium.otf", 40), fill=GRAY, anchor="mm")
            return L

        def character(fr, i, t):
            amp = float(sm[min(i, len(sm) - 1)])
            m = int(min(4, 1 + amp * 3.6)) if amp > 0.12 else 0
            bk = 0
            for b in blinks:
                if b <= t <= b + 0.18:
                    ph = (t - b) / 0.18
                    bk = 1 if 0.25 < ph < 0.75 else 0.4
                    break
            hd = A["heads"][(m, bk)]
            cxp = 440
            d = ImageDraw.Draw(fr)
            rr = 348 + amp * 22
            d.ellipse([cxp - rr, 700 - rr, cxp + rr, 700 + rr], outline=(178, 214, 206), width=5)
            breath = math.sin(t * 2.1) * 4
            by = int(640 + breath * 0.5)
            fr.alpha_composite(A["body"], (cxp - 350, by))
            ang = math.sin(t * 1.3) * 2.2 + amp * 2.5 * math.sin(t * 6)
            hr = hd.rotate(ang, resample=Image.BICUBIC, center=(hd.width / 2, hd.height * 0.8))
            fr.alpha_composite(hr, (cxp - hd.width // 2, int(by - 477 + 18 - amp * 8 + breath)))

        def frame(i):
            t = i / FPS
            fr = A["bg"].copy()
            character(fr, i, t)
            if t < intro_end:
                k = ease(t / 0.6) * (1 - ease((t - (intro_end - 0.35)) / 0.35))
                fr.alpha_composite(fade(intro(t), k), (0, int(30 * (1 - ease(t / 0.6)))))
            elif t >= end_start:
                k = ease((t - end_start) / 0.6)
                fr.alpha_composite(fade(outro(t), k), (0, int(30 * (1 - k))))
            else:
                sc = scene_at(t)
                if sc:
                    nxt = [s["start"] for s in scenes if s["start"] > sc["start"]] + [end_start]
                    k = ease((t - sc["start"]) / 0.5)
                    a_ = k * (1 - ease((t - (nxt[0] - 0.3)) / 0.3))
                    fr.alpha_composite(fade(card(t, sc), a_), (int(50 * (1 - k)), 0))
            return fr.convert("RGB")

        if only_frames:
            for tt in only_frames:
                frame(int(tt * FPS)).save(f"{out_path}_{tt:.1f}.png")
            return out_path

        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-", "-i", wav, "-af", f"apad=pad_dur={TAIL}", "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", os.environ.get("X264_PRESET", "veryfast"), "-crf", "21",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-t", f"{dur:.2f}",
               "-movflags", "+faststart", out_path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(nfr):
                p.stdin.write(frame(i).tobytes())
                if progress and i % 125 == 0:
                    progress(i / nfr)
        finally:
            p.stdin.close()
            p.wait()
        if p.returncode != 0:
            raise RuntimeError("ffmpeg xatosi")
    return out_path

