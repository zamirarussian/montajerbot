"""Bitta rasmdan gapiruvchi personaj: jag' siljishi, og'iz ichi, ko'z yumish.

Koordinatalar zamira.png (680x712) uchun belgilangan.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "zamira.png")

MOUTH = (380, 418)      # lablar tutashgan joy markazi
MOUTH_W = 40            # og'izning yarim kengligi
CHIN_Y = 520            # iyak pastki chegarasi
EYES = [(312, 289), (456, 292)]
EYE_RX, EYE_RY = 31, 19
LEVELS = [0, 5, 10, 15, 20, 25]   # jag' siljishi (px)


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _jaw(img, d):
    if d == 0:
        return img.copy()
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx, my = MOUTH
    # lab burchaklari joyida qoladi (tor), iyak esa keng harakatlanadi
    sig = 34.0 + 0.9 * np.clip(yy - my, 0, None)
    wx = np.exp(-((xx - mx) / sig) ** 2)
    up = _smooth((yy - (my - 1)) / 4.0)                          # lablar orasidan pastga
    down = 1 - _smooth((yy - CHIN_Y) / 70.0)                     # bo'yinda so'nadi
    disp = d * wx * up * down
    sy = np.clip(yy - disp, 0, h - 1)
    y0 = np.floor(sy).astype(int); y1 = np.minimum(y0 + 1, h - 1); f = (sy - y0)[..., None]
    xi = xx.astype(int)
    out = a[y0, xi] * (1 - f) + a[y1, xi] * f
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def _mouth_inside(img, d):
    if d == 0:
        return img
    mx, my = MOUTH
    W, H = img.size
    S = 4
    lay = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    hw = 30 + d * 0.15
    top, bot = my - 1, my + d * 0.98
    box = [(mx - hw) * S, top * S, (mx + hw) * S, bot * S]
    dr.ellipse(box, fill=(92, 28, 36, 255))
    # tishlar (yuqori)
    th = min(5, 1.5 + d * 0.18)
    teeth = Image.new("L", lay.size, 0)
    ImageDraw.Draw(teeth).rectangle([(mx - hw) * S, top * S, (mx + hw) * S, (top + th) * S], fill=255)
    mask_e = Image.new("L", lay.size, 0); ImageDraw.Draw(mask_e).ellipse(box, fill=255)
    from PIL import ImageChops
    tm = ImageChops.multiply(teeth, mask_e)
    lay.paste((226, 214, 208, 255), (0, 0), tm)
    # til
    tg = Image.new("L", lay.size, 0)
    ImageDraw.Draw(tg).ellipse([(mx - hw * 0.6) * S, (bot - d * 0.45) * S, (mx + hw * 0.6) * S, (bot + d * 0.3) * S], fill=255)
    lay.paste((200, 92, 100, 255), (0, 0), ImageChops.multiply(tg, mask_e))
    lay = lay.resize((W, H), Image.LANCZOS)
    al = lay.getchannel("A").filter(ImageFilter.GaussianBlur(1.2))
    lay.putalpha(al)
    out = img.copy()
    out.alpha_composite(lay)
    return out


def _blink_overlay(img):
    """Ko'z ustiga teri rangidagi qovoq + kiprik chizig'i."""
    W, H = img.size
    S = 4
    lay = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    px = img.load()
    for ex, ey in EYES:
        skin = tuple(int(sum(px[ex + dx, ey + 34][c] for dx in (-8, 0, 8)) / 3) for c in range(3))
        skin = tuple(int(c * 0.94) for c in skin)
        dr.ellipse([(ex - EYE_RX) * S, (ey - EYE_RY - 2) * S, (ex + EYE_RX) * S, (ey + EYE_RY) * S], fill=skin + (255,))
        dr.arc([(ex - EYE_RX + 1) * S, (ey - 14) * S, (ex + EYE_RX - 1) * S, (ey + 10) * S], 12, 168,
               fill=(40, 26, 24, 255), width=3 * S)
    lay = lay.resize((W, H), Image.LANCZOS)
    lay.putalpha(lay.getchannel("A").filter(ImageFilter.GaussianBlur(2.2)))
    return lay


_cache = {}


def frames(scale=1.0):
    """{(daraja, ko'z_yumuq): RGBA} — oldindan tayyorlangan kadrlar."""
    if scale in _cache:
        return _cache[scale]
    base = Image.open(BASE).convert("RGBA")
    # chetidagi oq hoshiyani olib tashlash
    al = base.getchannel("A").filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(1))
    base.putalpha(al)
    blink = _blink_overlay(base)
    out = {}
    for i, d in enumerate(LEVELS):
        f = _mouth_inside(_jaw(base, d), d)
        fb = f.copy(); fb.alpha_composite(blink)
        if scale != 1.0:
            sz = (int(f.width * scale), int(f.height * scale))
            f = f.resize(sz, Image.LANCZOS); fb = fb.resize(sz, Image.LANCZOS)
        out[(i, 0)] = f
        out[(i, 1)] = fb
    _cache[scale] = out
    return out
