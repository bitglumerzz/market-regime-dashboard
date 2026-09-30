"""Звук для промо: phonk-house 120 BPM + саунд-дизайн, синхронно с кадром.

    python3 sound.py full [hard|deep|funk]   -> promo_audio[_стиль].wav (48.3 с, под promo.mp4)
    python3 sound.py ad   [hard|deep|funk]   -> ad_audio[_стиль].wav    (18.6 с, под рекламную нарезку)

Стили:
    hard — phonk-house 120 BPM: дисторшн-808, фонк-ковбелл, хлопки, дроби хэтов (по умолчанию)
    deep — мягкий deep/lo-fi house 120 BPM: без ковбелла, чистый глубокий саб, пышные пэды, щелчки вместо хлопков
    funk — бразильский фонк 130 BPM: рваная «тамборзан»-бочка, перегруженные ковбелл и 808
           (склейки видео нарезаны под 120, поэтому в долю попадает дроп на 13.6 с; удары на склейках — по кадру)

Всё синтезируется здесь же (никаких сэмплов и стоков), поэтому прав на музыку ни у кого, кроме вас, нет.
Сетка: такт 2 с, доля 0.5 с, шаг 1/16 = 0.125 с; сильные доли на ... 1.6, 3.6, 5.6, ... 13.6 (дроп).
"""
import sys
import numpy as np
import scipy.signal as ss
from scipy.io import wavfile

SR = 44100
BAR0, STEP = -0.4, 0.125          # такты начинаются в -0.4 + 2k (для 120 BPM)
STYLES = {
    'hard': dict(bpm=120, bass='808', cow=1.0, kicks=(0, 4, 8, 12), punch=1.1, clap=.95, snap=False, hats='16', pad=.5, arp=.42, duck=.65),
    'deep': dict(bpm=120, bass='deep', cow=0.0, kicks=(0, 4, 8, 12), punch=.75, clap=.5, snap=True, hats='8', pad=.95, arp=.55, duck=.45),
    'funk': dict(bpm=130, bass='808', cow=1.35, kicks=(0, 3, 6, 8, 11, 14), punch=1.25, clap=1.0, snap=False, hats='16', pad=.35, arp=.3, duck=.7),
}
S = STYLES['hard']


def set_style(name):
    """темп и характер; сильная доля всегда остаётся на 13.6 с — на дропе презентации модели"""
    global S, BAR0, STEP
    S = STYLES[name]; beat = 60 / S['bpm']; STEP = beat / 4; bar = beat * 4
    BAR0 = 13.6 - np.ceil(13.6 / bar + 1e-9) * bar
rng = np.random.default_rng(7)


def T(d): return np.arange(int(d * SR)) / SR
def mf(m): return 440.0 * 2 ** ((m - 69) / 12)
def noise(d): return rng.standard_normal(int(d * SR))
def filt(x, kind, f, order=2): return ss.sosfilt(ss.butter(order, f, btype=kind, fs=SR, output='sos'), x)
def sat(x, k): return np.tanh(x * k) / np.tanh(k)


def sweep(x, kind, f0, f1, order=2, chunk=1024):
    """фильтр с меняющейся частотой среза (по кускам, с переносом состояния)"""
    y, zi, n = np.zeros_like(x), None, len(x)
    for i in range(0, n, chunk):
        f = f0 * (f1 / f0) ** (i / max(1, n - 1))
        sos = ss.butter(order, f, btype=kind, fs=SR, output='sos')
        if zi is None: zi = np.zeros((sos.shape[0], 2))
        y[i:i + chunk], zi = ss.sosfilt(sos, x[i:i + chunk], zi=zi)
    return y


# ---------------------------------------------------------------- инструменты
def kick(punch=1.0):
    t = T(.5); f = 42 + 125 * np.exp(-t * 30)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    return sat(s * punch, 2.4) + filt(noise(.5), 'highpass', 3000) * np.exp(-t * 420) * .22


def clap():
    t = T(.4); env = sum(np.exp(-np.clip(t - d, 0, None) * 150) * (t >= d) for d in (0, .011, .023)) + np.exp(-t * 13) * .4
    return filt(noise(.4) * env, 'bandpass', [900, 7000]) * 1.6


def hat(open_=False):
    d = .32 if open_ else .07; t = T(d)
    return filt(noise(d), 'highpass', 7500) * np.exp(-t * (9 if open_ else 70)) * .9


def b808(m, dur, glide=None):
    t = T(dur + .08); f = np.full(len(t), mf(m))
    if glide is not None: f = mf(m) + (mf(glide) - mf(m)) * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1, t / .004) * np.exp(-t * .9) * np.clip((dur + .08 - t) / .08, 0, 1)
    s = (np.sin(ph) + .3 * np.sin(2 * ph)) * env
    return filt(sat(s, 3.2), 'lowpass', 1400) * .85


def sub(m, dur):
    t = T(dur); env = np.minimum(1, t / .01) * np.clip((dur - t) / .1, 0, 1)
    return np.sin(2 * np.pi * mf(m) * t) * env


def snap():
    t = T(.18); return filt(noise(.18), 'bandpass', [1500, 6000]) * np.exp(-t * 55) * 1.2


def deepbass(m, dur):
    t = T(dur + .1); f = mf(m); env = np.minimum(1, t / .015) * np.exp(-t * .6) * np.clip((dur + .1 - t) / .1, 0, 1)
    return filt(np.sin(2 * np.pi * f * t) + .15 * np.sin(4 * np.pi * f * t), 'lowpass', 400) * env * .9


def cowbell(m, dur=.17, drive=1.6):
    t = T(dur); f = mf(m)
    s = .6 * np.tanh(3 * np.sin(2 * np.pi * f * t)) + .4 * np.tanh(3 * np.sin(2 * np.pi * f * 1.504 * t))
    return sat(filt(s * np.exp(-t * 15), 'bandpass', [f * .7, 7000]), drive) * .8


def supersaw(ms, dur, cut=2200, att=.25, rel=.6):
    t = T(dur + rel); out = []
    for side, det in ((0, (-14, -5, 3, 11)), (1, (-11, -2, 6, 15))):
        s = np.zeros(len(t))
        for m in ms:
            for c in det:
                f = mf(m) * 2 ** (c / 1200); s += 2 * ((f * t + rng.random()) % 1) - 1
        env = np.minimum(1, t / att) * np.clip((dur + rel - t) / rel, 0, 1)
        out.append(filt(s * env, 'lowpass', cut) / (len(ms) * 4))
    return np.vstack(out)


def pluck(m, dur=.28):
    t = T(dur); f = mf(m); saw = 2 * ((f * t) % 1) - 1
    return (filt(saw, 'lowpass', 2600) * .6 + np.sin(2 * np.pi * f * t) * .4) * np.exp(-t * 11)


def riser(dur, f0=180, f1=2600):
    t = T(dur); f = f0 * (f1 / f0) ** (t / dur)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * .25
    return (tone + sweep(noise(dur), 'highpass', 300, 6000) * .5) * (t / dur) ** 2.2


def whoosh(dur, up=True):
    t = T(dur); x = sweep(noise(dur), 'bandpass' if False else 'lowpass', 400 if up else 6000, 6000 if up else 400, 2)
    x = x * np.sin(np.pi * t / dur) ** 2 * 1.4
    pan = np.linspace(-.8, .8, len(t))
    return np.vstack([x * np.cos((pan + 1) * np.pi / 4), x * np.sin((pan + 1) * np.pi / 4)]) * 1.3


def revcym(dur):
    t = T(dur); return filt(noise(dur), 'highpass', 4500) * (t / dur) ** 3 * .9


def impact(big=1.0):
    t = T(2.4); f = 30 + 32 * np.exp(-t * 2.2)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.7)
    crash = filt(noise(2.4), 'lowpass', 3500) * np.exp(-t * 4.5) * .5
    return sat((boom * 1.2 + crash) * big, 1.8)


def hit():
    t = T(.6); f = 45 + 80 * np.exp(-t * 25)
    return sat(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) + filt(noise(.6), 'highpass', 1500) * np.exp(-t * 26) * .7, 2)


def ping(f=1760, dur=1.6):
    t = T(dur); return (np.sin(2 * np.pi * f * t) * np.exp(-t * 3) + .35 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t * 6)) * .5


def laser(dur=.22):
    t = T(dur); f = 300 * (5000 / 300) ** (t / dur)
    return sat(np.sin(2 * np.pi * np.cumsum(f) / SR), 2) * np.exp(-t * 6) * .5


def click():
    t = T(.02); return np.sign(np.sin(2 * np.pi * 2300 * t)) * np.exp(-t * 260) * .5


def tick():
    t = T(.02); return filt(noise(.02), 'highpass', 3500) * np.exp(-t * 400) * .6


# ---------------------------------------------------------------- микшер
class Mix:
    BUSES = ('drums', 'bass', 'music', 'sfx', 'verb')

    def __init__(self, dur):
        self.dur, self.N = dur, int(dur * SR)
        self.b = {k: np.zeros((2, self.N)) for k in self.BUSES}
        self.kicks = []

    def add(self, bus, sig, t, g=1.0, pan=0.0, verb=0.0):
        if sig.ndim == 1:
            a = (pan + 1) * np.pi / 4; sig = np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)
        i = int(round(t * SR))
        if i < 0: sig, i = sig[:, -i:], 0
        if i >= self.N: return
        n = min(sig.shape[1], self.N - i)
        self.b[bus][:, i:i + n] += sig[:, :n] * g
        if verb: self.b['verb'][:, i:i + n] += sig[:, :n] * g * verb

    def span(self, bus, t0, t1): return self.b[bus][:, int(t0 * SR):int(t1 * SR)]


PROG = {'minor': [(33, [57, 60, 64, 69]), (29, [53, 57, 60, 65]), (38, [50, 57, 62, 65]), (40, [52, 56, 59, 64])],
        'major': [(36, [55, 60, 64, 67]), (31, [55, 59, 62, 67]), (33, [57, 60, 64, 69]), (29, [53, 57, 60, 65])]}
COW = {0: 81, 2: 81, 3: 79, 5: 76, 7: 81, 8: 84, 10: 83, 11: 81, 13: 80, 14: 76}      # фонк-ковбелл, ля минор
COW2 = {0: 76, 3: 76, 6: 79, 8: 81, 11: 79, 12: 76, 14: 74}
BASS = {0: 0, 3: 0, 6: 0, 10: 0, 11: 12, 14: 0}                                        # шаг: сдвиг от корня
ARP = [0, 1, 2, 3, 2, 1, 2, 3]


def groove(mx, t0, t1, *, kick_=True, clap_=True, hats='16', bass='808', cow=None, pad=True, pad_cut=2400,
           arp=False, prog='minor', energy=1.0):
    s0 = int(np.ceil((t0 - BAR0) / STEP - 1e-6)); s1 = int(np.floor((t1 - BAR0) / STEP - 1e-6))
    for s in range(s0, s1 + 1):
        t = BAR0 + s * STEP
        if t >= t1: break
        bar, pos = divmod(s, 16); root, chord = PROG[prog][bar % 4]
        if kick_ and pos in S['kicks']:
            mx.add('drums', kick(S['punch'] * energy), t, .95); mx.kicks.append(t)
        if clap_ and pos in (4, 12): mx.add('drums', snap() if S['snap'] else clap(), t, S['clap'] * energy, verb=.35 if S['snap'] else .25)
        if clap_ and S['bpm'] > 125 and pos in (7, 15): mx.add('drums', snap(), t, .5 * energy)
        if hats == '16' and S['hats'] == '8': hats = '8'
        if hats:
            if hats == '16' or pos % 2 == 0:
                acc = .9 if pos % 4 == 2 else .55
                mx.add('drums', hat(pos % 8 == 6 and hats == '16'), t, acc * energy, pan=.25)
            if hats == '16' and bar % 2 == 1 and pos >= 13:              # дробь в конце каждого второго такта
                for k in (1, 2): mx.add('drums', hat(), t + k * STEP / 3, .3 * energy, pan=-.2)
        if bass == '808' and S['bass'] == 'deep': bass = 'deep'
        if bass == 'deep' and pos in (0, 10):
            mx.add('bass', deepbass(root + 12, (10 if pos == 0 else 6) * STEP), t, .75)
        if bass == '808' and pos in BASS:
            nxt = min([p for p in BASS if p > pos] + [16]); m = root + BASS[pos]
            mx.add('bass', b808(m, (nxt - pos) * STEP * .95, glide=root + 12 if pos == 11 else None), t, .8 * energy)
        if bass == 'sub' and pos in (0, 8): mx.add('bass', sub(root + 12, 1.0) * .6, t, .9)
        if cow and S['cow'] and pos in cow: mx.add('music', cowbell(cow[pos], drive=1.6 + S['cow'] * 1.5 - 1.5), t, .75 * energy * S['cow'], pan=-.15, verb=.18)
        if pad and pos == 0:
            ch = chord + [chord[1] + 14] if S['bass'] == 'deep' else chord          # в deep — аккорды пышнее (+9)
            mx.add('music', supersaw(ch, min(16 * STEP, t1 - t), cut=pad_cut * (1.3 if S['bass'] == 'deep' else 1)), t, S['pad'], verb=.3)
        if arp and pos % 2 == 0:
            m = chord[ARP[(pos // 2) % 8]] + 12; mx.add('music', pluck(m), t, S['arp'], pan=.3 if pos % 4 else -.3, verb=.35)


def tapestop(x, a, b):
    """замедление «лентой» на куске [a, b] (в сэмплах): скорость 1 → 0"""
    seg = x[:, a:]; n = b - a; rate = (1 - np.arange(n) / n) ** 1.6
    pos = np.cumsum(rate); out = np.vstack([np.interp(pos, np.arange(seg.shape[1]), seg[c]) for c in range(2)])
    x[:, a:b] = out * np.linspace(1, .2, n); x[:, b:] = 0
    return x


def hook_sfx(mx):
    """0–3.3 с: падение свечи → стоп-кадр → перемотка → точка"""
    H = Mix(.8)
    H.add('sfx', impact(1.2), 0, 1.0)
    H.add('sfx', b808(45, .62, glide=57), 0, .9)                         # 808 срывается вниз вместе с ценой
    H.add('sfx', whoosh(.62, up=False), 0, .6)
    for k, tt in enumerate((0, .125, .25, .375, .4375, .5)):              # удары-заикания
        H.add('sfx', kick(1.3), tt, .7 if k < 3 else .5)
    H.add('sfx', clap(), .25, .6)
    if S['cow']: H.add('sfx', cowbell(81), 0, .5); H.add('sfx', cowbell(76), .1875, .45)
    h = sum(H.b.values())
    h = tapestop(h, int(.42 * SR), int(.64 * SR))                         # стоп-кадр = остановка ленты
    mx.add('sfx', h, 0, 1.0)
    # стоп-кадр: низкий гул, биение сердца, золотая линия-сканер, подъём текста
    t = T(1.0); drone = (np.sin(2 * np.pi * 55 * t) + .6 * np.sin(2 * np.pi * 55.7 * t)) * np.minimum(1, t / .2) * np.clip((1 - t) / .2, 0, 1)
    mx.add('sfx', drone * .35, .6)
    for tt in (.75, 1.25): mx.add('sfx', kick(.6), tt, .45)
    sc = T(.85); mx.add('sfx', np.sin(2 * np.pi * np.cumsum(1800 * (3.2 ** (sc / .85))) / SR) * np.sin(np.pi * sc / .85) * .09, .72, 1, verb=.6)
    mx.add('sfx', whoosh(.5), .74, .35)
    # перемотка: хук задом наперёд, быстрее и с «плаванием»
    r = h[:, :int(.62 * SR)][:, ::-1]; n = int(.75 * SR); pos = np.linspace(0, r.shape[1] - 1, n)
    pos = np.clip(pos + np.sin(np.linspace(0, 40, n)) * 120, 0, r.shape[1] - 1)
    rw = np.vstack([np.interp(pos, np.arange(r.shape[1]), r[c]) for c in range(2)])
    mx.add('sfx', rw * .75, 1.6); mx.add('sfx', filt(noise(.75), 'bandpass', [1500, 5000]) * .12, 1.6)
    # схлопывание в точку: засасывающий свист, глухой удар, чистый «пинг» точки
    mx.add('sfx', revcym(.5), 2.35, .6); mx.add('sfx', hit(), 2.85, .7); mx.add('sfx', ping(2093, 2.0), 2.8, .45, verb=.8)


def terminal_intro(mx):
    """3.3–5.6 с: линия, окно, свечи рисуются, окна щёлкают"""
    mx.add('sfx', whoosh(.5), 3.3, .5); mx.add('sfx', laser(.3), 3.32, .25)
    mx.add('sfx', whoosh(.45), 3.75, .45); mx.add('sfx', hit(), 3.8, .35)
    for k in range(24): mx.add('sfx', tick(), 4.35 + k * .06, .25, pan=np.sin(k) * .5)
    mx.add('music', supersaw(PROG['minor'][0][1], 2.3, cut=900, att=1.2), 3.3, .5, verb=.4)
    for tt in (5.85, 6.0, 6.1, 6.2, 6.3): mx.add('sfx', click(), tt, .5); mx.add('sfx', whoosh(.25), tt - .05, .2)


def fx_bus(mx, bus, t0, t1, kind, f0, f1):
    a, b = int(t0 * SR), int(t1 * SR)
    for c in range(2): mx.b[bus][c, a:b] = sweep(mx.b[bus][c, a:b], kind, f0, f1)


def master(mx, out):
    # сайдчейн: бас и музыка приседают под бочку — «качает»
    duck = np.ones(mx.N); t = np.arange(mx.N) / SR
    for k in mx.kicks:
        i = int(k * SR); j = min(mx.N, i + int(.3 * SR)); tt = t[i:j] - k
        duck[i:j] = np.minimum(duck[i:j], 1 - S['duck'] * np.exp(-tt / .09))
    for bus in ('bass', 'music'): mx.b[bus] *= duck
    # реверб: свёртка с затухающим шумом
    ir_t = T(2.2); ir = np.vstack([filt(noise(2.2), 'lowpass', 5000) * np.exp(-ir_t * 2.6) for _ in range(2)])
    ir /= np.abs(ir).sum(axis=1, keepdims=True) ** .5 * 8
    verb = np.vstack([ss.fftconvolve(mx.b['verb'][c], ir[c])[:mx.N] for c in range(2)])
    mix = mx.b['drums'] * .85 + mx.b['bass'] * .5 + mx.b['music'] * 1.5 + mx.b['sfx'] * .9 + verb * .7
    # «эксайтер»: насыщенный верх подмешивается параллельно — звук становится ярче и сочнее
    mix = mix + np.vstack([np.tanh(filt(mix[c], 'highpass', 2500) * 3) * .12 for c in range(2)])
    mix = np.vstack([filt(mix[c], 'highpass', 25) for c in range(2)])
    mix = sat(mix / (np.abs(mix).max() + 1e-9) * 1.6, 1.4) * .95           # мягкий лимитер
    fade = np.clip((mx.dur - t) / .5, 0, 1); mix *= fade
    wavfile.write(out, SR, (mix.T * 32767 * .89).astype(np.int16))
    print('written', out, f'{mx.dur:.1f}s')


# ---------------------------------------------------------------- аранжировки
def full():
    mx = Mix(48.3)
    hook_sfx(mx); terminal_intro(mx)
    groove(mx, 4.6, 5.6, kick_=False, clap_=False, hats='8', bass=None, pad=False, arp=True, energy=.7)
    groove(mx, 5.6, 10.9, cow=None, arp=True, pad_cut=1800)                                  # грув A
    groove(mx, 7.6, 10.9, kick_=False, clap_=False, hats=None, bass=None, pad=False, cow=COW2)
    for tt in (7.45, 9.05): mx.add('sfx', whoosh(.8), tt, .45)
    mx.add('sfx', ping(1568, 1.2), 8.7, .35, verb=.5); mx.add('sfx', hit(), 8.7, .35)
    # нырок в субпиксели: дробь ускоряется, райзер, фильтр открывается, обрыв перед дропом
    for k, tt in enumerate(np.concatenate([np.arange(10.9, 11.9, .25), np.arange(11.9, 12.7, .125), np.arange(12.7, 13.35, .0625)])):
        mx.add('drums', clap(), tt, .25 + .35 * k / 40)
    mx.add('sfx', riser(2.45), 10.9, .7); mx.add('music', supersaw(PROG['minor'][3][1], 2.4, cut=3000, att=2), 10.9, .35)
    fx_bus(mx, 'music', 10.9, 13.35, 'highpass', 40, 900)
    mx.add('sfx', revcym(.6), 13.0, .6); mx.add('sfx', whoosh(.3), 13.3, .6)
    # 13.6 ДРОП: презентация модели
    groove(mx, 13.6, 20.4, cow=COW, energy=1.1, pad_cut=2600)
    mx.add('sfx', impact(1.0), 13.6, .8)
    for k in range(30): mx.add('sfx', tick(), 13.75 + k * .025, .3)
    for k in range(14): mx.add('sfx', click(), 14.75 + k * .05, .18)
    for tt in (15.6, 16.6, 17.6, 18.6, 19.1, 19.6): mx.add('sfx', hit(), tt, .6, verb=.2); mx.add('sfx', whoosh(.2), tt - .1, .3)
    mx.add('sfx', laser(.25), 20.4, .6); mx.add('sfx', whoosh(.3), 20.35, .5)
    # 20.6 логотип: удар и пауза-брейкдаун на аккорде
    mx.add('sfx', impact(1.1), 20.6, .9)
    mx.add('music', supersaw([57, 64, 69, 71, 76], 2.6, cut=2800, att=.05, rel=1.2), 20.6, .6, verb=.5)
    mx.add('bass', sub(45, 2.6), 20.6, .5); mx.add('sfx', ping(2637, 2.0), 20.9, .2, verb=.8)
    mx.add('sfx', revcym(.4), 23.2, .5)
    # 23.6 терминал возвращается → отъезд в космос: фильтр закрывается
    groove(mx, 23.6, 26.3, cow=None, arp=True, energy=.9)
    fx_bus(mx, 'music', 24.4, 26.3, 'lowpass', 8000, 300); fx_bus(mx, 'bass', 24.4, 26.3, 'lowpass', 1400, 120)
    fx_bus(mx, 'drums', 24.4, 26.3, 'lowpass', 12000, 400)
    mx.add('sfx', whoosh(1.9, up=False), 24.4, .4)
    mx.add('music', supersaw([57, 64, 71], 3.3, cut=1500, att=1.0, rel=.8), 24.6, .45, verb=.6)
    for k in range(10): mx.add('sfx', ping(2000 + rng.random() * 2500, .8), 26.3 + k * .06, .05, pan=rng.random() * 2 - 1, verb=.7)
    mx.add('sfx', riser(.7, 400, 4000), 26.9, .7); mx.add('sfx', revcym(.7), 26.9, .6)
    # 27.6 телефон: грув B
    groove(mx, 27.6, 35.6, cow=COW2, arp=True, energy=1.0)
    mx.add('sfx', impact(.8), 27.6, .6)
    mx.add('sfx', click(), 32.6, .9); mx.add('sfx', hit(), 32.6, .3)
    mx.add('sfx', ping(1318, 1.2), 32.75, .35, verb=.4); mx.add('sfx', ping(1976, 1.2), 32.82, .35, verb=.4)
    mx.add('sfx', whoosh(.8), 35.6, .45)
    # 35.6 «Жизнь на полную»: светлый мажор, чистый саб
    groove(mx, 35.6, 41.6, bass='sub', cow=None, arp=True, prog='major', energy=.85, pad_cut=3200)
    for tt in (39.15, 39.77, 40.39, 41.01): mx.add('sfx', tick(), tt, .6); mx.add('sfx', whoosh(.2), tt - .08, .2)
    mx.add('sfx', revcym(.6), 41.0, .5)
    # 41.6 Telegram: финальный грув, удар в конце
    groove(mx, 41.6, 47.6, cow=COW, arp=True, energy=1.05)
    mx.add('sfx', impact(.9), 41.6, .7); mx.add('sfx', ping(1760, 1.4), 44.25, .3, verb=.5)
    mx.add('sfx', impact(1.0), 47.6, .8); mx.add('music', supersaw([45, 57, 64, 69], .1, cut=2000, att=.01, rel=.7), 47.6, .5, verb=.6)
    master(mx, OUT('promo_audio'))


# рекламная нарезка: (источник в promo.mp4, начало, конец) → стыки на долях 0.5 с
AD_CUTS = [(0.0, 3.3), (3.3, 7.1), (8.4, 9.4), (16.6, 18.6), (20.4, 22.4), (32.0, 33.5), (41.8, 46.8)]


def ad():
    mx = Mix(18.6)
    hook_sfx(mx); terminal_intro(mx)
    groove(mx, 4.6, 5.6, kick_=False, clap_=False, hats='8', bass=None, pad=False, arp=True, energy=.7)
    groove(mx, 5.6, 7.6, arp=True, pad_cut=1800)
    mx.add('sfx', hit(), 7.1, .5); mx.add('sfx', ping(1568, 1.2), 7.4, .35, verb=.5)    # ИИ: 78% · LONG
    mx.add('sfx', revcym(.5), 7.6, .6)
    groove(mx, 8.1, 10.1, cow=COW, energy=1.1)                                          # дроп на монтаже модели
    mx.add('sfx', impact(1.0), 8.1, .8); mx.add('sfx', hit(), 9.1, .6)
    mx.add('sfx', laser(.25), 10.1, .6); mx.add('sfx', impact(1.1), 10.3, .9)            # росчерк → логотип
    mx.add('music', supersaw([57, 64, 69, 71, 76], 1.8, cut=2800, att=.05, rel=.8), 10.3, .6, verb=.5)
    mx.add('bass', sub(45, 1.8), 10.3, .5); mx.add('sfx', revcym(.4), 11.7, .5)
    groove(mx, 12.1, 13.6, cow=COW2, energy=1.0)                                         # телефон
    mx.add('sfx', click(), 12.7, .9); mx.add('sfx', ping(1318, 1.2), 12.85, .35, verb=.4); mx.add('sfx', ping(1976, 1.2), 12.92, .35, verb=.4)
    groove(mx, 13.6, 18.1, cow=COW, arp=True, energy=1.05)                               # Telegram
    mx.add('sfx', impact(.9), 13.6, .7); mx.add('sfx', ping(1760, 1.4), 16.0, .3, verb=.5)
    mx.add('sfx', impact(1.0), 18.1, .8)
    master(mx, OUT('ad_audio'))


if __name__ == '__main__':
    STYLE = sys.argv[2] if len(sys.argv) > 2 else 'hard'
    set_style(STYLE)
    OUT = lambda base: f'{base}.wav' if STYLE == 'hard' else f'{base}_{STYLE}.wav'
    {'full': full, 'ad': ad}[sys.argv[1] if len(sys.argv) > 1 else 'full']()
