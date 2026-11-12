"""Forecast Roblox's Up-and-Coming top 10 over the next 72 hourly charts.

Reads the chart history in $MAKEFUN_INPUT_DIR (default /input) and writes forecast.json to
$MAKEFUN_OUTPUT_DIR (default /output). Standard library only, deterministic, no network.

Method (v3): a linear pool of three different forecasters, weighted to maximise the contest's
log score on rolling backtests (Geweke and Amisano's optimal prediction pools):

1. Monte Carlo of the chart's latent order with skewed (reflected Gumbel) noise: each current game
   sits at -log(rank) plus a drift learned from the history (ridge regression of future log-rank
   change on its features times the horizon, recent reads weighted more) plus noise growing with
   the horizon; every sample is sorted to count who lands at each place, at every hour.
2. The same simulation with heavy-tailed (Student t) noise.
3. A conditional logit (discrete choice) model: each place of a later chart is a choice among the
   games on the earlier chart, with a softmax over features of the game's distance to that place
   and its momentum, fitted by maximum likelihood on every pair of history reads whose gap
   matches the window, i.e. trained directly on the log score pooled over the window.

The share of each place that games not yet on the chart have been taking at each horizon is
measured on the history and left unnamed, so the scorer spreads it over the games that join
later. Hourly stretches of the history are density-weighted so they count about as much per day
as the sparse archive. If a forecaster fails, the pool renormalises over the others; if all fail,
a persistence forecast is written.
"""
import json
import math
import os
from datetime import datetime, timedelta

PLACES = 10


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows = d.get("up_and_coming")
            if not d.get("ok", True) or not rows:
                continue
            rows = sorted(rows, key=lambda r: r["rank"])
            reads.append({
                "t": ts(d["at"]),
                "ids": [int(r["universe_id"]) for r in rows],
                "playing": {int(r["universe_id"]): (r.get("playing") or 0) for r in rows},
                "details": d.get("details") or {},
            })
    reads.sort(key=lambda r: r["t"])
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    return reads, task


DEFAULTS = {
    "bw_h": 3.0,        # horizon kernel half-width (hours)
    "bw_r": 0.25,       # rank smoothing bandwidth in log-rank
    "max_rank": 100,    # ranks tracked
    "eps": 0.0001,      # floor mass spread over every tracked game at each place
    "mix_kernel": 0.0,  # weight of the empirical kernel vs the Monte Carlo model
    "ridge": 0.002,     # ridge penalty of the drift regression (per sample)
    "kappa": 1.5,       # multiplier on the fitted drift
    "sig24": 0.45,      # latent noise sd at 24 h
    "alpha": 0.65,      # noise growth exponent
    "pn_half": 1.5,     # half-life (days) of the recency weight on the new-game rate
    "pn_slope": 0.5,    # tilt of the new-game rate from place 1 to place 10
    "pn_mult": 1.25,    # multiplier on the new-game rate
    "use_g72": True,    # include the 72-hour player growth feature
    "dens_h": 3.0,      # reads within this many hours share one read's weight in the drift fit
    "top_sig": 0.3,     # extra noise for the top three places (1 at #1, fading to 0 at #4)
    "gamma": 0.0,       # Box-Cox exponent of the latent rank scale (0: log rank)
    "drift_pow": 1.0,   # drift grows as (h/24) ** drift_pow
    "drop_p": 0.0,      # leave games below this probability unnamed (their mass joins the unnamed share)
    "temp": 1.0,        # power applied to each place's named probabilities (below 1 flattens)
    "jump_p": 0.0,      # chance of a heavy-tailed move
    "jump_sd": 3.0,     # its noise multiplier
    "samples": 1000,    # Monte Carlo samples per sampled horizon
    "h_step": 1,        # sample every h_step-th slot
    "seed": 20261113,
    "f_prank": True,    # rank by players vs chart rank feature
    "drift_half": 3.0,  # half-life (days) of the recency weight in the drift fit
    "pn_dens": True,    # density-weight the new-game rate like the drift fit
    "sig_low": 0.0,
    "f_nonlin": False,  # extra nonlinear features
    "f_life": False,    # age-on-chart bins (life-cycle curve)
    "noise": "gauss",   # gauss | gumbel | ngumbel | logistic | resid | student
    "resid_scale": 1.0,
    "t_df": 4.0,
    "reloc": 0.0,       # per-day chance a game is displaced to a uniform random rank
    "vol_beta": 0.0,    # sigma *= (recent churn / overall churn) ** vol_beta
    "vol_hours": 48.0,
}


def horizons(reads, task):
    w = task["window"]
    start = ts(w["from"])
    slots = int(w.get("slots", 72))
    step = float(w.get("slot_minutes", 60)) / 60.0
    last = reads[-1]["t"]
    h0 = (start - last).total_seconds() / 3600.0
    return [h0 + s * step for s in range(slots)]


def kernel(reads, hs, prm):
    """Pooled transition counts C[r][k] (r: start rank 1.., k: 1..10) and new-game share per place."""
    R = prm["max_rank"]
    bw = prm["bw_h"]
    n = len(reads)
    times = [r["t"] for r in reads]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            h = (times[j] - times[i]).total_seconds() / 3600.0
            if h > hs[-1] + bw:
                break
            if h >= hs[0] - bw and h > 0:
                pairs.append((i, j, h))
    # weight of a pair: sum over target horizons of its triangular kernel, normalised per horizon
    z = [0.0] * len(hs)
    kv = []
    for (i, j, h) in pairs:
        ks = []
        for s, hh in enumerate(hs):
            d = abs(h - hh)
            if d < bw:
                k = 1.0 - d / bw
                ks.append((s, k))
                z[s] += k
        kv.append(ks)
    live = [s for s in range(len(hs)) if z[s] > 0]
    C = [[0.0] * (PLACES + 1) for _ in range(R + 2)]  # index PLACES = elsewhere/gone
    N = [0.0] * (R + 2)
    newk = [0.0] * PLACES
    W = 0.0
    for (i, j, h), ks in zip(pairs, kv):
        w = sum(k / z[s] for s, k in ks) / max(1, len(live))
        if w <= 0:
            continue
        a, b = reads[i], reads[j]
        pos_b = {u: q for q, u in enumerate(b["ids"], 1)}
        set_a = set(a["ids"])
        for r, u in enumerate(a["ids"][:R], 1):
            q = pos_b.get(u)
            if q is not None and q <= PLACES:
                C[r][q - 1] += w
            else:
                C[r][PLACES] += w
            N[r] += w
        for q, u in enumerate(b["ids"][:PLACES], 1):
            if u not in set_a:
                newk[q - 1] += w
        W += w
    pnew = [x / W if W > 0 else 0.0 for x in newk]
    return C, N, pnew


def smooth(C, N, prm):
    R = prm["max_rank"]
    bw = prm["bw_r"]
    M = [[0.0] * (PLACES + 1) for _ in range(R + 2)]
    for r in range(1, R + 1):
        num = [0.0] * (PLACES + 1)
        den = 0.0
        lr = math.log(r)
        for r2 in range(1, R + 1):
            if N[r2] <= 0:
                continue
            a = math.exp(-abs(math.log(r2) - lr) / bw)
            if a < 1e-4:
                continue
            for k in range(PLACES + 1):
                num[k] += a * C[r2][k]
            den += a * N[r2]
        if den > 0:
            M[r] = [x / den for x in num]
    return M


def hours(a, b):
    return (b - a).total_seconds() / 3600.0


def back_read(reads, i, target, lo, hi):
    """Index of the read closest to `target` hours before read i, within [lo, hi] hours, or None."""
    ti = reads[i]["t"]
    p = None
    best = None
    for q in range(i - 1, -1, -1):
        d = hours(reads[q]["t"], ti)
        if d > hi:
            break
        if d >= lo and (best is None or abs(d - target) < best):
            best, p = abs(d - target), q
    return p


def features(reads, i, first, use_g72=True, prm=None):
    """Per-game features at read i: [trend, growth, uid recency, log rank, age, new-in-24h, growth72, top]."""
    ti = reads[i]["t"]
    p = back_read(reads, i, 24.0, 12.0, 36.0)
    p3 = back_read(reads, i, 72.0, 54.0, 90.0) if use_g72 else None
    pl3 = reads[p3]["playing"] if p3 is not None else {}
    ids = reads[i]["ids"]
    order = {u: q for q, u in enumerate(sorted(ids))}
    n = max(1, len(ids) - 1)
    pp = {u: q for q, u in enumerate(reads[p]["ids"], 1)} if p is not None else {}
    pl_p = reads[p]["playing"] if p is not None else {}
    pl_i = reads[i]["playing"]
    out = {}
    fpr = bool(prm and prm.get("f_prank"))
    if fpr:
        prk = {u: q for q, u in enumerate(sorted(ids, key=lambda u: (-(pl_i.get(u) or 0), u)), 1)}
    for r, u in enumerate(ids, 1):
        if p is not None:
            tr = math.log(pp.get(u, 110) / r)
            newin = 0.0 if u in pp else 1.0
            g = math.log((pl_i.get(u, 0) + 20) / (pl_p[u] + 20)) if u in pl_p else 0.0
        else:
            tr, newin, g = 0.0, 0.0, 0.0
        f = first.get(u, 0)
        age = min(hours(reads[f]["t"], ti), 168.0) / 168.0 if f > 0 else 1.0
        g3 = math.log((pl_i.get(u, 0) + 20) / (pl3[u] + 20)) if u in pl3 else 0.0
        top = max(0.0, 1.0 - math.log(r) / math.log(4.0))
        out[u] = [tr, g, order[u] / n, math.log(r), age, newin, g3, top]
        if fpr:
            out[u].append(math.log(prk[u] / r))
        if prm and prm.get("f_nonlin"):
            lr = math.log(r)
            out[u] += [tr * tr, g * g, tr * lr, age * lr]
        if prm and prm.get("f_life"):
            ah = (hours(reads[f]["t"], ti) if f > 0 else 999.0)
            out[u] += [1.0 if ah < 12 else 0.0, 1.0 if 12 <= ah < 36 else 0.0, 1.0 if 36 <= ah < 72 else 0.0,
                       1.0 if 72 <= ah < 144 else 0.0]
    return out


def solve(A, B):
    m = len(B)
    M = [A[i][:] + [B[i]] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                if f:
                    M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [M[i][m] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(m)]


def density(reads, width):
    """How many reads lie within +-width hours of each read (itself included): hourly stretches
    of the history count about as much per day as the sparse archive."""
    if width <= 0:
        return [1.0] * len(reads)
    t = [r["t"] for r in reads]
    out = []
    lo = 0
    hi = 0
    n = len(t)
    for i in range(n):
        while hours(t[lo], t[i]) > width:
            lo += 1
        while hi + 1 < n and hours(t[i], t[hi + 1]) <= width:
            hi += 1
        out.append(float(hi - lo + 1))
    return out


def fit_drift(reads, first, prm):
    """Least squares of future log-rank change on features times horizon (per day), from the history."""
    nf = 8 + (1 if prm.get("f_prank") else 0) + (4 if prm.get("f_nonlin") else 0) + (4 if prm.get("f_life") else 0)
    m = nf + 1
    tend = reads[-1]["t"]
    dh = prm.get("drift_half", 0.0) * 24.0
    A = [[0.0] * m for _ in range(m)]
    B = [0.0] * m
    cnt = 0
    wsum = 0.0
    t0 = reads[0]["t"]
    dens = density(reads, prm["dens_h"])
    for i in range(len(reads)):
        if hours(t0, reads[i]["t"]) < 30:
            continue
        F = None
        for j in range(i + 1, len(reads)):
            h = hours(reads[i]["t"], reads[j]["t"])
            if h > 72:
                break
            if h < 6:
                continue
            if F is None:
                F = features(reads, i, first, prm["use_g72"], prm)
            w = 1.0 / (dens[i] * dens[j])
            if dh > 0:
                w *= 0.5 ** (hours(reads[i]["t"], tend) / dh)
            pos = {u: q for q, u in enumerate(reads[j]["ids"], 1)}
            hd = h / 24.0
            for r, u in enumerate(reads[i]["ids"], 1):
                if r > 60:
                    break
                y = math.log(pos.get(u, 130) / r)
                x = [hd] + [hd * v for v in F[u]]
                for a in range(m):
                    xa = x[a] * w
                    B[a] += xa * y
                    row = A[a]
                    for b in range(m):
                        row[b] += xa * x[b]
                cnt += 1
                wsum += w
    if cnt < 200:
        return [0.0] * nf
    lam = prm["ridge"] * wsum
    for a in range(1, m):
        A[a][a] += lam
    beta = solve(A, B)
    return beta[1:]


def resid_pool(reads, first, beta, prm, cap=6000):
    """Standardised residuals of the drift fit, -(y - x.beta) / (h/24)^alpha, for bootstrap noise."""
    t0 = reads[0]["t"]
    out = []
    al = prm["alpha"]
    kap = prm["kappa"]
    for i in range(len(reads)):
        if hours(t0, reads[i]["t"]) < 30:
            continue
        F = None
        for j in range(i + 1, len(reads)):
            h = hours(reads[i]["t"], reads[j]["t"])
            if h > 72:
                break
            if h < 6:
                continue
            if F is None:
                F = features(reads, i, first, prm["use_g72"], prm)
            pos = {u: q for q, u in enumerate(reads[j]["ids"], 1)}
            hd = h / 24.0
            for r, u in enumerate(reads[i]["ids"], 1):
                if r > 60:
                    break
                y = math.log(pos.get(u, 130) / r)
                pred = kap * hd * sum(b * v for b, v in zip(beta, F[u]))
                out.append(-(y - pred) / (hd ** al))
    if not out:
        return [0.0]
    m = sum(out) / len(out)
    out = [v - m for v in out]
    if len(out) > cap:
        step = len(out) / cap
        out = [out[int(k * step)] for k in range(cap)]
    return out


def vol_ratio(reads, prm):
    """Recent churn of the chart vs its whole-history churn (mean |log rank change| per day^alpha)."""
    al = prm["alpha"]
    tend = reads[-1]["t"]
    rec, allv = [0.0, 0], [0.0, 0]
    for i in range(len(reads)):
        pos_i = None
        for j in range(i + 1, len(reads)):
            h = hours(reads[i]["t"], reads[j]["t"])
            if h > 30:
                break
            if h < 6:
                continue
            if pos_i is None:
                pos_i = {u: q for q, u in enumerate(reads[i]["ids"], 1)}
            s, n = 0.0, 0
            for q, u in enumerate(reads[j]["ids"][:40], 1):
                r = pos_i.get(u)
                if r is not None:
                    s += abs(math.log(q / r))
                    n += 1
            if n:
                v = s / n / (h / 24.0) ** al
                allv[0] += v
                allv[1] += 1
                if hours(reads[j]["t"], tend) <= prm["vol_hours"]:
                    rec[0] += v
                    rec[1] += 1
    if rec[1] < 3 or allv[1] < 3:
        return 1.0
    return (rec[0] / rec[1]) / (allv[0] / allv[1])


def pnew_by_h(reads, hs, prm):
    """Share of each top-10 place held by games absent from the earlier read, per target horizon.

    Pooled over the ten places (one rate per horizon, tilted by place with a fixed shape) and
    weighted toward recent pairs, since the chart's churn changes from week to week.
    """
    bw = prm["bw_h"]
    num = [0.0] * len(hs)
    den = [0.0] * len(hs)
    times = [r["t"] for r in reads]
    tend = times[-1]
    half = prm["pn_half"] * 24.0
    dens = density(reads, prm["dens_h"]) if prm.get("pn_dens") else [1.0] * len(reads)
    for i in range(len(reads)):
        set_a = None
        rec = 0.5 ** (hours(times[i], tend) / half) if half > 0 else 1.0
        for j in range(i + 1, len(reads)):
            h = hours(times[i], times[j])
            if h > hs[-1] + bw:
                break
            if h < hs[0] - bw or h <= 0:
                continue
            if set_a is None:
                set_a = set(reads[i]["ids"])
            fl = sum(1.0 for u in reads[j]["ids"][:PLACES] if u not in set_a) / PLACES
            for s, hh in enumerate(hs):
                d = abs(h - hh)
                if d < bw:
                    k = (1.0 - d / bw) * rec / (dens[i] * dens[j])
                    den[s] += k
                    num[s] += k * fl
    g = prm["pn_slope"]
    shape = [1.0 + g * (q - (PLACES - 1) / 2.0) / ((PLACES - 1) / 2.0) for q in range(PLACES)]
    mult = prm["pn_mult"]
    out = []
    for s in range(len(hs)):
        if den[s] > 0:
            rate = min(0.9, mult * num[s] / den[s])
            out.append([min(0.95, rate * sh) for sh in shape])
        else:
            out.append(None)
    # fill gaps from neighbours (horizons beyond what the history covers)
    lastv = None
    for s in range(len(hs)):
        if out[s] is None:
            out[s] = lastv
        else:
            lastv = out[s]
    nxt = None
    for s in range(len(hs) - 1, -1, -1):
        if out[s] is None:
            out[s] = nxt
        else:
            nxt = out[s]
    return [o if o is not None else [0.0] * PLACES for o in out]


def mc_probs(reads, hs, prm):
    """Monte Carlo of the latent order: drift from features, Gaussian noise growing with the horizon."""
    first = {}
    for i, r in enumerate(reads):
        for u in r["ids"]:
            first.setdefault(u, i)
    beta = fit_drift(reads, first, prm)
    F = features(reads, len(reads) - 1, first, prm["use_g72"], prm)
    ids = reads[-1]["ids"][:prm["max_rank"]]
    kappa = prm["kappa"]
    gam = prm["gamma"]
    if gam == 0:
        base = [-math.log(r) for r in range(1, len(ids) + 1)]
        jac = [1.0] * len(ids)
    else:
        # Box-Cox rank scale: -(r^g - 1) / g; the drift is fitted in log-rank units, so it is
        # carried over with the scale's slope d(latent)/d(log r) = r^g
        base = [-(r ** gam - 1.0) / gam for r in range(1, len(ids) + 1)]
        jac = [r ** gam for r in range(1, len(ids) + 1)]
    drift = [-kappa * jac[q] * sum(b * v for b, v in zip(beta, F[u])) for q, u in enumerate(ids)]  # per day, latent up = better
    tt = prm["top_sig"]
    sm = [1.0 + tt * max(0.0, 1.0 - math.log(r) / math.log(4.0)) for r in range(1, len(ids) + 1)]
    sl = prm.get("sig_low", 0.0)
    if sl:
        sm = [v * max(0.2, 1.0 + sl * max(0.0, math.log(r) - math.log(10.0))) for r, v in zip(range(1, len(ids) + 1), sm)]
    rnd = __import__("random").Random(prm["seed"])
    noise = prm.get("noise", "gauss")
    pool = resid_pool(reads, first, beta, prm) if noise == "resid" else None
    rs = prm.get("resid_scale", 1.0)
    vmult = vol_ratio(reads, prm) ** prm["vol_beta"] if prm.get("vol_beta") else 1.0
    reloc = prm.get("reloc", 0.0)
    tdf = prm.get("t_df", 4.0)
    tsc = math.sqrt((tdf - 2.0) / tdf) if tdf > 2 else 1.0
    G = 0.5772156649
    GS = math.pi / math.sqrt(6.0)
    LS = math.pi / math.sqrt(3.0)

    def draw():
        if noise == "gauss":
            return rnd.gauss(0.0, 1.0)
        if noise == "gumbel":
            return (-math.log(-math.log(rnd.random() or 1e-12)) - G) / GS
        if noise == "ngumbel":
            return -(-math.log(-math.log(rnd.random() or 1e-12)) - G) / GS
        if noise == "logistic":
            p = min(max(rnd.random(), 1e-12), 1 - 1e-12)
            return math.log(p / (1 - p)) / LS
        if noise == "student":
            z = rnd.gauss(0.0, 1.0)
            c = rnd.gammavariate(tdf / 2.0, 2.0)
            return z / math.sqrt(c / tdf) * tsc
        return pool[rnd.randrange(len(pool))]
    S = prm["samples"]
    step = prm["h_step"]
    counts = [dict() for _ in range(PLACES)]
    hsel = list(range(0, len(hs), step))
    pn = pnew_by_h(reads, hs, prm)
    acc = [dict() for _ in range(PLACES)]
    n = len(ids)
    pj, tau = prm["jump_p"], prm["jump_sd"]
    for s in hsel:
        h = max(hs[s], 0.0)
        sig = prm["sig24"] * (max(h, 0.5) / 24.0) ** prm["alpha"] * vmult
        if noise == "resid":
            sig = rs * (max(h, 0.5) / 24.0) ** prm["alpha"] * vmult
        prl = 1.0 - math.exp(-reloc * max(h, 0.0) / 24.0) if reloc > 0 else 0.0
        lowz = -math.log(n + 1.0)
        hd = h / 24.0
        hdp = hd ** prm["drift_pow"] if hd > 0 else 0.0
        mu = [base[q] + drift[q] * hdp for q in range(n)]
        cnt = [[0] * n for _ in range(PLACES)]
        for _ in range(S):
            z = []
            for q in range(n):
                if prl > 0 and rnd.random() < prl:
                    z.append((-math.log(1.0 + rnd.random() * n), q))
                    continue
                e = draw()
                if pj > 0 and rnd.random() < pj:
                    e *= tau
                z.append((mu[q] + sig * sm[q] * e, q))
            z.sort(reverse=True)
            for k in range(PLACES):
                cnt[k][z[k][1]] += 1
        # weight of this sampled horizon = number of slots it stands for
        wsl = min(step, len(hs) - s)
        for k in range(PLACES):
            keep = 1.0 - pn[s][k]
            row = cnt[k]
            a = acc[k]
            for q in range(n):
                if row[q]:
                    a[ids[q]] = a.get(ids[q], 0.0) + wsl * keep * row[q] / S
    pnew = [0.0] * PLACES
    for s in range(len(hs)):
        for k in range(PLACES):
            pnew[k] += pn[s][k] / len(hs)
    for k in range(PLACES):
        for u in acc[k]:
            acc[k][u] /= len(hs)
    return acc, pnew, beta


def build(reads, task, prm=None):
    prm = dict(DEFAULTS, **(prm or {}))
    hs = horizons(reads, task)
    last = reads[-1]["ids"][:prm["max_rank"]]
    eps = prm["eps"]
    lam = prm["mix_kernel"]
    if lam > 0:
        C, N, pnew_k = kernel(reads, hs, prm)
        M = smooth(C, N, prm)
    else:  # the empirical kernel is off: skip it
        M = [[0.0] * (PLACES + 1) for _ in range(len(last) + 2)]
        pnew_k = [0.0] * PLACES
    if lam < 1.0:
        mc, pnew_m, _ = mc_probs(reads, hs, prm)
    P = {}
    pnew_out = []
    for k in range(PLACES):
        kern = {u: M[r][k] for r, u in enumerate(last, 1)}
        tot = sum(kern.values())
        target = max(0.0, 1.0 - pnew_k[k])
        if tot > 0:
            kern = {u: v / tot * target for u, v in kern.items()}
        if lam < 1.0:
            named = {u: lam * kern.get(u, 0.0) + (1 - lam) * mc[k].get(u, 0.0) for u in last}
            pn = lam * pnew_k[k] + (1 - lam) * pnew_m[k]
        else:
            named, pn = kern, pnew_k[k]
        target = max(0.0, 1.0 - pn)
        temp = prm["temp"]
        if temp != 1.0:
            tot = sum(named.values())
            powd = {u: v ** temp for u, v in named.items()}
            tp = sum(powd.values())
            if tp > 0:
                named = {u: v / tp * tot for u, v in powd.items()}
        m = len(named)
        named = {u: (1 - eps * m) * v + eps * target for u, v in named.items()}
        drop = prm["drop_p"]
        if drop > 0:
            named = {u: v for u, v in named.items() if v >= drop}
        P[k] = named
        pnew_out.append(pn)
    return P, pnew_out


def to_doc(P, last_ids):
    rows = []
    for k in range(PLACES):
        items = sorted(P[k].items(), key=lambda x: (-x[1], x[0]))[:100]
        s = sum(p for _, p in items)
        scale = min(1.0, (1.0 - 1e-9) / s) if s > 0 else 1.0
        for u, p in items:
            q = round(p * scale, 7)
            if q > 0:
                rows.append({"universe_id": u, "place": k + 1, "p": q})
    # guarantee each place sums to at most 1 after rounding
    sums = [0.0] * PLACES
    for r in rows:
        sums[r["place"] - 1] += r["p"]
    for r in rows:
        if sums[r["place"] - 1] > 1.0:
            r["p"] = math.floor(r["p"] / sums[r["place"] - 1] * 1e7) / 1e7
    # point guess: greedy by probability, place 1 first
    used = set()
    top = []
    for k in range(PLACES):
        cand = sorted(P[k].items(), key=lambda x: (-x[1], x[0]))
        for u, _ in cand:
            if u not in used:
                used.add(u)
                top.append(u)
                break
    for u in last_ids:
        if len(top) >= PLACES:
            break
        if u not in used:
            used.add(u)
            top.append(u)
    return {"probabilities": rows, "top10": top[:PLACES]}


def fallback(reads):
    """Persistence spread over neighbouring places: used only if the model fails on its input."""
    last = reads[-1]["ids"] if reads else []
    P = {}
    for k in range(PLACES):
        named = {}
        for r, u in enumerate(last[:30], 1):
            named[u] = 0.8 * math.exp(-abs(math.log(r) - math.log(k + 1)) / 0.35)
        tot = sum(named.values())
        if tot > 0:
            named = {u: v / tot * 0.8 for u, v in named.items()}
        P[k] = named
    return P


CL_DEFAULTS = dict(DEFAULTS, cl_maxr=80, cl_l2=2.0, cl_iters=12, cl_bw=4.0, cl_rich=True, cl_hbal=False)
NF = 15
RICH = False


def phi(Fu, r, k):
    tr, g, _uid, lr, age, newin, g3, _top, prank = Fu[:9]
    d = math.log(r) - math.log(k)
    out = [d * d, abs(d), d, 1.0 if r == k else 0.0, d * tr, d * g, d * g3, d * prank,
           tr, g, g3, prank, newin, age, d * age]
    if RICH:
        tk = 1.0 if k <= 3 else 0.0
        out += [d * newin, d * d * tr, (1.0 if r == k else 0.0) * tk, lr, d * tk, lr * tr,
                1.0 if abs(r - k) == 1 else 0.0]
    return out


def objective(groups, theta, lam):
    L = 0.0
    for Phi, ys in groups:
        s = [sum(a * b for a, b in zip(theta, f)) for f in Phi]
        mx = max(s)
        lz = mx + math.log(sum(math.exp(v - mx) for v in s))
        for y, wy in ys.items():
            L += wy * (s[y - 1] - lz)
    return L - 0.5 * lam * sum(v * v for v in theta)


def build_clogit(reads, task, prm=None):
    global NF, RICH
    prm = dict(CL_DEFAULTS, **(prm or {}))
    prm["f_prank"] = True
    RICH = bool(prm["cl_rich"])
    NF = 22 if RICH else 15
    hs = horizons(reads, task)
    first = {}
    for i, r in enumerate(reads):
        for u in r["ids"]:
            first.setdefault(u, i)
    n = len(reads)
    t = [r["t"] for r in reads]
    dens = density(reads, prm["dens_h"])
    bw = prm["cl_bw"]
    maxr = prm["cl_maxr"]
    tend = t[-1]
    dh = prm.get("drift_half", 0.0) * 24.0
    # horizon balance: every gap bin gets the weight the window gives it, however many read pairs
    # the history happens to have at that gap (an hourly tail would otherwise flood the short gaps)
    fbin = None
    if prm.get("cl_hbal"):
        T = {}
        Mw = {}
        for i in range(n):
            if hours(t[0], t[i]) < 30:
                continue
            for j in range(i + 1, n):
                h = hours(t[i], t[j])
                if h > hs[-1] + bw:
                    break
                b = int(h)
                w = 1.0 / (dens[i] * dens[j])
                if dh > 0:
                    w *= 0.5 ** (hours(t[i], tend) / dh)
                T[b] = T.get(b, 0.0) + w
        for b in T:
            m_ = 0.0
            for hh in hs:
                dd = abs(b + 0.5 - hh)
                if dd < bw:
                    m_ += 1.0 - dd / bw
            Mw[b] = m_
        st, sm = sum(T.values()), sum(Mw.values())
        fbin = {b: (Mw[b] / T[b]) * (st / sm) if T[b] > 0 and sm > 0 else 0.0 for b in T}
    groups = []  # (Phi[a] list, {y_rank: weight})
    for i in range(n):
        if hours(t[0], t[i]) < 30:
            continue
        ys = [dict() for _ in range(10)]
        any_ev = False
        rank_i = {u: q for q, u in enumerate(reads[i]["ids"][:maxr], 1)}
        for j in range(i + 1, n):
            h = hours(t[i], t[j])
            if h > hs[-1] + bw:
                break
            ws = 0.0
            for hh in hs:
                dd = abs(h - hh)
                if dd < bw:
                    ws += 1.0 - dd / bw
            if ws <= 0:
                continue
            w = ws / (dens[i] * dens[j])
            if dh > 0:
                w *= 0.5 ** (hours(t[i], tend) / dh)
            if fbin is not None:
                w = fbin.get(int(h), 0.0) / (dens[i] * dens[j]) * (0.5 ** (hours(t[i], tend) / dh) if dh > 0 else 1.0)
                if w <= 0:
                    continue
            for k in range(10):
                if k >= len(reads[j]["ids"]):
                    break
                y = rank_i.get(reads[j]["ids"][k])
                if y is None:
                    continue
                ys[k][y] = ys[k].get(y, 0.0) + w
                any_ev = True
        if not any_ev:
            continue
        F = features(reads, i, first, True, prm)
        ids = reads[i]["ids"][:maxr]
        for k in range(10):
            if ys[k]:
                Phi = [phi(F[u], r, k + 1) for r, u in enumerate(ids, 1)]
                groups.append((Phi, ys[k]))
    theta = [0.0] * NF
    theta[0] = -1.0
    lam = prm["cl_l2"]
    for _ in range(prm["cl_iters"]):
        g = [0.0] * NF
        H = [[0.0] * NF for _ in range(NF)]
        for Phi, ys in groups:
            s = [sum(a * b for a, b in zip(theta, f)) for f in Phi]
            mx = max(s)
            e = [math.exp(v - mx) for v in s]
            z = sum(e)
            p = [v / z for v in e]
            W = sum(ys.values())
            mu = [0.0] * NF
            for pa, f in zip(p, Phi):
                if pa < 1e-12:
                    continue
                for c in range(NF):
                    mu[c] += pa * f[c]
            for y, wy in ys.items():
                fy = Phi[y - 1]
                for c in range(NF):
                    g[c] += wy * (fy[c] - mu[c])
            for pa, f in zip(p, Phi):
                if pa < 1e-9:
                    continue
                wp = W * pa
                for c in range(NF):
                    dc = f[c] - mu[c]
                    if dc == 0.0:
                        continue
                    row = H[c]
                    v = wp * dc
                    for c2 in range(c, NF):
                        row[c2] += v * (f[c2] - mu[c2])
        for c in range(NF):
            for c2 in range(c):
                H[c][c2] = H[c2][c]
            g[c] -= lam * theta[c]
            H[c][c] += lam
        step = solve(H, g)
        base = objective(groups, theta, lam)
        a = 1.0
        for _ in range(12):
            cand = [x + a * y for x, y in zip(theta, step)]
            if objective(groups, cand, lam) > base:
                break
            a *= 0.5
        else:
            break
        theta = cand
        if max(abs(v) for v in step) * a < 1e-4:
            break
    pn = pnew_by_h(reads, hs, prm)
    pnew = [sum(pn[s][k] for s in range(len(hs))) / len(hs) for k in range(10)]
    F = features(reads, n - 1, first, True, prm)
    last = reads[-1]["ids"]
    P = {}
    eps = prm["eps"]
    for k in range(10):
        s = [sum(a * b for a, b in zip(theta, phi(F[u], r, k + 1))) for r, u in enumerate(last, 1)]
        mx = max(s)
        e = [math.exp(v - mx) for v in s]
        z = sum(e)
        tgt = 1.0 - pnew[k]
        m = len(last)
        P[k] = {u: (1 - eps * m) * tgt * v / z + eps * tgt for u, v in zip(last, e)}
    build_clogit.theta = theta
    return P, pnew


# pool weights fitted by EM on the backtest cuts (leave-one-cut-out -35.41 vs -35.68 for the
# single Gaussian simulation)
POOL = [
    ("mc_reflected_gumbel", 0.510, {"noise": "ngumbel"}),
    ("mc_student_t", 0.149, {"noise": "student"}),
    ("conditional_logit", 0.341, None),
]


# a deterministic guard on the input size (not on wall time, which would break the
# byte-for-byte rerun): a very long history runs only the pool's main forecaster
MAX_READS_FULL_POOL = 260


def build_pool(reads, task):
    parts = []
    pool = POOL if len(reads) <= MAX_READS_FULL_POOL else POOL[:1]
    for _name, w, prm in pool:
        try:
            if prm is None:
                P, _ = build_clogit(reads, task)
            else:
                P, _ = build(reads, task, prm)
            parts.append((w, P))
        except Exception:  # a failing forecaster drops out; the others are renormalised
            continue
    if not parts:
        raise RuntimeError("no forecaster ran")
    tw = sum(w for w, _ in parts)
    out = {}
    for k in range(PLACES):
        d = {}
        for w, P in parts:
            for u, p in P[k].items():
                d[u] = d.get(u, 0.0) + w / tw * p
        out[k] = d
    return out


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    reads, task = load(input_dir)
    try:
        doc = to_doc(build_pool(reads, task), reads[-1]["ids"])
    except Exception:  # never leave the grader without a forecast
        doc = to_doc(fallback(reads), reads[-1]["ids"] if reads else [])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
