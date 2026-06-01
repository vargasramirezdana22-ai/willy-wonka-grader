"""
Reto 01 — Willy Wonka | Flow Shop 3 máquinas
Algoritmo: NEH-multi-start + BL first-improve + ILS híbrido SA
Versión optimizada: menor Cmax + menor runtime
"""

import random
import math
import time
import json
import numpy as np

TIEMPO_POR_INSTANCIA = 7.8  # margen de seguridad

# ─── Mapeo de tipos ──────────────────────────────────────────────────────────
_TYPE_MAP = {"S": 0, "N": 1, "R": 2}


def _precompute(lotes, setup, ids):
    idx = {l: i for i, l in enumerate(ids)}
    n = len(ids)
    P = [None] * n
    Tv = [0] * n
    for l in ids:
        i = idx[l]; d = lotes[l]
        P[i] = (d["M1"], d["M2"], d["M3"])
        Tv[i] = _TYPE_MAP[d["tipo"]]
    # Setup como array plano [machine][from][to] → lista 3x3x3
    S = [[[0]*3 for _ in range(3)] for _ in range(3)]
    for mi, m in enumerate(["M1", "M2", "M3"]):
        for ta in "SNR":
            for ts in "SNR":
                if ta != ts:
                    S[mi][_TYPE_MAP[ta]][_TYPE_MAP[ts]] = setup[m][f"{ta}-{ts}"]
    return P, Tv, S, idx


# ─── Cmax ultra-rápido con early-exit ────────────────────────────────────────
def _cmax(seq, P, Tv, S0, S1, S2):
    tp = -1; C0 = C1 = C2 = 0
    for ii in seq:
        tc = Tv[ii]; p = P[ii]
        if tp == 1 and tc == 2:
            return 10**9
        if tp >= 0 and tp != tc:
            s0 = S0[tp][tc]; s1 = S1[tp][tc]; s2 = S2[tp][tc]
            C0 += s0 + p[0]
            C1 = (C0 if C0 > C1 else C1) + s1 + p[1]
            C2 = (C1 if C1 > C2 else C2) + s2 + p[2]
        else:
            C0 += p[0]
            C1 = (C0 if C0 > C1 else C1) + p[1]
            C2 = (C1 if C1 > C2 else C2) + p[2]
        tp = tc
    return C2


def _cmax_partial(seq, P, Tv, S0, S1, S2, limit):
    """Cmax con early-exit si supera limit."""
    tp = -1; C0 = C1 = C2 = 0
    for ii in seq:
        tc = Tv[ii]; p = P[ii]
        if tp == 1 and tc == 2:
            return 10**9
        if tp >= 0 and tp != tc:
            s0 = S0[tp][tc]; s1 = S1[tp][tc]; s2 = S2[tp][tc]
            C0 += s0 + p[0]
            C1 = (C0 if C0 > C1 else C1) + s1 + p[1]
            C2 = (C1 if C1 > C2 else C2) + s2 + p[2]
        else:
            C0 += p[0]
            C1 = (C0 if C0 > C1 else C1) + p[1]
            C2 = (C1 if C1 > C2 else C2) + p[2]
        if C2 >= limit:
            return C2
        tp = tc
    return C2


def _feasible(seq, Tv):
    for i in range(1, len(seq)):
        if Tv[seq[i-1]] == 1 and Tv[seq[i]] == 2:
            return False
    return True


def _repair(seq, Tv):
    s = list(seq); changed = True
    while changed:
        changed = False
        for i in range(1, len(s)):
            if Tv[s[i-1]] == 1 and Tv[s[i]] == 2:
                item = s.pop(i)
                for j in range(len(s)+1):
                    if j == 0 or Tv[s[j-1]] != 1:
                        s.insert(j, item); break
                changed = True; break
    return s


# ─── NEH optimizado con early-exit ───────────────────────────────────────────
def _neh(init, P, Tv, S0, S1, S2):
    seq = [init[0]]
    for l in init[1:]:
        best_v = 10**9; best_s = None
        for pos in range(len(seq)+1):
            c = seq[:pos] + [l] + seq[pos:]
            if Tv[seq[pos-1]] == 1 and Tv[l] == 2 and pos > 0:
                continue  # skip infactible rápido
            v = _cmax(c, P, Tv, S0, S1, S2)
            if v < best_v:
                best_v = v; best_s = c
        seq = best_s if best_s else seq + [l]
    return seq


# ─── Variantes NEH (mayor diversidad) ────────────────────────────────────────
def _neh_variants(ids, lotes, P, Tv, S0, S1, S2, idx):
    by_type = [[], [], []]
    for l in ids:
        by_type[_TYPE_MAP[lotes[l]["tipo"]]].append(idx[l])
    suma = {i: P[i][0]+P[i][1]+P[i][2] for i in range(len(ids))}
    p0 = {i: P[i][0] for i in range(len(ids))}
    p01 = {i: P[i][0]+P[i][1] for i in range(len(ids))}
    p12 = {i: P[i][1]+P[i][2] for i in range(len(ids))}
    p1 = {i: P[i][1] for i in range(len(ids))}

    best_c = 10**9; best_s = None

    crits = [
        lambda i: -suma[i],
        lambda i: -p0[i],
        lambda i: -p01[i],
        lambda i: -p12[i],
        lambda i: -p1[i],
        lambda i: suma[i],
        lambda i: p0[i],
    ]

    block_orders = [
        [1, 0, 2], [0, 2, 1], [2, 0, 1],
        [2, 1, 0], [0, 1, 2], [1, 2, 0],
    ]

    for bo in block_orders:
        for crit in crits:
            init = []
            for t in bo:
                init += sorted(by_type[t], key=crit)
            seq = _neh(init, P, Tv, S0, S1, S2)
            if not _feasible(seq, Tv):
                seq = _repair(seq, Tv)
            if _feasible(seq, Tv):
                c = _cmax(seq, P, Tv, S0, S1, S2)
                if c < best_c:
                    best_c = c; best_s = seq[:]

    # Intercalado N-S-R + variantes globales
    n_l, s_l, r_l = list(by_type[1]), list(by_type[0]), list(by_type[2])
    interleaved = []; s_idx = 0
    for n in n_l:
        if s_idx < len(s_l): interleaved.append(s_l[s_idx]); s_idx += 1
        interleaved.append(n)
    for r in r_l:
        if s_idx < len(s_l): interleaved.append(s_l[s_idx]); s_idx += 1
        interleaved.append(r)
    while s_idx < len(s_l): interleaved.append(s_l[s_idx]); s_idx += 1

    extra_orders = [
        sorted(range(len(ids)), key=lambda i: -suma[i]),
        sorted(range(len(ids)), key=lambda i: -p0[i]),
        sorted(range(len(ids)), key=lambda i: -p01[i]),
        sorted(range(len(ids)), key=lambda i: -p12[i]),
        sorted(range(len(ids)), key=lambda i: suma[i]),
        interleaved,
        # Aleatorio con semilla fija
        sorted(range(len(ids)), key=lambda i: p0[i]-p12[i]),
        sorted(range(len(ids)), key=lambda i: -(2*p0[i]+p1[i])),
    ]

    for order in extra_orders:
        seq = _neh(order, P, Tv, S0, S1, S2)
        if not _feasible(seq, Tv): seq = _repair(seq, Tv)
        if _feasible(seq, Tv):
            c = _cmax(seq, P, Tv, S0, S1, S2)
            if c < best_c: best_c = c; best_s = seq[:]

    if best_s is None:
        fallback = list(range(len(ids)))
        best_s = _repair(_neh(fallback, P, Tv, S0, S1, S2), Tv)
    return best_s


# ─── Búsqueda local first-improve con todas las vecindades ───────────────────
def _busqueda_local(seq, P, Tv, S0, S1, S2, t_corte=None):
    mejor = seq[:]; mejor_c = _cmax(mejor, P, Tv, S0, S1, S2)
    n = len(mejor); mejorado_global = True

    while mejorado_global:
        if t_corte and time.perf_counter() > t_corte: break
        mejorado_global = False

        # Or-opt 1, 2, 3
        for tam in [1, 2, 3]:
            mejorado = True
            while mejorado:
                if t_corte and time.perf_counter() > t_corte: return mejor, mejor_c
                mejorado = False
                for i in range(n - tam + 1):
                    if mejorado: break
                    seg = mejor[i:i+tam]
                    base = mejor[:i] + mejor[i+tam:]
                    for j in range(len(base)+1):
                        if j > 0 and Tv[base[j-1]] == 1 and Tv[seg[0]] == 2:
                            continue
                        cand = base[:j] + seg + base[j:]
                        if not _feasible(cand, Tv): continue
                        c = _cmax_partial(cand, P, Tv, S0, S1, S2, mejor_c)
                        if c < mejor_c:
                            mejor_c = c; mejor = cand
                            mejorado = True; mejorado_global = True; break

        # 2-opt swap
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte: return mejor, mejor_c
            mejorado = False
            for i in range(n):
                if mejorado: break
                for j in range(i+1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    if not _feasible(cand, Tv): continue
                    c = _cmax_partial(cand, P, Tv, S0, S1, S2, mejor_c)
                    if c < mejor_c:
                        mejor_c = c; mejor = cand
                        mejorado = True; mejorado_global = True; break

        # Or-opt invertido
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte: return mejor, mejor_c
            mejorado = False
            for i in range(n-1):
                if mejorado: break
                for tam in [2, 3]:
                    if i+tam > n: continue
                    seg_inv = mejor[i:i+tam][::-1]
                    base = mejor[:i] + mejor[i+tam:]
                    for j in range(len(base)+1):
                        cand = base[:j] + seg_inv + base[j:]
                        if not _feasible(cand, Tv): continue
                        c = _cmax_partial(cand, P, Tv, S0, S1, S2, mejor_c)
                        if c < mejor_c:
                            mejor_c = c; mejor = cand
                            mejorado = True; mejorado_global = True; break

        # 3-opt inversión de subsecuencia
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte: return mejor, mejor_c
            mejorado = False
            for i in range(n-2):
                if mejorado: break
                for j in range(i+2, n):
                    cand = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                    if not _feasible(cand, Tv): continue
                    c = _cmax_partial(cand, P, Tv, S0, S1, S2, mejor_c)
                    if c < mejor_c:
                        mejor_c = c; mejor = cand
                        mejorado = True; mejorado_global = True; break

        # 4-opt: reinserción de par de lotes no contiguos
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte: return mejor, mejor_c
            mejorado = False
            for i in range(n):
                if mejorado: break
                for j in range(i+2, n):
                    # Extraer i y j, reinsertarlos juntos en otra posición
                    a, b = mejor[i], mejor[j]
                    base = [x for k, x in enumerate(mejor) if k != i and k != j]
                    for k in range(len(base)+1):
                        for order in [[a, b], [b, a]]:
                            cand = base[:k] + order + base[k:]
                            if not _feasible(cand, Tv): continue
                            c = _cmax_partial(cand, P, Tv, S0, S1, S2, mejor_c)
                            if c < mejor_c:
                                mejor_c = c; mejor = cand
                                mejorado = True; mejorado_global = True; break
                        if mejorado: break

    return mejor, mejor_c


# ─── Búsqueda local rápida (solo Or-opt1 + swap) para ILS ────────────────────
def _bl_rapida(seq, P, Tv, S0, S1, S2, t_corte):
    mejor = seq[:]; mejor_c = _cmax(mejor, P, Tv, S0, S1, S2)
    n = len(mejor); mejorado_global = True

    while mejorado_global:
        if time.perf_counter() > t_corte: break
        mejorado_global = False

        # Or-opt 1
        mejorado = True
        while mejorado:
            if time.perf_counter() > t_corte: return mejor, mejor_c
            mejorado = False
            for i in range(n):
                if mejorado: break
                item = mejor[i]; base = mejor[:i] + mejor[i+1:]
                for j in range(len(base)+1):
                    if j > 0 and Tv[base[j-1]] == 1 and Tv[item] == 2: continue
                    cand = base[:j] + [item] + base[j:]
                    if not _feasible(cand, Tv): continue
                    c = _cmax_partial(cand, P, Tv, S0, S1, S2, mejor_c)
                    if c < mejor_c:
                        mejor_c = c; mejor = cand
                        mejorado = True; mejorado_global = True; break

        # swap
        mejorado = True
        while mejorado:
            if time.perf_counter() > t_corte: return mejor, mejor_c
            mejorado = False
            for i in range(n):
                if mejorado: break
                for j in range(i+1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    if not _feasible(cand, Tv): continue
                    c = _cmax_partial(cand, P, Tv, S0, S1, S2, mejor_c)
                    if c < mejor_c:
                        mejor_c = c; mejor = cand
                        mejorado = True; mejorado_global = True; break

    return mejor, mejor_c


# ─── Perturbaciones ──────────────────────────────────────────────────────────
def _double_bridge(seq):
    n = len(seq)
    if n < 8: return seq[:]
    pos = sorted(random.sample(range(1, n), 3))
    a, b, c = pos
    return seq[0:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturb_guided(seq, Tv):
    tps = [[], [], []]
    for i, ii in enumerate(seq): tps[Tv[ii]].append(i)
    order = [0, 1, 2]; random.shuffle(order)
    for tp in order:
        if len(tps[tp]) < 2: continue
        k = random.randint(1, min(3, len(tps[tp])))
        sel = sorted(random.sample(tps[tp], k), reverse=True)
        nuevo = seq[:]; ext = []
        for i2 in sel: ext.insert(0, nuevo.pop(i2))
        pos = random.randint(0, len(nuevo))
        nuevo = nuevo[:pos] + ext + nuevo[pos:]
        if _feasible(nuevo, Tv): return nuevo
    return _double_bridge(seq)


def _perturb_rand(seq, Tv):
    nuevo = seq[:]; k = random.randint(2, min(5, len(seq)))
    indices = sorted(random.sample(range(len(nuevo)), k), reverse=True)
    ext = [nuevo.pop(i) for i in indices]; random.shuffle(ext)
    for l in ext: nuevo.insert(random.randint(0, len(nuevo)), l)
    return nuevo if _feasible(nuevo, Tv) else _repair(nuevo, Tv)


def _perturb_swap(seq, Tv):
    by_t = [[], [], []]
    for i, ii in enumerate(seq): by_t[Tv[ii]].append(i)
    tps = [t for t in range(3) if by_t[t]]
    if len(tps) < 2: return _double_bridge(seq)
    random.shuffle(tps); t1, t2 = tps[0], tps[1]
    i1 = random.choice(by_t[t1]); i2 = random.choice(by_t[t2])
    nuevo = seq[:]; nuevo[i1], nuevo[i2] = nuevo[i2], nuevo[i1]
    return nuevo if _feasible(nuevo, Tv) else _repair(nuevo, Tv)


def _perturb_block_move(seq, Tv):
    """Mueve un bloque de 2-4 lotes a otra posición."""
    n = len(seq)
    tam = random.randint(2, min(4, n//2))
    i = random.randint(0, n-tam)
    seg = seq[i:i+tam]; base = seq[:i] + seq[i+tam:]
    j = random.randint(0, len(base))
    nuevo = base[:j] + seg + base[j:]
    return nuevo if _feasible(nuevo, Tv) else _repair(nuevo, Tv)


def _perturb_type_sort(seq, Tv, P):
    """Reordena lotes del mismo tipo por algún criterio."""
    nuevo = seq[:]
    t = random.randint(0, 2)
    positions = [i for i, ii in enumerate(nuevo) if Tv[ii] == t]
    if len(positions) < 2: return nuevo
    items = [nuevo[i] for i in positions]
    crit = random.choice([0, 1, 2])
    items_sorted = sorted(items, key=lambda ii: P[ii][crit], reverse=random.random() < 0.5)
    for i, pos in enumerate(positions):
        nuevo[pos] = items_sorted[i]
    return nuevo if _feasible(nuevo, Tv) else _repair(nuevo, Tv)


# ─── ILS híbrido con SA + pool élite amplio ──────────────────────────────────
def _ils_sa(seq_ini, cmax_ini, P, Tv, S0, S1, S2, t_corte):
    mejor = seq_ini[:]; mejor_c = cmax_ini
    actual = seq_ini[:]; actual_c = cmax_ini
    elite = [(mejor_c, mejor[:])]
    sin_mejora = 0; it = 0
    T = max(mejor_c * 0.06, 1.0)
    T_min = 0.2; alpha = 0.995

    while time.perf_counter() < t_corte:
        t_rest = t_corte - time.perf_counter()
        if t_rest < 0.1: break

        # Reinicio desde élite
        if sin_mejora > 40:
            _, base = random.choice(elite)
            actual = base[:]; actual_c = _cmax(actual, P, Tv, S0, S1, S2)
            T = max(mejor_c * 0.05, 1.0); sin_mejora = 0

        # 7 perturbaciones rotativas
        mv = it % 7
        if mv == 0:   perturb = _double_bridge(actual)
        elif mv == 1: perturb = _perturb_guided(actual, Tv)
        elif mv == 2: perturb = _perturb_rand(actual, Tv)
        elif mv == 3: perturb = _perturb_swap(actual, Tv)
        elif mv == 4: perturb = _double_bridge(_double_bridge(actual))
        elif mv == 5: perturb = _perturb_block_move(actual, Tv)
        else:         perturb = _perturb_type_sort(actual, Tv, P)

        if not _feasible(perturb, Tv): perturb = _repair(perturb, Tv)
        if not _feasible(perturb, Tv): it += 1; continue

        # BL rápida (solo Or-opt1 + swap) durante fracción del tiempo restante
        t_bl = time.perf_counter() + min(t_rest * 0.25, 0.8)
        s_loc, c_loc = _bl_rapida(perturb, P, Tv, S0, S1, S2, t_corte=t_bl)

        delta = c_loc - actual_c
        if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
            actual = s_loc[:]; actual_c = c_loc
            sin_mejora = 0 if delta < 0 else sin_mejora + 1
        else:
            sin_mejora += 1

        if actual_c < mejor_c:
            mejor_c = actual_c; mejor = actual[:]
            elite.append((mejor_c, mejor[:]))
            elite.sort(key=lambda x: x[0])
            elite = elite[:8]  # pool élite más grande

        T = max(T * alpha, T_min); it += 1

    return mejor, mejor_c


# ─── Solver principal ─────────────────────────────────────────────────────────
def solve(data: dict) -> dict:
    start = time.time()
    t0 = time.perf_counter()
    t_fin = t0 + TIEMPO_POR_INSTANCIA

    random.seed(42)
    np.random.seed(42)

    lotes = data["lotes"]; setup = data["setup"]
    ids = list(lotes.keys())

    P, Tv, S, idx = _precompute(lotes, setup, ids)
    S0, S1, S2 = S[0], S[1], S[2]

    # ── Fase 1: NEH multi-variante ────────────────────────────────────────────
    seq = _neh_variants(ids, lotes, P, Tv, S0, S1, S2, idx)
    cmax = _cmax(seq, P, Tv, S0, S1, S2)

    # ── Fase 2: BL exhaustiva (~2.0s) ────────────────────────────────────────
    t_bl = t0 + 2.0
    seq, cmax = _busqueda_local(seq, P, Tv, S0, S1, S2, t_corte=t_bl)

    # ── Fase 3: ILS+SA con tiempo restante ───────────────────────────────────
    if time.perf_counter() < t_fin - 0.5:
        seq_ils, c_ils = _ils_sa(seq, cmax, P, Tv, S0, S1, S2, t_corte=t_fin - 0.15)
        if c_ils < cmax: cmax = c_ils; seq = seq_ils[:]

    # ── BL final de pulido ────────────────────────────────────────────────────
    t_left = t_fin - time.perf_counter()
    if t_left > 0.3:
        seq2, c2 = _busqueda_local(seq, P, Tv, S0, S1, S2, t_corte=t_fin - 0.1)
        if c2 < cmax: cmax = c2; seq = seq2[:]

    # ── Validación ────────────────────────────────────────────────────────────
    if not _feasible(seq, Tv) or cmax >= 10**8:
        seq = _repair(seq, Tv)
        cmax = _cmax(seq, P, Tv, S0, S1, S2)

    inv_idx = {v: k for k, v in idx.items()}
    seq_nombres = [inv_idx[i] for i in seq]

    return {
        "Cmax": int(cmax),
        "secuencia": seq_nombres,
        "tiempo": int((time.time() - start) * 1000),
    }


# ─── Prueba local ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import pandas as pd

    if os.path.exists("data.json"):
        with open("data.json") as f:
            data = json.load(f)
        print("📂 Usando data.json\n")
        resultado = solve(data)
        lotes = data["lotes"]; seq = resultado["secuencia"]
        factible = all(
            not (lotes[seq[i-1]]["tipo"] == "N" and lotes[seq[i]]["tipo"] == "R")
            for i in range(1, len(seq))
        )
        print(f"✅ Factible : {factible}")
        print(f"📦 Secuencia: {seq}")
        print(f"⏱️  Cmax     : {resultado['Cmax']} minutos")
        print(f"🕐 Tiempo   : {resultado['tiempo']} ms")
        print()
        print(json.dumps(resultado, indent=2))
    else:
        # Correr 30 instancias de prueba
        print("⚠️  Sin data.json — corriendo 30 instancias de prueba\n")

        def generar_instancia(semilla_local):
            tipos = ["S", "N", "R"]
            lotes_ids = [f"L{i}" for i in range(1, 16)]
            rangos = {"M1": (25, 45), "M2": (20, 40), "M3": (10, 25)}
            lotes_dict = {}
            for lote in lotes_ids:
                lotes_dict[lote] = {
                    "tipo": random.choice(tipos),
                    "M1": random.randint(*rangos["M1"]),
                    "M2": random.randint(*rangos["M2"]),
                    "M3": random.randint(*rangos["M3"]),
                }
            setup_dict = {}
            for maquina in ["M1", "M2", "M3"]:
                setup_dict[maquina] = {}
                for ta in tipos:
                    for ts in tipos:
                        if ta == ts: t = 0
                        elif maquina == "M1": t = random.randint(10, 40)
                        elif maquina == "M2": t = random.randint(5, 30)
                        else: t = random.randint(3, 15)
                        setup_dict[maquina][f"{ta}-{ts}"] = t
            return {"lotes": lotes_dict, "setup": setup_dict}

        random.seed(42); np.random.seed(42)
        resultados = []
        inicio_total = time.time()
        for i in range(1, 31):
            inst = generar_instancia(i)
            t0 = time.time()
            res = solve(inst)
            t1 = time.time()
            resultados.append(res["Cmax"])
            print(f"Instancia {i:2d} | Cmax={res['Cmax']:>6} | {round(t1-t0,2)}s")

        tiempo_total = time.time() - inicio_total
        print(f"\n==== REPORTE ====")
        print(f"Cmax promedio  : {sum(resultados)/len(resultados):.1f}")
        print(f"Tiempo total   : {tiempo_total:.1f}s")
        print(f"Tiempo promedio: {tiempo_total/30:.2f}s")
