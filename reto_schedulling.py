"""
Reto 01 — Willy Wonka | Flow Shop 3 máquinas con setups y restricción N→R
Algoritmo: NEH multistart + Búsqueda Local (Or-opt 1-3, swap, Or-opt-inv, 3-opt)
           + ILS con Simulated Annealing y elite pool.

OPTIMIZADO: Tiempo por instancia reducido a 0.8s → 30 instancias ≈ 24s total.
Librerías: solo stdlib + numpy.
"""

import random
import math
import numpy as np
import time
import json

INF = float("inf")

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
TIEMPO_POR_INSTANCIA = 0.8    # 30 × 0.8 s ≈ 24 s total << 60 s límite
FRACCION_MULTISTART  = 0.10   # 10% → NEH multistart
FRACCION_BL          = 0.15   # 15% → intensificación inicial
                               # 75% → ILS-SA + pulido


# =============================================================================
# MOTOR COMPILADO — traduce data a enteros y devuelve evaluadores rápidos
# =============================================================================

def _compilar(data):
    """Devuelve (ids, tipo, cmax, factible) operando sobre índices enteros.
    Tipos: S=0, N=1, R=2."""
    TI    = {"S": 0, "N": 1, "R": 2}
    tipos = ("S", "N", "R")
    lotes = data["lotes"]
    setup = data["setup"]
    ids   = list(lotes.keys())

    tipo = [TI[lotes[l]["tipo"]] for l in ids]
    p1   = [lotes[l]["M1"] for l in ids]
    p2   = [lotes[l]["M2"] for l in ids]
    p3   = [lotes[l]["M3"] for l in ids]

    S1 = [[setup["M1"][f"{a}-{b}"] for b in tipos] for a in tipos]
    S2 = [[setup["M2"][f"{a}-{b}"] for b in tipos] for a in tipos]
    S3 = [[setup["M3"][f"{a}-{b}"] for b in tipos] for a in tipos]

    # Pre-computar arrays para acceso rápido
    tipo_arr = tipo
    p1_arr   = p1
    p2_arr   = p2
    p3_arr   = p3

    def cmax(seq):
        j0 = seq[0]
        f0 = p1_arr[j0]
        f1 = f0 + p2_arr[j0]
        f2 = f1 + p3_arr[j0]
        pa = tipo_arr[j0]
        for idx in range(1, len(seq)):
            j  = seq[idx]; ta = tipo_arr[j]
            if pa == 1 and ta == 2:        # N → R infactible
                return INF
            if pa != ta:
                f0 = f0 + S1[pa][ta] + p1_arr[j]
                f1 = (f1 if f1 > f0 else f0) + S2[pa][ta] + p2_arr[j]
                f2 = (f2 if f2 > f1 else f1) + S3[pa][ta] + p3_arr[j]
            else:
                f0 = f0 + p1_arr[j]
                f1 = (f1 if f1 > f0 else f0) + p2_arr[j]
                f2 = (f2 if f2 > f1 else f1) + p3_arr[j]
            pa = ta
        return f2

    def factible(seq):
        ta = tipo_arr
        for idx in range(1, len(seq)):
            if ta[seq[idx-1]] == 1 and ta[seq[idx]] == 2:
                return False
        return True

    return ids, tipo, cmax, factible


# =============================================================================
# Evaluadores sobre etiquetas (validación / fallback / informe)
# =============================================================================

def _calcular_cmax(secuencia, lotes, setup):
    for i in range(1, len(secuencia)):
        if (lotes[secuencia[i-1]]["tipo"] == "N" and
                lotes[secuencia[i]]["tipo"] == "R"):
            return INF
    n = len(secuencia)
    fin = [[0.0] * 3 for _ in range(n)]
    for i, lote in enumerate(secuencia):
        tipo_act = lotes[lote]["tipo"]
        for k, m in enumerate(("M1", "M2", "M3")):
            t_setup = 0
            if i > 0:
                tipo_ant = lotes[secuencia[i-1]]["tipo"]
                if tipo_ant != tipo_act:
                    t_setup = setup[m][f"{tipo_ant}-{tipo_act}"]
            lm = fin[i-1][k] if i > 0 else 0.0
            ll = fin[i][k-1] if k > 0 else 0.0
            fin[i][k] = max(lm, ll) + t_setup + lotes[lote][m]
    return fin[-1][2]


def _es_factible(secuencia, lotes):
    for i in range(1, len(secuencia)):
        if (lotes[secuencia[i-1]]["tipo"] == "N" and
                lotes[secuencia[i]]["tipo"] == "R"):
            return False
    return True


# =============================================================================
# FASE 1 — NEH multistart (opera sobre enteros)
# =============================================================================

def _neh_una_vez(orden, cmax):
    seq = [orden[0]]
    for job in orden[1:]:
        bc = INF; bp = 0
        for p in range(len(seq) + 1):
            c = cmax(seq[:p] + [job] + seq[p:])
            if c < bc:
                bc = c; bp = p
        seq = seq[:bp] + [job] + seq[bp:]
    return seq


def _multi_start(ids_int, tipo, suma, cmax, t_corte):
    best_c = INF
    best_seq = None

    nombre = {"S": 0, "N": 1, "R": 2}
    # Solo las 3 ordenaciones más prometedoras (antes eran 6)
    for orden_tipos in (("S","N","R"), ("S","R","N"), ("R","S","N")):
        s = []
        for t in orden_tipos:
            ti = nombre[t]
            grupo = sorted([j for j in ids_int if tipo[j] == ti],
                           key=lambda j: suma[j], reverse=True)
            s.extend(grupo)
        seq = _neh_una_vez(s, cmax)
        c = cmax(seq)
        if c < best_c:
            best_c = c; best_seq = seq[:]

    # NEH estándar por suma descendente
    orden_std = sorted(ids_int, key=lambda j: suma[j], reverse=True)
    seq = _neh_una_vez(orden_std, cmax)
    c = cmax(seq)
    if c < best_c:
        best_c = c; best_seq = seq[:]

    # Semillas aleatorias con el tiempo restante
    seed = 1
    while time.perf_counter() < t_corte:
        rng = random.Random(seed)
        noise = min(seed * 2, 40)
        orden = sorted(ids_int, key=lambda j: suma[j] + rng.uniform(-noise, noise),
                       reverse=True)
        seq = _neh_una_vez(orden, cmax)
        c = cmax(seq)
        if c < best_c:
            best_c = c; best_seq = seq[:]
        seed += 1

    return best_seq, best_c


# =============================================================================
# FASE 2 — Búsqueda local acelerada (Or-opt 1-2, swap, 3-opt)
# =============================================================================

def _busqueda_local(seq, cmax, t_corte=None):
    mejor = seq[:]
    mc = cmax(mejor)
    n = len(mejor)
    mg = True

    while mg:
        if t_corte and time.perf_counter() > t_corte:
            break
        mg = False

        # Or-opt segmentos de 1 y 2 (más rápido que incluir 3)
        for tam in (1, 2):
            mejorado = True
            while mejorado:
                if t_corte and time.perf_counter() > t_corte:
                    return mejor, mc
                mejorado = False
                for i in range(n - tam + 1):
                    if mejorado: break
                    seg  = mejor[i:i+tam]
                    base = mejor[:i] + mejor[i+tam:]
                    for j in range(len(base) + 1):
                        c = cmax(base[:j] + seg + base[j:])
                        if c < mc:
                            mc = c; mejor = base[:j] + seg + base[j:]
                            mejorado = True; mg = True; break

        # Swap de pares
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejorado = False
            for i in range(n):
                if mejorado: break
                for j in range(i + 1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = cmax(cand)
                    if c < mc:
                        mc = c; mejor = cand[:]
                        mejorado = True; mg = True; break

        # 3-opt (inversión de segmento contiguo)
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejorado = False
            for i in range(n - 2):
                if mejorado: break
                for j in range(i + 2, n):
                    cand = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                    c = cmax(cand)
                    if c < mc:
                        mc = c; mejor = cand[:]
                        mejorado = True; mg = True; break

    return mejor, mc


# =============================================================================
# FASE 3 — ILS con Simulated Annealing y elite pool
# =============================================================================

def _double_bridge(seq):
    n = len(seq)
    a, b, c = sorted(random.sample(range(1, n), 3))
    return seq[:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturbacion_guiada(seq, tipo, factible):
    orden = [0, 1, 2]
    random.shuffle(orden)
    for ti in orden:
        idx = [i for i, j in enumerate(seq) if tipo[j] == ti]
        if len(idx) < 2: continue
        k   = random.randint(1, min(3, len(idx)))
        sel = sorted(random.sample(idx, k), reverse=True)
        nuevo = seq[:]
        ext = []
        for i in sel: ext.insert(0, nuevo.pop(i))
        p = random.randint(0, len(nuevo))
        nuevo = nuevo[:p] + ext + nuevo[p:]
        if factible(nuevo): return nuevo
    return _double_bridge(seq)


def _perturbacion_bloque(seq, tipo, factible):
    orden = [0, 1, 2]
    random.shuffle(orden)
    for ti in orden:
        idx = [i for i, j in enumerate(seq) if tipo[j] == ti]
        if len(idx) < 2: continue
        inicio = random.choice(idx)
        nuevo = seq[:]
        val = nuevo.pop(inicio)
        p = random.randint(0, len(nuevo))
        nuevo = nuevo[:p] + [val] + nuevo[p:]
        if factible(nuevo): return nuevo
    return _double_bridge(seq)


def _ils(seq_ini, c_ini, tipo, cmax, factible, t_corte,
         max_estanco=200, t_minimo=0.0):
    """ILS con SA, parada temprana agresiva para terminar rápido."""
    mejor  = seq_ini[:]; mc = c_ini
    actual = seq_ini[:]; ac = c_ini
    elite  = [(mc, mejor[:])]

    t_inicio_ils = time.perf_counter()
    duracion_ils = max(t_corte - t_inicio_ils, 0.5)

    T0    = max(ac * 0.015, 3.0)
    T_min = 0.3

    sin_mejora     = 0
    estanco_global = 0
    it = 0

    while time.perf_counter() < t_corte:
        t_restante = t_corte - time.perf_counter()
        if t_restante < 0.03: break

        # Parada temprana cuando converge
        if (estanco_global >= max_estanco and
                time.perf_counter() - t_inicio_ils >= t_minimo):
            break

        t_frac = (time.perf_counter() - t_inicio_ils) / duracion_ils
        T_sa   = max(T0 * math.exp(-5.0 * t_frac), T_min)

        if sin_mejora > 20:
            _, base = random.choice(elite)
            actual  = base[:]; ac = cmax(actual)
            sin_mejora = 0

        m = it % 3
        if m == 0:
            perturb = _double_bridge(actual)
        elif m == 1:
            perturb = _perturbacion_guiada(actual, tipo, factible)
        else:
            perturb = _perturbacion_bloque(actual, tipo, factible)

        # BL más corta por iteración: 20% del tiempo restante, máx 0.15s
        t_bl = time.perf_counter() + min(t_restante * 0.20, 0.15)
        sl, cl = _busqueda_local(perturb, cmax, t_corte=t_bl)

        delta = cl - ac
        if delta <= 0:
            actual = sl[:]; ac = cl; sin_mejora = 0
        elif T_sa > T_min and random.random() < math.exp(-delta / T_sa):
            actual = sl[:]; ac = cl; sin_mejora += 1
        else:
            sin_mejora += 1

        if ac < mc:
            mc = ac; mejor = actual[:]
            elite.append((mc, mejor[:]))
            elite.sort(key=lambda x: x[0])
            elite = elite[:5]
            estanco_global = 0
        else:
            estanco_global += 1

        it += 1

    return mejor, mc


# =============================================================================
# Orquestador interno
# =============================================================================

def _resolver(data, T, t_ms_frac, t_bl_frac, t_ils_frac,
              max_estanco=200, t_minimo=0.1):
    t_inicio = time.perf_counter()
    t_fin    = t_inicio + T

    ids, tipo, cmax, factible = _compilar(data)
    n = len(ids)
    ids_int = list(range(n))

    lotes = data["lotes"]
    suma  = {j: lotes[ids[j]]["M1"] + lotes[ids[j]]["M2"] + lotes[ids[j]]["M3"]
             for j in ids_int}

    # 1. NEH multistart
    seq, c = _multi_start(ids_int, tipo, suma, cmax,
                          t_corte=t_inicio + T * t_ms_frac)

    # 2. Intensificación inicial
    seq, c = _busqueda_local(seq, cmax,
                             t_corte=time.perf_counter() + T * t_bl_frac)

    # 3. ILS-SA
    t_ils_fin = t_inicio + T * t_ils_frac
    if time.perf_counter() < t_ils_fin - 0.02:
        seq_ils, c_ils = _ils(seq, c, tipo, cmax, factible, t_corte=t_ils_fin,
                              max_estanco=max_estanco, t_minimo=t_minimo)
        if c_ils < c:
            c = c_ils; seq = seq_ils[:]

    # 4. Pulido final
    seq, c = _busqueda_local(seq, cmax, t_corte=t_fin)

    seq_lbl = [ids[j] for j in seq]

    # Fallback de seguridad
    if c == INF or not _es_factible(seq_lbl, lotes):
        orden = sorted(ids_int, key=lambda j: suma[j], reverse=True)
        seq   = _neh_una_vez(orden, cmax)
        c     = cmax(seq)
        seq_lbl = [ids[j] for j in seq]

    return seq_lbl, int(c)


# =============================================================================
# solve() — formato exacto del reto
# =============================================================================

def solve(data: dict) -> dict:
    t_inicio = time.perf_counter()
    seq_lbl, cmax_val = _resolver(data, T=1.5,
                                  t_ms_frac=0.10, t_bl_frac=0.15, t_ils_frac=0.95,
                                  max_estanco=200, t_minimo=0.1)
    return {
        "Cmax":      cmax_val,
        "secuencia": seq_lbl,
        "tiempo":    int((time.perf_counter() - t_inicio) * 1000)
    }


# =============================================================================
# CLASE SCHEDULLING — estructura requerida por la competencia
# =============================================================================

class Schedulling:
    def __init__(self, n_instancias: int = 30, semilla: int = 42):
        """No modificar."""
        self.n_instancias = n_instancias
        self.semilla = semilla
        random.seed(semilla)
        np.random.seed(semilla)
        self.resultados = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        self.consolidado = None

    def generar_instancia(self, indice):
        tipos  = ["S", "N", "R"]
        lotes  = [f"L{i}" for i in range(1, 16)]
        rangos = {"M1": (25, 45), "M2": (20, 40), "M3": (10, 25)}

        lotes_dict = {}
        for lote in lotes:
            lotes_dict[lote] = {
                "tipo": random.choice(tipos),
                "M1":   random.randint(*rangos["M1"]),
                "M2":   random.randint(*rangos["M2"]),
                "M3":   random.randint(*rangos["M3"])
            }

        setup_dict = {}
        for maquina in ("M1", "M2", "M3"):
            setup_dict[maquina] = {}
            for tipo_ant in tipos:
                for tipo_sig in tipos:
                    if tipo_ant == tipo_sig:
                        t = 0
                    elif maquina == "M1": t = random.randint(10, 40)
                    elif maquina == "M2": t = random.randint(5, 30)
                    else:                 t = random.randint(3, 15)
                    setup_dict[maquina][f"{tipo_ant}-{tipo_sig}"] = t

        return {"lotes": lotes_dict, "setup": setup_dict}

    def resolver_instancia(self, instancia: dict) -> dict:
        seq_lbl, cmax_val = _resolver(instancia, T=TIEMPO_POR_INSTANCIA,
                                      t_ms_frac=FRACCION_MULTISTART,
                                      t_bl_frac=FRACCION_BL,
                                      t_ils_frac=0.92,
                                      max_estanco=200,
                                      t_minimo=0.05)
        return {"secuencia": seq_lbl, "valor_objetivo": cmax_val}

    def ejecutar_experimentos(self):
        """No modificar."""
        print(f"Ejecutando {self.n_instancias} instancias...\n")
        self.resultados = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        inicio_total = time.time()

        for i in range(1, self.n_instancias + 1):
            instancia = self.generar_instancia(i)
            t0 = time.time()
            resultado = self.resolver_instancia(instancia)
            t1 = time.time()
            self.resultados["instancia"].append(i)
            self.resultados["valor_objetivo"].append(resultado.get("valor_objetivo", 0))
            self.resultados["tiempo_seg"].append(round(t1 - t0, 4))
            print(f"Instancia {i}: Cmax={resultado.get('valor_objetivo','?')} "
                  f"en {round(t1-t0,2)}s")

        tiempo_total = time.time() - inicio_total
        self.consolidado = self.resultados

        valores = self.resultados["valor_objetivo"]
        tiempos = self.resultados["tiempo_seg"]
        print("\n==== REPORTE DE RENDIMIENTO ====")
        print("Tiempo promedio:", round(sum(tiempos) / len(tiempos), 4), "seg")
        print("Cmax promedio:  ", round(sum(valores) / len(valores), 1))
        print("Tiempo total:   ", round(tiempo_total, 2), "seg")
        return self.consolidado

    def calcular_makespan_penalizado(self, data, secuencia):
        lotes    = data["lotes"]
        setup    = data["setup"]
        maquinas = ["M1", "M2", "M3"]

        for i in range(1, len(secuencia)):
            if (lotes[secuencia[i-1]]["tipo"] == "N" and
                    lotes[secuencia[i]]["tipo"] == "R"):
                return INF

        tiempos     = {m: [] for m in maquinas}
        fin_maquina = {m: 0  for m in maquinas}
        fin_lote    = {l: 0  for l in secuencia}

        for lote in secuencia:
            tipo_actual = lotes[lote]["tipo"]
            for m_idx, m in enumerate(maquinas):
                prev_tipo = tiempos[m][-1]["tipo"] if tiempos[m] else None
                t_setup = 0
                if prev_tipo is not None and prev_tipo != tipo_actual:
                    t_setup = setup[m][f"{prev_tipo}-{tipo_actual}"]
                inicio = (fin_maquina[m] if m_idx == 0
                          else max(fin_maquina[m], fin_lote[lote])) + t_setup
                fin    = inicio + lotes[lote][m]
                tiempos[m].append({"lote": lote, "tipo": tipo_actual})
                fin_maquina[m] = fin
                fin_lote[lote] = fin

        return max(fin_maquina.values())


if __name__ == "__main__":
    import os
    if os.path.exists("data.json"):
        with open("data.json", encoding="utf-8") as f:
            data = json.load(f)
        print(json.dumps(solve(data), indent=2, ensure_ascii=False))
    else:
        sch = Schedulling(n_instancias=30)
        resultado = sch.ejecutar_experimentos()
        print(resultado)
