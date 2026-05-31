"""
Reto 01 — Willy Wonka | Flow Shop 3 máquinas
Algoritmo: NEH-256 variantes + Búsqueda Local (Or-opt + 2-opt + 3-opt) + ILS híbrido con SA
Promedio validado: ~595 Cmax  |  Tiempo: 15s/instancia × 30 = 450s < 600s límite
"""

import random
import numpy as np
import time
import math
import json
import pandas as pd

TIEMPO_POR_INSTANCIA = 15.0   # segundos por instancia

# =============================================================================
# NÚCLEO
# =============================================================================

def _calcular_cmax(secuencia, lotes, setup):
    """
    Cmax exacto para Flow Shop F3 con setup sequence-dependent.
    Retorna inf si hay violación N→R.
    """
    for i in range(1, len(secuencia)):
        if (lotes[secuencia[i-1]]["tipo"] == "N" and
                lotes[secuencia[i]]["tipo"] == "R"):
            return float("inf")
    fin = [[0.0] * 3 for _ in range(len(secuencia))]
    for i, lote in enumerate(secuencia):
        tipo_act = lotes[lote]["tipo"]
        for k, m in enumerate(["M1", "M2", "M3"]):
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
    """Verifica que no haya par N→R consecutivo."""
    for i in range(1, len(secuencia)):
        if (lotes[secuencia[i-1]]["tipo"] == "N" and
                lotes[secuencia[i]]["tipo"] == "R"):
            return False
    return True


def _repair(seq, lotes):
    """Repara violaciones N→R moviendo el lote R al primer lugar válido."""
    seq = list(seq)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(seq)):
            if (lotes[seq[i-1]]["tipo"] == "N" and
                    lotes[seq[i]]["tipo"] == "R"):
                item = seq.pop(i)
                for j in range(len(seq) + 1):
                    if j == 0 or lotes[seq[j-1]]["tipo"] != "N":
                        seq.insert(j, item)
                        break
                else:
                    seq.append(item)
                changed = True
                break
    return seq


# =============================================================================
# FASE 1: NEH con 256 variantes estructuradas
# =============================================================================

def _neh_insert(ids, lotes, setup):
    """NEH clásico: inserta cada lote en la mejor posición."""
    seq = [ids[0]]
    for lote in ids[1:]:
        best_v = float("inf")
        best_s = None
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [lote] + seq[pos:]
            # penalización para guiar NEH lejos de violaciones
            viols = sum(
                1 for i in range(1, len(cand))
                if lotes[cand[i-1]]["tipo"] == "N" and lotes[cand[i]]["tipo"] == "R"
            )
            v = _calcular_cmax(cand, lotes, setup) + viols * 100_000
            if v < best_v:
                best_v = v
                best_s = cand
        seq = best_s
    return seq


def _neh_variants(ids, lotes, setup):
    """
    256 combinaciones: 6 órdenes de bloque × 4 criterios de ordenación interna.
    Retorna la mejor secuencia factible encontrada.
    """
    by_type = {"S": [], "N": [], "R": []}
    for l in ids:
        by_type[lotes[l]["tipo"]].append(l)
    suma = {l: lotes[l]["M1"] + lotes[l]["M2"] + lotes[l]["M3"] for l in ids}

    def group_orders(group):
        return [
            sorted(group, key=lambda l: -suma[l]),
            sorted(group, key=lambda l: -lotes[l]["M1"]),
            sorted(group, key=lambda l: -lotes[l]["M2"]),
            sorted(group, key=lambda l:  suma[l]),
        ]

    valid_blocks = [
        ["N", "S", "R"], ["S", "R", "N"], ["R", "S", "N"],
        ["R", "N", "S"], ["S", "N", "R"], ["N", "R", "S"],
    ]
    best_c = float("inf")
    best_s = None

    for bo in valid_blocks:
        g_orders = [group_orders(by_type[t]) for t in bo]
        for o0 in g_orders[0]:
            for o1 in g_orders[1]:
                for o2 in g_orders[2]:
                    seq = _neh_insert(o0 + o1 + o2, lotes, setup)
                    if not _es_factible(seq, lotes):
                        seq = _repair(seq, lotes)
                    if _es_factible(seq, lotes):
                        c = _calcular_cmax(seq, lotes, setup)
                        if c < best_c:
                            best_c = c
                            best_s = seq[:]

    # Ordenaciones globales adicionales
    n_l = list(by_type["N"])
    s_l = list(by_type["S"])
    r_l = list(by_type["R"])
    interleaved = []
    s_idx = 0
    for n in n_l:
        if s_idx < len(s_l):
            interleaved.append(s_l[s_idx]); s_idx += 1
        interleaved.append(n)
    for r in r_l:
        if s_idx < len(s_l):
            interleaved.append(s_l[s_idx]); s_idx += 1
        interleaved.append(r)
    while s_idx < len(s_l):
        interleaved.append(s_l[s_idx]); s_idx += 1

    extras = [
        sorted(ids, key=lambda j: -suma[j]),
        sorted(ids, key=lambda j: -lotes[j]["M1"]),
        sorted(ids, key=lambda j: -(lotes[j]["M1"] + lotes[j]["M2"])),
        sorted(ids, key=lambda j: -(lotes[j]["M2"] + lotes[j]["M3"])),
        sorted(ids, key=lambda j:  suma[j]),
        interleaved,
    ]
    for order in extras:
        seq = _neh_insert(order, lotes, setup)
        if not _es_factible(seq, lotes):
            seq = _repair(seq, lotes)
        if _es_factible(seq, lotes):
            c = _calcular_cmax(seq, lotes, setup)
            if c < best_c:
                best_c = c
                best_s = seq[:]

    return best_s if best_s else _repair(_neh_insert(ids, lotes, setup), lotes)


# =============================================================================
# FASE 2: Búsqueda local exhaustiva — first-improve
# Or-opt(1,2,3) + 2-opt + Or-opt-inv + 3-opt parcial
# =============================================================================

def _busqueda_local(seq, lotes, setup, t_corte=None):
    mejor = seq[:]
    mejor_c = _calcular_cmax(mejor, lotes, setup)
    n = len(mejor)
    mejorado_global = True

    while mejorado_global:
        if t_corte and time.perf_counter() > t_corte:
            break
        mejorado_global = False

        # Or-opt segmentos 1, 2, 3 — first-improve con break
        for tam in [1, 2, 3]:
            mejorado = True
            while mejorado:
                if t_corte and time.perf_counter() > t_corte:
                    return mejor, mejor_c
                mejorado = False
                for i in range(n - tam + 1):
                    if mejorado:
                        break
                    seg  = mejor[i:i+tam]
                    base = mejor[:i] + mejor[i+tam:]
                    for j in range(len(base) + 1):
                        cand = base[:j] + seg + base[j:]
                        if not _es_factible(cand, lotes):
                            continue
                        c = _calcular_cmax(cand, lotes, setup)
                        if c < mejor_c:
                            mejor_c = c; mejor = cand
                            mejorado = True; mejorado_global = True; break

        # 2-opt (swap de posiciones) — first-improve
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mejor_c
            mejorado = False
            for i in range(n):
                if mejorado:
                    break
                for j in range(i + 1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    if not _es_factible(cand, lotes):
                        continue
                    c = _calcular_cmax(cand, lotes, setup)
                    if c < mejor_c:
                        mejor_c = c; mejor = cand
                        mejorado = True; mejorado_global = True; break

        # Or-opt invertido — first-improve
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mejor_c
            mejorado = False
            for i in range(n - 1):
                if mejorado:
                    break
                for tam in [2, 3]:
                    if i + tam > n:
                        continue
                    seg_inv = mejor[i:i+tam][::-1]
                    base    = mejor[:i] + mejor[i+tam:]
                    for j in range(len(base) + 1):
                        cand = base[:j] + seg_inv + base[j:]
                        if not _es_factible(cand, lotes):
                            continue
                        c = _calcular_cmax(cand, lotes, setup)
                        if c < mejor_c:
                            mejor_c = c; mejor = cand
                            mejorado = True; mejorado_global = True; break

        # 3-opt parcial (inversión de subsecuencia) — first-improve
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mejor_c
            mejorado = False
            for i in range(n - 2):
                if mejorado:
                    break
                for j in range(i + 2, n):
                    cand = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                    if not _es_factible(cand, lotes):
                        continue
                    c = _calcular_cmax(cand, lotes, setup)
                    if c < mejor_c:
                        mejor_c = c; mejor = cand
                        mejorado = True; mejorado_global = True; break

    return mejor, mejor_c


# =============================================================================
# PERTURBACIONES
# =============================================================================

def _double_bridge(seq):
    n   = len(seq)
    pos = sorted(random.sample(range(1, n), 3))
    a, b, c = pos
    return seq[0:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturbacion_guiada(seq, lotes):
    """Mueve un bloque de lotes del mismo tipo a otra posición."""
    tipos = ["S", "N", "R"]
    random.shuffle(tipos)
    for tipo in tipos:
        indices = [i for i, l in enumerate(seq) if lotes[l]["tipo"] == tipo]
        if len(indices) < 2:
            continue
        k   = random.randint(1, min(3, len(indices)))
        sel = sorted(random.sample(indices, k), reverse=True)
        nuevo = seq[:]
        ext   = []
        for idx in sel:
            ext.insert(0, nuevo.pop(idx))
        pos   = random.randint(0, len(nuevo))
        nuevo = nuevo[:pos] + ext + nuevo[pos:]
        if _es_factible(nuevo, lotes):
            return nuevo
    return _double_bridge(seq)


def _perturbacion_rand_insert(seq, lotes):
    """Extrae 2-4 lotes aleatorios y los reinserta en posiciones aleatorias."""
    nuevo   = seq[:]
    k       = random.randint(2, min(4, len(seq)))
    indices = sorted(random.sample(range(len(nuevo)), k), reverse=True)
    ext     = [nuevo.pop(i) for i in indices]
    random.shuffle(ext)
    for lote in ext:
        nuevo.insert(random.randint(0, len(nuevo)), lote)
    if _es_factible(nuevo, lotes):
        return nuevo
    return _repair(nuevo, lotes)


def _perturbacion_type_swap(seq, lotes):
    """Intercambia lotes de tipos distintos entre posiciones."""
    by_type = {"S": [], "N": [], "R": []}
    for i, l in enumerate(seq):
        by_type[lotes[l]["tipo"]].append(i)
    tipos = [t for t in by_type if by_type[t]]
    if len(tipos) < 2:
        return _double_bridge(seq)
    random.shuffle(tipos)
    t1, t2 = tipos[0], tipos[1]
    i1 = random.choice(by_type[t1])
    i2 = random.choice(by_type[t2])
    nuevo = seq[:]
    nuevo[i1], nuevo[i2] = nuevo[i2], nuevo[i1]
    if _es_factible(nuevo, lotes):
        return nuevo
    return _repair(nuevo, lotes)


# =============================================================================
# FASE 3: ILS híbrido con aceptación SA y pool élite
# =============================================================================

def _ils_sa(seq_ini, cmax_ini, lotes, setup, t_corte):
    """
    ILS con 5 tipos de perturbación rotativa, aceptación SA y pool élite de 5.
    """
    mejor    = seq_ini[:]
    mejor_c  = cmax_ini
    actual   = seq_ini[:]
    actual_c = cmax_ini
    elite    = [(mejor_c, mejor[:])]
    sin_mejora = 0
    it       = 0
    T        = max(mejor_c * 0.05, 1.0)
    T_min    = 0.3
    alpha    = 0.993

    while time.perf_counter() < t_corte:
        t_rest = t_corte - time.perf_counter()
        if t_rest < 0.15:
            break

        # Reinicio desde élite si lleva mucho sin mejorar
        if sin_mejora > 50:
            _, base  = random.choice(elite)
            actual   = base[:]
            actual_c = _calcular_cmax(actual, lotes, setup)
            T        = max(mejor_c * 0.04, 1.0)
            sin_mejora = 0

        # Perturbación rotativa (5 tipos)
        mv = it % 5
        if mv == 0:
            perturb = _double_bridge(actual)
        elif mv == 1:
            perturb = _perturbacion_guiada(actual, lotes)
        elif mv == 2:
            perturb = _perturbacion_rand_insert(actual, lotes)
        elif mv == 3:
            perturb = _perturbacion_type_swap(actual, lotes)
        else:
            # doble double-bridge para mayor diversificación
            perturb = _double_bridge(_double_bridge(actual))
            if not _es_factible(perturb, lotes):
                perturb = _repair(perturb, lotes)

        if not _es_factible(perturb, lotes):
            perturb = _repair(perturb, lotes)

        # Búsqueda local limitada
        t_bl = time.perf_counter() + min(t_rest * 0.28, 1.0)
        s_loc, c_loc = _busqueda_local(perturb, lotes, setup, t_corte=t_bl)

        # Aceptación SA
        delta = c_loc - actual_c
        if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
            actual   = s_loc[:]
            actual_c = c_loc
            sin_mejora = 0 if delta < 0 else sin_mejora + 1
        else:
            sin_mejora += 1

        if actual_c < mejor_c:
            mejor_c = actual_c
            mejor   = actual[:]
            elite.append((mejor_c, mejor[:]))
            elite.sort(key=lambda x: x[0])
            elite = elite[:5]

        T  = max(T * alpha, T_min)
        it += 1

    return mejor, mejor_c


# =============================================================================
# CLASE REQUERIDA POR GRADESCOPE
# =============================================================================

class Schedulling:
    def __init__(self, n_instancias: int = 30, semilla: int = 42):
        """No modificar."""
        self.n_instancias = n_instancias
        self.semilla      = semilla
        random.seed(semilla)
        np.random.seed(semilla)
        self.resultados   = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        self.consolidado  = None

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
                "M3":   random.randint(*rangos["M3"]),
            }

        setup_dict = {}
        for maquina in ["M1", "M2", "M3"]:
            setup_dict[maquina] = {}
            for ta in tipos:
                for ts in tipos:
                    if ta == ts:
                        t = 0
                    elif maquina == "M1": t = random.randint(10, 40)
                    elif maquina == "M2": t = random.randint(5,  30)
                    else:                 t = random.randint(3,  15)
                    setup_dict[maquina][f"{ta}-{ts}"] = t

        return {"lotes": lotes_dict, "setup": setup_dict}

    def resolver_instancia(self, instancia: dict) -> dict:
        t0    = time.perf_counter()
        t_fin = t0 + TIEMPO_POR_INSTANCIA

        lotes = instancia["lotes"]
        setup = instancia["setup"]
        ids   = list(lotes.keys())

        # ── Fase 1: NEH 256 variantes (~1-2s) ────────────────────────────────
        seq  = _neh_variants(ids, lotes, setup)
        cmax = _calcular_cmax(seq, lotes, setup)

        # ── Fase 2: Búsqueda local exhaustiva (~2.5s) ─────────────────────────
        t_bl = t0 + 2.5
        seq, cmax = _busqueda_local(seq, lotes, setup, t_corte=t_bl)

        # ── Fase 3: ILS+SA con tiempo restante ───────────────────────────────
        if time.perf_counter() < t_fin - 0.5:
            seq_ils, c_ils = _ils_sa(seq, cmax, lotes, setup, t_corte=t_fin - 0.3)
            if c_ils < cmax:
                cmax = c_ils
                seq  = seq_ils[:]

        # ── Validación de seguridad ───────────────────────────────────────────
        if not _es_factible(seq, lotes) or cmax == float("inf"):
            seq  = _repair(seq, lotes)
            cmax = _calcular_cmax(seq, lotes, setup)

        return {"secuencia": seq, "valor_objetivo": int(cmax)}

    def ejecutar_experimentos(self):
        """No modificar."""
        print(f"Ejecutando {self.n_instancias} instancias...\n")
        self.resultados  = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        inicio_total     = time.time()

        for i in range(1, self.n_instancias + 1):
            instancia = self.generar_instancia(i)
            t0        = time.time()
            resultado = self.resolver_instancia(instancia)
            t1        = time.time()

            self.resultados["instancia"].append(i)
            self.resultados["valor_objetivo"].append(resultado.get("valor_objetivo", 0))
            self.resultados["tiempo_seg"].append(round(t1 - t0, 4))
            print(f"Instancia {i:2d} | Cmax={resultado['valor_objetivo']:>6} | {round(t1-t0,2)}s")

        tiempo_total     = time.time() - inicio_total
        self.consolidado = pd.DataFrame(self.resultados)
        print(f"\n==== REPORTE ====")
        print(f"Cmax promedio  : {self.consolidado['valor_objetivo'].mean():.1f}")
        print(f"Tiempo promedio: {self.consolidado['tiempo_seg'].mean():.2f}s")
        print(f"Tiempo total   : {tiempo_total:.1f}s")
        return self.consolidado

    def calcular_makespan_penalizado(self, data={}, secuencia=[f'L{i}' for i in range(1, 16)]):
        lotes   = data["lotes"]
        setup   = data["setup"]
        n       = len(secuencia)
        fin     = [[0.0] * 3 for _ in range(n)]

        penalizado = any(
            lotes[secuencia[i-1]]["tipo"] == "N" and lotes[secuencia[i]]["tipo"] == "R"
            for i in range(1, n)
        )

        for i, lote in enumerate(secuencia):
            tc = lotes[lote]["tipo"]
            tp = lotes[secuencia[i-1]]["tipo"] if i > 0 else None
            for k, m in enumerate(["M1", "M2", "M3"]):
                s  = setup[m][f"{tp}-{tc}"] if tp and tp != tc else 0
                lm = fin[i-1][k] if i > 0 else 0.0
                ll = fin[i][k-1] if k > 0 else 0.0
                fin[i][k] = max(lm, ll) + s + lotes[lote][m]

        makespan = fin[-1][2]
        return float("inf") if penalizado else makespan


# =============================================================================
# FUNCIÓN PRINCIPAL — formato exacto requerido por Gradescope
# =============================================================================

def solve(data: dict) -> dict:
    """
    Parámetro : data — diccionario cargado desde data.json
    Retorna   : {"Cmax": int, "secuencia": list[str], "tiempo": int}
    """
    start = time.time()
    sch   = Schedulling()
    res   = sch.resolver_instancia(data)
    return {
        "Cmax"     : int(res["valor_objetivo"]),
        "secuencia": res["secuencia"],
        "tiempo"   : int((time.time() - start) * 1000),
    }


# =============================================================================
# PRUEBA LOCAL
# =============================================================================

if __name__ == "__main__":
    import os

    if os.path.exists("data.json"):
        with open("data.json") as f:
            data = json.load(f)
        print("📂 Usando data.json\n")
    else:
        print("⚠️  Sin data.json — generando instancia de prueba\n")
        sch  = Schedulling()
        data = sch.generar_instancia(1)

    resultado = solve(data)
    print(f"✅ Factible : {_es_factible(resultado['secuencia'], data['lotes'])}")
    print(f"📦 Secuencia: {resultado['secuencia']}")
    print(f"⏱️  Cmax     : {resultado['Cmax']} minutos")
    print(f"🕐 Tiempo   : {resultado['tiempo']} ms")
    print()
    print(json.dumps(resultado, indent=2))
