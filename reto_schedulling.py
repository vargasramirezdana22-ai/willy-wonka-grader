"""
Reto 01 — Willy Wonka | Flow Shop 3 máquinas con setups y restricción N→R
Algoritmo: Multi-start NEH + Búsqueda Local (Or-opt + 2-opt + Or-opt-inv) + ILS con elite pool
CORRECCIONES aplicadas:
  - solve() con formato {"Cmax":..., "secuencia":..., "tiempo":...} (ms)
  - Setup tipo igual = 0 siempre correcto
  - Búsqueda local con first-improve (break al encontrar mejora) → más rápida
  - t_corte aplicado en TODAS las fases incluyendo BL inicial
  - calcular_makespan_penalizado con prev_tipo rastreado correctamente
  - Multi-start NEH con semillas de tipo-bloque + ruido aleatorio
  - ILS con elite pool de 5 soluciones y 3 tipos de perturbación
"""

import random
import numpy as np
import time
import pandas as pd
import json

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
TIEMPO_POR_INSTANCIA = 15.0   # 30 × 15s = 450s < 600s límite Gradescope
FRACCION_MULTISTART  = 0.25   # 25% → ~3.75s para multi-start NEH
FRACCION_BL          = 0.10   # 10% → ~1.5s para búsqueda local inicial
                               # 65% → ~9.75s para ILS


# =============================================================================
# NÚCLEO: simulador Flow Shop
# =============================================================================

def _calcular_cmax(secuencia, lotes, setup):
    """Cmax correcto con setup dependiente del tipo. Retorna inf si viola N→R."""
    for i in range(1, len(secuencia)):
        if (lotes[secuencia[i-1]]["tipo"] == "N" and
                lotes[secuencia[i]]["tipo"] == "R"):
            return float("inf")
    fin = [[0.0] * 3 for _ in range(len(secuencia))]
    for i, lote in enumerate(secuencia):
        tipo_act = lotes[lote]["tipo"]
        for k, m in enumerate(["M1", "M2", "M3"]):
            # Setup = 0 si mismo tipo, valor tabla si diferente
            t_setup = 0
            if i > 0:
                tipo_ant = lotes[secuencia[i-1]]["tipo"]
                if tipo_ant != tipo_act:          # ← setup tipo igual = 0 correcto
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
# FASE 1 — Multi-start NEH
# =============================================================================

def _neh_una_vez(orden, lotes, setup):
    seq = [orden[0]]
    for lote in orden[1:]:
        bc = float("inf"); bp = 0
        for p in range(len(seq) + 1):
            c = _calcular_cmax(seq[:p] + [lote] + seq[p:], lotes, setup)
            if c < bc:
                bc = c; bp = p
        seq = seq[:bp] + [lote] + seq[bp:]
    return seq


def _multi_start_neh(ids, lotes, setup, t_corte):
    suma = {l: lotes[l]["M1"] + lotes[l]["M2"] + lotes[l]["M3"] for l in ids}
    best_c = float("inf")
    best_seq = None

    # Semillas deterministas: 6 ordenamientos por bloque de tipo
    for orden_tipos in [["S","N","R"], ["S","R","N"], ["R","S","N"],
                        ["N","S","R"], ["R","N","S"], ["N","R","S"]]:
        s = []
        for t in orden_tipos:
            grupo = sorted([l for l in ids if lotes[l]["tipo"] == t],
                           key=lambda l: suma[l], reverse=True)
            s.extend(grupo)
        seq = _neh_una_vez(s, lotes, setup)
        c = _calcular_cmax(seq, lotes, setup)
        if c < best_c:
            best_c = c; best_seq = seq[:]

    # Semillas aleatorias con ruido creciente hasta agotar el tiempo
    seed = 0
    while time.perf_counter() < t_corte:
        rng = random.Random(seed)
        noise = 0 if seed == 0 else min(seed * 2, 60)
        orden = sorted(ids, key=lambda l: suma[l] + rng.uniform(-noise, noise), reverse=True)
        seq = _neh_una_vez(orden, lotes, setup)
        c = _calcular_cmax(seq, lotes, setup)
        if c < best_c:
            best_c = c; best_seq = seq[:]
        seed += 1

    return best_seq, best_c


# =============================================================================
# FASE 2 — Búsqueda local (first-improve para mayor velocidad)
# =============================================================================

def _busqueda_local(seq, lotes, setup, t_corte=None):
    """Or-opt(1,2,3) + 2-opt + Or-opt-inv + 3-opt reversal.
    Usa first-improve: al encontrar mejora reinicia el bucle (más rápido que best-improve)."""
    mejor = seq[:]
    mc = _calcular_cmax(mejor, lotes, setup)
    n = len(mejor)
    mg = True

    while mg:
        if t_corte and time.perf_counter() > t_corte:
            break
        mg = False

        # ── Or-opt: mover segmento de tam=1,2,3 ──────────────────────────────
        for tam in [1, 2, 3]:
            mejorado = True
            while mejorado:
                if t_corte and time.perf_counter() > t_corte:
                    return mejor, mc
                mejorado = False
                for i in range(n - tam + 1):
                    if mejorado: break          # first-improve ← CORRECCIÓN
                    seg = mejor[i:i+tam]
                    base = mejor[:i] + mejor[i+tam:]
                    for j in range(len(base) + 1):
                        c = _calcular_cmax(base[:j] + seg + base[j:], lotes, setup)
                        if c < mc:
                            mc = c; mejor = base[:j] + seg + base[j:]
                            mejorado = True; mg = True
                            break               # first-improve

        # ── 2-opt: swap de pares ──────────────────────────────────────────────
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejorado = False
            for i in range(n):
                if mejorado: break              # first-improve
                for j in range(i + 1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = _calcular_cmax(cand, lotes, setup)
                    if c < mc:
                        mc = c; mejor = cand[:]
                        mejorado = True; mg = True
                        break                   # first-improve

        # ── Or-opt inverso: mover segmento revertido ──────────────────────────
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejorado = False
            for i in range(n - 1):
                if mejorado: break              # first-improve
                for tam in [2, 3]:
                    if i + tam > n: continue
                    seg_inv = mejor[i:i+tam][::-1]
                    base = mejor[:i] + mejor[i+tam:]
                    for j in range(len(base) + 1):
                        c = _calcular_cmax(base[:j] + seg_inv + base[j:], lotes, setup)
                        if c < mc:
                            mc = c; mejor = base[:j] + seg_inv + base[j:]
                            mejorado = True; mg = True
                            break               # first-improve

        # ── 3-opt: reversal de subsecuencias ─────────────────────────────────
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejorado = False
            for i in range(n - 2):
                if mejorado: break              # first-improve
                for j in range(i + 2, n):
                    cand = mejor[:i] + mejor[i:j+1][::-1] + mejor[j+1:]
                    c = _calcular_cmax(cand, lotes, setup)
                    if c < mc:
                        mc = c; mejor = cand[:]
                        mejorado = True; mg = True
                        break                   # first-improve

    return mejor, mc


# =============================================================================
# FASE 3 — ILS con elite pool y 3 tipos de perturbación
# =============================================================================

def _double_bridge(seq):
    n = len(seq)
    a, b, c = sorted(random.sample(range(1, n), 3))
    return seq[:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturbacion_guiada(seq, lotes):
    tipos = ["S", "N", "R"]
    random.shuffle(tipos)
    for tipo in tipos:
        idx = [i for i, l in enumerate(seq) if lotes[l]["tipo"] == tipo]
        if len(idx) < 2: continue
        k = random.randint(1, min(3, len(idx)))
        sel = sorted(random.sample(idx, k), reverse=True)
        nuevo = seq[:]
        ext = []
        for i in sel: ext.insert(0, nuevo.pop(i))
        p = random.randint(0, len(nuevo))
        nuevo = nuevo[:p] + ext + nuevo[p:]
        if _es_factible(nuevo, lotes): return nuevo
    return _double_bridge(seq)


def _perturbacion_bloque(seq, lotes):
    tipos = ["S", "N", "R"]
    random.shuffle(tipos)
    for tipo in tipos:
        idx = [i for i, l in enumerate(seq) if lotes[l]["tipo"] == tipo]
        if len(idx) < 2: continue
        inicio = random.choice(idx)
        bloque = [seq[inicio]]
        nuevo = seq[:]
        nuevo.pop(inicio)
        p = random.randint(0, len(nuevo))
        nuevo = nuevo[:p] + bloque + nuevo[p:]
        if _es_factible(nuevo, lotes): return nuevo
    return _double_bridge(seq)


def _ils(seq_ini, c_ini, lotes, setup, t_corte):
    mejor = seq_ini[:]
    mc = c_ini
    actual = seq_ini[:]
    ac = c_ini
    elite = [(mc, mejor[:])]   # pool de 5 mejores soluciones
    sin_mejora = 0
    it = 0

    while time.perf_counter() < t_corte:
        t_restante = t_corte - time.perf_counter()
        if t_restante < 0.3: break

        # Reinicio desde elite si lleva 30 iteraciones sin mejorar
        if sin_mejora > 30:
            _, base = random.choice(elite)
            actual = base[:]
            ac = _calcular_cmax(actual, lotes, setup)
            sin_mejora = 0
            perturb = _double_bridge(actual)
        elif it % 3 == 0:
            perturb = _double_bridge(actual)
        elif it % 3 == 1:
            perturb = _perturbacion_guiada(actual, lotes)
        else:
            perturb = _perturbacion_bloque(actual, lotes)

        t_bl = time.perf_counter() + min(t_restante * 0.35, 1.5)
        sl, cl = _busqueda_local(perturb, lotes, setup, t_corte=t_bl)

        if cl < ac:
            actual = sl[:]; ac = cl; sin_mejora = 0
        else:
            sin_mejora += 1

        if ac < mc:
            mc = ac; mejor = actual[:]
            elite.append((mc, mejor[:]))
            elite.sort(key=lambda x: x[0])
            elite = elite[:5]

        it += 1

    return mejor, mc


# =============================================================================
# solve() — función principal exigida por Gradescope (reto1.py)
# Retorna exactamente: {"Cmax": int, "secuencia": list[str], "tiempo": int}
# =============================================================================

def solve(data: dict) -> dict:
    t_inicio = time.perf_counter()
    T_LIMITE = 58.0
    t_corte = t_inicio + T_LIMITE

    lotes = data["lotes"]
    setup = data["setup"]
    ids = list(lotes.keys())

    t_ms = t_inicio + T_LIMITE * 0.25
    seq, cmax = _multi_start_neh(ids, lotes, setup, t_corte=t_ms)

    t_bl = t_ms + T_LIMITE * 0.10
    seq, cmax = _busqueda_local(seq, lotes, setup, t_corte=t_bl)

    if time.perf_counter() < t_corte - 0.5:
        seq_ils, c_ils = _ils(seq, cmax, lotes, setup, t_corte=t_corte)
        if c_ils < cmax:
            cmax = c_ils; seq = seq_ils[:]

    if not _es_factible(seq, lotes) or cmax == float("inf"):
        orden = sorted(ids, key=lambda l: lotes[l]["M1"]+lotes[l]["M2"]+lotes[l]["M3"], reverse=True)
        seq = _neh_una_vez(orden, lotes, setup)
        cmax = _calcular_cmax(seq, lotes, setup)

    elapsed_ms = int((time.perf_counter() - t_inicio) * 1000)
    return {
        "Cmax":      int(cmax),
        "secuencia": seq,
        "tiempo":    elapsed_ms          # ← milisegundos, formato exigido
    }


# =============================================================================
# CLASE SCHEDULLING — estructura requerida por Gradescope (reto_schedulling.py)
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
        """Igual al grader del profesor — no modificar la lógica."""
        tipos = ["S", "N", "R"]
        lotes = [f"L{i}" for i in range(1, 16)]
        rangos = {"M1": (25,45), "M2": (20,40), "M3": (10,25)}

        lotes_dict = {}
        for lote in lotes:
            lotes_dict[lote] = {
                "tipo": random.choice(tipos),
                "M1":   random.randint(*rangos["M1"]),
                "M2":   random.randint(*rangos["M2"]),
                "M3":   random.randint(*rangos["M3"])
            }

        setup_dict = {}
        for maquina in ["M1", "M2", "M3"]:
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
        """
        El grader llama este método y solo usa result["secuencia"].
        Usa TIEMPO_POR_INSTANCIA = 15s → 30 × 15 = 450s < 600s límite.
        """
        t0 = time.perf_counter()
        t_fin = t0 + TIEMPO_POR_INSTANCIA

        lotes = instancia["lotes"]
        setup = instancia["setup"]
        ids   = list(lotes.keys())

        # Fase 1: Multi-start NEH
        t_ms = t0 + TIEMPO_POR_INSTANCIA * FRACCION_MULTISTART
        seq, cmax = _multi_start_neh(ids, lotes, setup, t_corte=t_ms)

        # Fase 2: Búsqueda local con t_corte ← CORRECCIÓN (antes no tenía)
        t_bl = t_ms + TIEMPO_POR_INSTANCIA * FRACCION_BL
        seq, cmax = _busqueda_local(seq, lotes, setup, t_corte=t_bl)

        # Fase 3: ILS con elite pool
        if time.perf_counter() < t_fin - 0.5:
            seq_ils, c_ils = _ils(seq, cmax, lotes, setup, t_corte=t_fin)
            if c_ils < cmax:
                cmax = c_ils; seq = seq_ils[:]

        # Validación de seguridad
        if not _es_factible(seq, lotes) or cmax == float("inf"):
            orden = sorted(ids, key=lambda l: lotes[l]["M1"]+lotes[l]["M2"]+lotes[l]["M3"], reverse=True)
            seq = _neh_una_vez(orden, lotes, setup)
            cmax = _calcular_cmax(seq, lotes, setup)

        return {
            "secuencia":      seq,
            "valor_objetivo": int(cmax)
        }

    def ejecutar_experimentos(self):
        """No modificar — estructura requerida por Gradescope."""
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
            print(f"Instancia {i}: Cmax={resultado.get('valor_objetivo','?')} en {round(t1-t0,2)}s")

        tiempo_total = time.time() - inicio_total
        self.consolidado = pd.DataFrame(self.resultados)
        print("\n==== REPORTE DE RENDIMIENTO ====")
        print("Tiempo promedio por instancia:", round(self.consolidado["tiempo_seg"].mean(), 4), "seg")
        print("Cmax promedio:               ", round(self.consolidado["valor_objetivo"].mean(), 1))
        print("Tiempo total de ejecución:   ", round(tiempo_total, 2), "seg")
        return self.consolidado

    def calcular_makespan_penalizado(self, data, secuencia):
        """
        Réplica exacta del grader del profesor.
        prev_tipo se rastrea con tiempos[m][-1]["tipo"] ← CORRECCIÓN del bug.
        """
        lotes = data["lotes"]
        setup = data["setup"]
        maquinas = ["M1", "M2", "M3"]

        # Verificar N→R
        for i in range(1, len(secuencia)):
            if lotes[secuencia[i-1]]["tipo"] == "N" and lotes[secuencia[i]]["tipo"] == "R":
                return float("inf")

        tiempos    = {m: [] for m in maquinas}
        fin_maquina = {m: 0  for m in maquinas}
        fin_lote    = {l: 0  for l in secuencia}

        for lote in secuencia:
            tipo_actual = lotes[lote]["tipo"]
            for m_idx, m in enumerate(maquinas):
                # prev_tipo desde el último elemento procesado en esta máquina ← correcto
                prev_tipo = tiempos[m][-1]["tipo"] if tiempos[m] else None
                t_setup = 0
                if prev_tipo is not None and prev_tipo != tipo_actual:
                    t_setup = setup[m][f"{prev_tipo}-{tipo_actual}"]
                if m_idx == 0:
                    inicio = fin_maquina[m] + t_setup
                else:
                    inicio = max(fin_maquina[m], fin_lote[lote]) + t_setup
                fin = inicio + lotes[lote][m]
                tiempos[m].append({"lote": lote, "tipo": tipo_actual})
                fin_maquina[m] = fin
                fin_lote[lote]  = fin

        return max(fin_maquina.values())


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    if os.path.exists("data.json"):
        with open("data.json", encoding="utf-8") as f:
            data = json.load(f)
        # Modo reto1.py: usa solve()
        resultado = solve(data)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    else:
        print("Sin data.json — ejecutando experimentos con instancias aleatorias\n")
        sch = Schedulling(n_instancias=3)
        df = sch.ejecutar_experimentos()
        print("\nConsolidado:")
        print(df)

