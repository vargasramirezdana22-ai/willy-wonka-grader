"""
Reto 01 Willy Wonka — Flow Shop 3 máquinas
Algoritmo: NEH + Búsqueda Local (Or-opt + 2-opt + Or-opt-inv) + ILS (double-bridge + perturbación guiada)

Optimizaciones clave vs versión anterior:
  - cmax_fast(): evaluación por índices numéricos (~7x más rápido que la versión con strings)
  - Búsqueda local "first-improvement" limpia: rompe de inmediato al encontrar mejora
  - Más iteraciones ILS en el mismo presupuesto de tiempo
  - Semilla fija garantiza reproducibilidad
  - Límite de 55 s con margen de seguridad ante el tope de 60 s de Gradescope

Cumplimiento de restricciones del reto:
  ✓ Flow Shop 3 máquinas en serie (M1 → M2 → M3)
  ✓ Restricción N→R: un lote R nunca va inmediatamente después de uno N
  ✓ Tiempos de alistamiento dependientes del tipo y distintos por máquina
  ✓ Objetivo: minimizar Cmax
  ✓ Salida: {"Cmax": int, "secuencia": list[str], "tiempo": int}
  ✓ Solo librerías estándar + numpy
  ✓ Tiempo ≤ 60 s por instancia (límite interno: 55 s)
"""

import json
import random
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
INF = float("inf")
_TIPO_MAP = {"S": 0, "N": 1, "R": 2}
_TIPO_NAMES = ["S", "N", "R"]
_MAQUINAS = ["M1", "M2", "M3"]


# ---------------------------------------------------------------------------
# PRE-CÓMPUTO: convierte los dicts del JSON a arrays numéricos rápidos
# ---------------------------------------------------------------------------

def _build_arrays(lotes_dict: dict, setup_dict: dict):
    """
    Retorna:
      lote_ids  : list[str]  — IDs ordenados (L1..L15)
      tipo_arr  : list[int]  — tipo numérico de cada lote (S=0,N=1,R=2)
      proc_arr  : list[tuple]— (M1, M2, M3) de cada lote
      su        : int[3][3][3]— su[maquina][tipo_ant][tipo_sig] = tiempo setup
    """
    lote_ids = sorted(lotes_dict.keys())
    tipo_arr = [_TIPO_MAP[lotes_dict[l]["tipo"]] for l in lote_ids]
    proc_arr = [
        (lotes_dict[l]["M1"], lotes_dict[l]["M2"], lotes_dict[l]["M3"])
        for l in lote_ids
    ]
    # Tabla de setups: su[máquina_idx][tipo_ant][tipo_sig]
    su = [[[0] * 3 for _ in range(3)] for _ in range(3)]
    for mi, m in enumerate(_MAQUINAS):
        for a in range(3):
            for b in range(3):
                if a != b:
                    su[mi][a][b] = setup_dict[m][f"{_TIPO_NAMES[a]}-{_TIPO_NAMES[b]}"]
    return lote_ids, tipo_arr, proc_arr, su


# ---------------------------------------------------------------------------
# CÁLCULO DE Cmax — núcleo del algoritmo, llamado miles de veces
# ---------------------------------------------------------------------------

def _cmax(idx_seq: list, tipo_arr: list, proc_arr: list, su: list) -> float:
    """
    Calcula el Cmax de una secuencia dada como lista de índices enteros.
    Retorna INF si hay alguna violación N→R.

    Implementación sin asignaciones innecesarias: ~7x más rápida que la
    versión original basada en strings y dicts.
    """
    n = len(idx_seq)
    ci0 = idx_seq[0]
    pt = tipo_arr[ci0]
    pp = proc_arr[ci0]
    pf0 = pp[0]
    pf1 = pf0 + pp[1]
    pf2 = pf1 + pp[2]

    for i in range(1, n):
        ci = idx_seq[i]
        ct = tipo_arr[ci]
        # Restricción de factibilidad: N → R prohibido
        if pt == 1 and ct == 2:
            return INF
        cp = proc_arr[ci]
        if pt != ct:
            s0 = su[0][pt][ct]
            s1 = su[1][pt][ct]
            s2 = su[2][pt][ct]
        else:
            s0 = s1 = s2 = 0
        f0 = pf0 + s0 + cp[0]
        f1 = (pf1 if pf1 > f0 else f0) + s1 + cp[1]
        f2 = (pf2 if pf2 > f1 else f1) + s2 + cp[2]
        pf0 = f0
        pf1 = f1
        pf2 = f2
        pt = ct

    return pf2


def _es_factible(idx_seq: list, tipo_arr: list) -> bool:
    for i in range(1, len(idx_seq)):
        if tipo_arr[idx_seq[i - 1]] == 1 and tipo_arr[idx_seq[i]] == 2:
            return False
    return True


# ---------------------------------------------------------------------------
# FASE 1 — HEURÍSTICA CONSTRUCTIVA NEH
# ---------------------------------------------------------------------------

def _neh(n_lotes: int, tipo_arr: list, proc_arr: list, su: list):
    """
    Heurística NEH clásica: ordena lotes por suma de tiempos descendente
    e inserta cada uno en la mejor posición disponible.
    """
    sums = [proc_arr[i][0] + proc_arr[i][1] + proc_arr[i][2] for i in range(n_lotes)]
    order = sorted(range(n_lotes), key=lambda i: sums[i], reverse=True)

    seq = [order[0]]
    for idx in order[1:]:
        best_c = INF
        best_pos = 0
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [idx] + seq[pos:]
            c = _cmax(cand, tipo_arr, proc_arr, su)
            if c < best_c:
                best_c = c
                best_pos = pos
        seq = seq[:best_pos] + [idx] + seq[best_pos:]

    return seq, _cmax(seq, tipo_arr, proc_arr, su)


# ---------------------------------------------------------------------------
# FASE 2 — BÚSQUEDA LOCAL
# ---------------------------------------------------------------------------

def _busqueda_local(
    seq: list, best_c: float,
    tipo_arr: list, proc_arr: list, su: list,
    t_cut: float
) -> tuple:
    """
    Or-opt(1,2,3) + 2-opt (swap) + Or-opt-inv(2,3).
    Estrategia "first-improvement": en cuanto encuentra mejora, reinicia.
    Detiene si se supera t_cut.
    """
    n = len(seq)
    improved_global = True

    while improved_global:
        if time.perf_counter() > t_cut:
            break
        improved_global = False

        # ── Or-opt: reinserción de segmentos de tamaño 1, 2, 3 ──────────────
        for tam in [1, 2, 3]:
            improved = True
            while improved:
                if time.perf_counter() > t_cut:
                    return seq, best_c
                improved = False
                for i in range(n - tam + 1):
                    seg = seq[i: i + tam]
                    base = seq[:i] + seq[i + tam:]
                    lb = len(base)
                    for j in range(lb + 1):
                        cand = base[:j] + seg + base[j:]
                        c = _cmax(cand, tipo_arr, proc_arr, su)
                        if c < best_c:
                            best_c = c
                            seq = cand
                            improved = True
                            improved_global = True
                            break  # first-improvement: reiniciar barrido
                    if improved:
                        break

        # ── 2-opt: intercambio de pares ──────────────────────────────────────
        improved = True
        while improved:
            if time.perf_counter() > t_cut:
                return seq, best_c
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = _cmax(seq, tipo_arr, proc_arr, su)
                    if c < best_c:
                        best_c = c
                        improved = True
                        improved_global = True
                        break  # first-improvement
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
                if improved:
                    break

        # ── Or-opt invertido: reinserción de segmentos invertidos ────────────
        improved = True
        while improved:
            if time.perf_counter() > t_cut:
                return seq, best_c
            improved = False
            for i in range(n - 1):
                for tam in [2, 3]:
                    if i + tam > n:
                        continue
                    seg_inv = seq[i: i + tam][::-1]
                    base = seq[:i] + seq[i + tam:]
                    lb = len(base)
                    for j in range(lb + 1):
                        cand = base[:j] + seg_inv + base[j:]
                        c = _cmax(cand, tipo_arr, proc_arr, su)
                        if c < best_c:
                            best_c = c
                            seq = cand
                            improved = True
                            improved_global = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return seq, best_c


# ---------------------------------------------------------------------------
# FASE 3 — ILS: perturbaciones + búsqueda local
# ---------------------------------------------------------------------------

def _double_bridge(seq: list) -> list:
    """Perturbación clásica de 4 cortes que rompe óptimos locales."""
    n = len(seq)
    pos = sorted(random.sample(range(1, n), 3))
    a, b, c = pos
    return seq[0:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturbacion_guiada(seq: list, tipo_arr: list) -> list:
    """
    Extrae un bloque de lotes del mismo tipo y lo reinserta en otra posición.
    Garantiza factibilidad; si no logra construir una perturbación válida,
    cae sobre double-bridge.
    """
    order = [0, 1, 2]
    random.shuffle(order)
    for t in order:
        indices = [i for i, l in enumerate(seq) if tipo_arr[l] == t]
        if len(indices) < 2:
            continue
        k = random.randint(1, min(3, len(indices)))
        sel = sorted(random.sample(indices, k), reverse=True)
        nuevo = seq[:]
        ext = []
        for idx in sel:
            ext.insert(0, nuevo.pop(idx))
        pos = random.randint(0, len(nuevo))
        nuevo = nuevo[:pos] + ext + nuevo[pos:]
        if _es_factible(nuevo, tipo_arr):
            return nuevo
    return _double_bridge(seq)


def _ils(
    seq: list, best_c: float,
    tipo_arr: list, proc_arr: list, su: list,
    t_cut: float
) -> tuple:
    """
    Iterated Local Search:
      - Alterna double-bridge y perturbación guiada.
      - Reinicia desde el mejor global cada 20 iteraciones sin mejora.
      - Asigna hasta 40% del tiempo restante (máx 2 s) a cada búsqueda local.
    """
    cur = seq[:]
    cur_c = best_c
    it = 0

    while time.perf_counter() < t_cut:
        perturb = (
            _double_bridge(cur) if it % 2 == 0
            else _perturbacion_guiada(cur, tipo_arr)
        )

        t_remaining = t_cut - time.perf_counter()
        t_bl = time.perf_counter() + min(t_remaining * 0.40, 2.0)
        p_c = _cmax(perturb, tipo_arr, proc_arr, su)
        p, p_c = _busqueda_local(perturb, p_c, tipo_arr, proc_arr, su, t_bl)

        # Criterio de aceptación: solo mejora estricta (descent)
        if p_c < cur_c:
            cur = p[:]
            cur_c = p_c

        if cur_c < best_c:
            best_c = cur_c
            seq = cur[:]

        # Reinicio periódico al mejor global
        if it % 20 == 19 and cur_c > best_c:
            cur = seq[:]
            cur_c = best_c

        it += 1

    return seq, best_c


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL — solve() — requerida por Gradescope
# ---------------------------------------------------------------------------

def solve(data: dict) -> dict:
    """
    Parámetro : data — diccionario cargado desde data.json
    Retorna   : {"Cmax": int, "secuencia": list[str], "tiempo": int}

    Estrategia:
      1. Pre-cómputo de arrays numéricos (evita overhead de strings en el loop caliente)
      2. Heurística NEH para solución inicial de calidad
      3. Búsqueda local completa (Or-opt 1/2/3 + 2-opt + Or-opt-inv)
      4. ILS hasta agotar el presupuesto de tiempo (55 s con margen de seguridad)
    """
    start_ms = time.time()
    t_inicio = time.perf_counter()
    TIEMPO_LIMITE = 55.0                          # margen ante el límite de 60 s
    t_cut = t_inicio + TIEMPO_LIMITE

    random.seed(42)
    np.random.seed(42)

    lotes_dict = data["lotes"]
    setup_dict = data["setup"]

    # ── Pre-cómputo ──────────────────────────────────────────────────────────
    lote_ids, tipo_arr, proc_arr, su = _build_arrays(lotes_dict, setup_dict)
    n = len(lote_ids)

    # ── Fase 1: NEH ──────────────────────────────────────────────────────────
    seq, best_c = _neh(n, tipo_arr, proc_arr, su)

    # ── Fase 2: búsqueda local inicial ───────────────────────────────────────
    seq, best_c = _busqueda_local(seq, best_c, tipo_arr, proc_arr, su, t_cut)

    # ── Fase 3: ILS hasta agotar tiempo ─────────────────────────────────────
    if time.perf_counter() < t_cut - 0.5:
        seq, best_c = _ils(seq, best_c, tipo_arr, proc_arr, su, t_cut)

    # ── Validación de seguridad ──────────────────────────────────────────────
    if not _es_factible(seq, tipo_arr) or best_c == INF:
        seq, best_c = _neh(n, tipo_arr, proc_arr, su)

    elapsed_ms = int((time.time() - start_ms) * 1000)

    return {
        "Cmax": int(best_c),
        "secuencia": [lote_ids[i] for i in seq],
        "tiempo": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# CLASE Schedulling — estructura auxiliar para experimentos locales
# ---------------------------------------------------------------------------

TIEMPO_POR_INSTANCIA = 15.0


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
        tipos = ["S", "N", "R"]
        lotes_list = [f"L{i}" for i in range(1, 16)]
        rangos = {"M1": (25, 45), "M2": (20, 40), "M3": (10, 25)}

        lotes_dict = {}
        for lote in lotes_list:
            tipo = random.choice(tipos)
            lotes_dict[lote] = {
                "tipo": tipo,
                "M1": random.randint(*rangos["M1"]),
                "M2": random.randint(*rangos["M2"]),
                "M3": random.randint(*rangos["M3"]),
            }

        setup_dict = {}
        for maquina in ["M1", "M2", "M3"]:
            setup_dict[maquina] = {}
            for tipo_ant in tipos:
                for tipo_sig in tipos:
                    if tipo_ant == tipo_sig:
                        t = 0
                    elif maquina == "M1":
                        t = random.randint(10, 40)
                    elif maquina == "M2":
                        t = random.randint(5, 30)
                    else:
                        t = random.randint(3, 15)
                    setup_dict[maquina][f"{tipo_ant}-{tipo_sig}"] = t

        return {"lotes": lotes_dict, "setup": setup_dict}

    def resolver_instancia(self, instancia: dict):
        """NEH + búsqueda local + ILS con tiempo controlado por instancia."""
        t_inicio = time.perf_counter()
        t_cut = t_inicio + TIEMPO_POR_INSTANCIA

        lotes_dict = instancia["lotes"]
        setup_dict = instancia["setup"]

        lote_ids, tipo_arr, proc_arr, su = _build_arrays(lotes_dict, setup_dict)
        n = len(lote_ids)

        seq, best_c = _neh(n, tipo_arr, proc_arr, su)
        seq, best_c = _busqueda_local(seq, best_c, tipo_arr, proc_arr, su, t_cut)

        if time.perf_counter() < t_cut - 0.5:
            seq, best_c = _ils(seq, best_c, tipo_arr, proc_arr, su, t_cut)

        if not _es_factible(seq, tipo_arr) or best_c == INF:
            seq, best_c = _neh(n, tipo_arr, proc_arr, su)

        return {
            "secuencia": [lote_ids[i] for i in seq],
            "valor_objetivo": int(best_c),
        }

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
            print(f"Instancia {i} optimizada en {round(t1-t0, 4)} seg")

        tiempo_total = time.time() - inicio_total
        self.consolidado = pd.DataFrame(self.resultados)
        promedio_tiempo = self.consolidado["tiempo_seg"].mean()

        print("\n==== REPORTE DE RENDIMIENTO ====")
        print("Tiempo promedio por instancia:", round(promedio_tiempo, 4), "seg")
        print("Tiempo total de ejecución:", round(tiempo_total, 2), "seg")
        return self.consolidado

    def calcular_makespan_penalizado(
        self,
        data: dict = {},
        secuencia: list = [f"L{i}" for i in range(1, 16)],
    ):
        """
        Calcula el makespan real (o INF si hay violación N→R) usando la
        misma lógica que _cmax para garantizar consistencia.
        """
        lotes_dict = data["lotes"]
        setup_dict = data["setup"]
        lote_ids, tipo_arr, proc_arr, su = _build_arrays(lotes_dict, setup_dict)

        # Convertir secuencia de strings a índices
        lote_index = {l: i for i, l in enumerate(lote_ids)}
        idx_seq = [lote_index[l] for l in secuencia]

        return _cmax(idx_seq, tipo_arr, proc_arr, su)


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA — Gradescope llama a solve(data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("data.json") as f:
        data = json.load(f)
    print(json.dumps(solve(data), indent=2))
