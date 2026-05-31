"""
Reto 01 Willy Wonka — Flow Shop 3 máquinas
Algoritmo: Multi-start NEH + Búsqueda Local (Or-opt 1/2/3 + Swap) + ILS (double-bridge + perturbación guiada)

Optimizaciones clave vs versión anterior:
- _cmax: cálculo 2x más rápido con arrays planos precomputados (sin dict lookup por llamada)
- Or-opt con break temprano al encontrar mejora (evita recorrer resto innecesariamente)
- Multi-start NEH con 4 ordenes distintos para mejor solución inicial
- ILS con presupuesto de tiempo adaptativo por iteración
- Tiempo límite conservador de 55 s (margen ante límite de 60 s de Gradescope)
"""

import json
import random
import time

# =============================================================================
# PREPROCESAMIENTO — se ejecuta una sola vez por instancia
# =============================================================================

def _preprocesar(lotes: dict, setup: dict):
    """
    Convierte los dicts de entrada en estructuras planas de acceso O(1):
      - tipo_int_arr[i]  : tipo del lote i como entero (S=0, N=1, R=2)
      - proc_flat[i]     : (M1, M2, M3) del lote i
      - setup_3d[t_ant][t_act][maquina]: tiempo de setup como entero
      - idx_map          : nombre_lote -> índice entero
      - names            : índice -> nombre_lote
    """
    TIPO_INT = {"S": 0, "N": 1, "R": 2}
    TIPOS = ["S", "N", "R"]
    MAQUINAS = ["M1", "M2", "M3"]

    names = list(lotes.keys())
    idx_map = {l: i for i, l in enumerate(names)}
    n = len(names)

    tipo_int_arr = [TIPO_INT[lotes[l]["tipo"]] for l in names]
    proc_flat = [(lotes[l]["M1"], lotes[l]["M2"], lotes[l]["M3"]) for l in names]

    setup_3d = [[[0] * 3 for _ in range(3)] for _ in range(3)]
    for ti, t1 in enumerate(TIPOS):
        for tj, t2 in enumerate(TIPOS):
            for km, m in enumerate(MAQUINAS):
                setup_3d[ti][tj][km] = setup[m][f"{t1}-{t2}"]

    return names, idx_map, n, tipo_int_arr, proc_flat, setup_3d


# =============================================================================
# CÁLCULO DE Cmax — función crítica, se llama millones de veces
# =============================================================================

def _cmax(seq, tipo_int_arr, proc_flat, setup_3d):
    """
    Calcula el makespan de la secuencia `seq` (lista de índices enteros).
    Retorna float('inf') si existe una transición N→R (violación de restricción).

    Usa arrays planos 1-D en lugar de matriz 2-D para minimizar accesos a memoria
    y eliminar dict lookups — aprox. 2x más rápido que la versión con dicts.
    """
    n = len(seq)

    # Verificar restricción N→R (N=1, R=2) en una sola pasada
    prev_tipo = tipo_int_arr[seq[0]]
    for i in range(1, n):
        cur_tipo = tipo_int_arr[seq[i]]
        if prev_tipo == 1 and cur_tipo == 2:   # N seguido de R
            return float("inf")
        prev_tipo = cur_tipo

    # Arrays 1-D de tiempos de fin: fin0=M1, fin1=M2, fin2=M3
    fin0 = [0.0] * n
    fin1 = [0.0] * n
    fin2 = [0.0] * n

    # Primer lote (sin predecesores, sin setup)
    j0 = seq[0]
    p0, p1, p2 = proc_flat[j0]
    fin0[0] = p0
    fin1[0] = p0 + p1
    fin2[0] = p0 + p1 + p2

    for i in range(1, n):
        ji = seq[i]
        jp = seq[i - 1]
        tp = tipo_int_arr[jp]
        tc = tipo_int_arr[ji]
        s = setup_3d[tp][tc]    # lista [s_M1, s_M2, s_M3]
        p = proc_flat[ji]       # tupla (pM1, pM2, pM3)

        # M1: predecesora = lote anterior en M1; no hay máquina anterior
        fin0[i] = fin0[i - 1] + s[0] + p[0]

        # M2: max(lote anterior en M2, este lote en M1) + setup + proceso
        f1_prev = fin1[i - 1]
        f0_cur  = fin0[i]
        fin1[i] = (f1_prev if f1_prev > f0_cur else f0_cur) + s[1] + p[1]

        # M3: max(lote anterior en M3, este lote en M2) + setup + proceso
        f2_prev = fin2[i - 1]
        f1_cur  = fin1[i]
        fin2[i] = (f2_prev if f2_prev > f1_cur else f1_cur) + s[2] + p[2]

    return fin2[n - 1]


def _es_factible(seq, tipo_int_arr):
    """Verifica restricción N→R sin calcular tiempos."""
    prev = tipo_int_arr[seq[0]]
    for i in range(1, len(seq)):
        cur = tipo_int_arr[seq[i]]
        if prev == 1 and cur == 2:
            return False
        prev = cur
    return True


# =============================================================================
# FASE 1: NEH multi-start
# =============================================================================

def _neh_orden(order, tipo_int_arr, proc_flat, setup_3d):
    """NEH con un orden de inserción dado (lista de índices enteros)."""
    seq = [order[0]]
    for lote_idx in order[1:]:
        best_c = float("inf")
        best_pos = 0
        n_cur = len(seq)
        for pos in range(n_cur + 1):
            candidate = seq[:pos] + [lote_idx] + seq[pos:]
            c = _cmax(candidate, tipo_int_arr, proc_flat, setup_3d)
            if c < best_c:
                best_c = c
                best_pos = pos
        seq = seq[:best_pos] + [lote_idx] + seq[best_pos:]
    return seq, _cmax(seq, tipo_int_arr, proc_flat, setup_3d)


def _neh_multistart(n, proc_flat, tipo_int_arr, setup_3d):
    """
    Ejecuta NEH con 4 órdenes distintos y retorna la mejor solución.
    Órdenes: suma total desc, M1 desc, M3 asc, aleatorio.
    """
    all_idx = list(range(n))

    ordenes = [
        # 1. Suma total procesamiento descendente (NEH clásico)
        sorted(all_idx, key=lambda i: -(proc_flat[i][0] + proc_flat[i][1] + proc_flat[i][2])),
        # 2. Tiempo M1 descendente (primera máquina bottleneck)
        sorted(all_idx, key=lambda i: -proc_flat[i][0]),
        # 3. Tiempo M3 ascendente (última máquina: los más cortos primero)
        sorted(all_idx, key=lambda i: proc_flat[i][2]),
        # 4. M1+M2 descendente, M3 ascendente (balance máquinas)
        sorted(all_idx, key=lambda i: -(proc_flat[i][0] + proc_flat[i][1]) + proc_flat[i][2]),
    ]

    mejor_seq, mejor_c = None, float("inf")
    for orden in ordenes:
        seq, c = _neh_orden(orden, tipo_int_arr, proc_flat, setup_3d)
        if c < mejor_c:
            mejor_c = c
            mejor_seq = seq[:]

    return mejor_seq, mejor_c


# =============================================================================
# FASE 2: Búsqueda local
# =============================================================================

def _busqueda_local(seq, tipo_int_arr, proc_flat, setup_3d, t_corte=None):
    """
    Or-opt(1,2,3) + Swap (2-opt) hasta convergencia o tiempo agotado.
    Break temprano al encontrar primera mejora en Or-opt para mayor velocidad.
    """
    mejor = seq[:]
    mejor_c = _cmax(mejor, tipo_int_arr, proc_flat, setup_3d)
    n = len(mejor)
    mejorado_global = True

    while mejorado_global:
        if t_corte and time.perf_counter() > t_corte:
            break
        mejorado_global = False

        # Or-opt con segmentos de tamaño 1, 2 y 3
        for tam in (1, 2, 3):
            mejorado = True
            while mejorado:
                if t_corte and time.perf_counter() > t_corte:
                    return mejor, mejor_c
                mejorado = False
                for i in range(n - tam + 1):
                    seg = mejor[i:i + tam]
                    base = mejor[:i] + mejor[i + tam:]
                    base_len = len(base)
                    for j in range(base_len + 1):
                        if j == i:
                            continue  # misma posición, no es movimiento
                        c = _cmax(base[:j] + seg + base[j:], tipo_int_arr, proc_flat, setup_3d)
                        if c < mejor_c - 0.001:
                            mejor_c = c
                            mejor = base[:j] + seg + base[j:]
                            mejorado = True
                            mejorado_global = True
                            break   # break temprano: reiniciar con nueva solución
                    if mejorado:
                        break

        # Swap (equivalente a 2-opt para permutaciones)
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mejor_c
            mejorado = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = _cmax(cand, tipo_int_arr, proc_flat, setup_3d)
                    if c < mejor_c - 0.001:
                        mejor_c = c
                        mejor = cand
                        mejorado = True
                        mejorado_global = True
                        break
                if mejorado:
                    break

        # Or-opt inverso: mover segmentos invertidos (diversificación)
        for tam in (2, 3):
            mejorado_inv = True
            while mejorado_inv:
                if t_corte and time.perf_counter() > t_corte:
                    return mejor, mejor_c
                mejorado_inv = False
                for i in range(n - tam + 1):
                    seg_inv = mejor[i:i + tam][::-1]
                    base = mejor[:i] + mejor[i + tam:]
                    for j in range(len(base) + 1):
                        c = _cmax(base[:j] + seg_inv + base[j:], tipo_int_arr, proc_flat, setup_3d)
                        if c < mejor_c - 0.001:
                            mejor_c = c
                            mejor = base[:j] + seg_inv + base[j:]
                            mejorado_inv = True
                            mejorado_global = True
                            break
                    if mejorado_inv:
                        break

    return mejor, mejor_c


# =============================================================================
# FASE 3: ILS — Iterated Local Search
# =============================================================================

def _double_bridge(seq):
    """Perturbación clásica 4-opt para ILS: rompe y recombina en 4 segmentos."""
    n = len(seq)
    if n < 4:
        return seq[:]
    pos = sorted(random.sample(range(1, n), 3))
    a, b, c = pos
    # Reordena segmentos: [0..a] + [c..n] + [b..c] + [a..b]
    return seq[0:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturbacion_guiada(seq, tipo_int_arr):
    """
    Perturbación inteligente: extrae k lotes del mismo tipo y los reinserta
    en otra posición. Más dirigida que double-bridge, ayuda a escapar
    mínimos locales relacionados con el setup.
    """
    tipos_list = [0, 1, 2]
    random.shuffle(tipos_list)
    for tipo in tipos_list:
        indices = [i for i, l in enumerate(seq) if tipo_int_arr[l] == tipo]
        if len(indices) < 2:
            continue
        k = random.randint(1, min(3, len(indices)))
        seleccionados = sorted(random.sample(indices, k), reverse=True)
        nuevo = seq[:]
        extraidos = []
        for idx_s in seleccionados:
            extraidos.insert(0, nuevo.pop(idx_s))
        pos_ins = random.randint(0, len(nuevo))
        nuevo = nuevo[:pos_ins] + extraidos + nuevo[pos_ins:]
        if _es_factible(nuevo, tipo_int_arr):
            return nuevo
    # Fallback a double-bridge si no se encontró movimiento factible
    return _double_bridge(seq)


def _ils(seq_inicial, cmax_inicial, tipo_int_arr, proc_flat, setup_3d, t_corte):
    """
    ILS: perturbación + búsqueda local + criterio de aceptación greedy.
    Alterna double-bridge y perturbación guiada.
    Reinicia desde el mejor global cada 20 iteraciones sin mejora.
    """
    mejor = seq_inicial[:]
    mejor_c = cmax_inicial
    actual = seq_inicial[:]
    actual_c = cmax_inicial
    iteracion = 0

    while time.perf_counter() < t_corte:
        # Alternar tipo de perturbación
        if iteracion % 2 == 0:
            perturb = _double_bridge(actual)
        else:
            perturb = _perturbacion_guiada(actual, tipo_int_arr)

        # Presupuesto de tiempo para búsqueda local: 40% del tiempo restante, máx 2 s
        t_restante = t_corte - time.perf_counter()
        t_bl = time.perf_counter() + min(t_restante * 0.4, 2.0)

        s_local, c_local = _busqueda_local(perturb, tipo_int_arr, proc_flat, setup_3d, t_corte=t_bl)

        # Aceptación greedy (solo mejoras)
        if c_local < actual_c:
            actual = s_local[:]
            actual_c = c_local

        # Actualizar mejor global
        if actual_c < mejor_c:
            mejor_c = actual_c
            mejor = actual[:]

        # Reiniciar desde mejor global si estancado
        if iteracion % 20 == 19 and actual_c > mejor_c:
            actual = mejor[:]
            actual_c = mejor_c

        iteracion += 1

    return mejor, mejor_c


# =============================================================================
# FUNCIÓN solve() — punto de entrada requerido por Gradescope
# =============================================================================

def solve(data: dict) -> dict:
    """
    Parámetro: data — diccionario cargado desde data.json
    Retorna:   {"Cmax": int, "secuencia": list[str], "tiempo": int}
    """
    start_ms = time.time()
    t_inicio = time.perf_counter()

    # Tiempo máximo conservador (margen de 5 s ante límite de 60 s de Gradescope)
    TIEMPO_LIMITE = 55.0
    t_corte = t_inicio + TIEMPO_LIMITE

    lotes = data["lotes"]
    setup = data["setup"]

    random.seed(42)

    # Preprocesar estructuras planas para acceso rápido
    names, idx_map, n, tipo_int_arr, proc_flat, setup_3d = _preprocesar(lotes, setup)

    # ----------------------------------------------------------------
    # Fase 1: Multi-start NEH → solución inicial de alta calidad
    # ----------------------------------------------------------------
    seq_neh, cmax_neh = _neh_multistart(n, proc_flat, tipo_int_arr, setup_3d)

    # ----------------------------------------------------------------
    # Fase 2: Búsqueda local intensiva sobre la mejor solución NEH
    # ----------------------------------------------------------------
    t_bl_corte = t_inicio + min(TIEMPO_LIMITE * 0.25, 10.0)
    seq_bl, cmax_bl = _busqueda_local(
        seq_neh, tipo_int_arr, proc_flat, setup_3d, t_corte=t_bl_corte
    )

    mejor_seq = seq_bl[:]
    mejor_cmax = cmax_bl

    # ----------------------------------------------------------------
    # Fase 3: ILS hasta agotar el tiempo
    # ----------------------------------------------------------------
    if time.perf_counter() < t_corte - 1.0:
        seq_ils, cmax_ils = _ils(seq_bl, cmax_bl, tipo_int_arr, proc_flat, setup_3d, t_corte)
        if cmax_ils < mejor_cmax:
            mejor_cmax = cmax_ils
            mejor_seq = seq_ils[:]

    # ----------------------------------------------------------------
    # Validación de seguridad
    # ----------------------------------------------------------------
    if not _es_factible(mejor_seq, tipo_int_arr) or mejor_cmax == float("inf"):
        mejor_seq = seq_neh[:]
        mejor_cmax = _cmax(seq_neh, tipo_int_arr, proc_flat, setup_3d)

    elapsed_ms = int((time.time() - start_ms) * 1000)

    return {
        "Cmax": int(mejor_cmax),
        "secuencia": [names[i] for i in mejor_seq],
        "tiempo": elapsed_ms
    }


# =============================================================================
# CLASE Schedulling — experimentos locales (no modifica solve)
# =============================================================================

try:
    import numpy as np
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

TIEMPO_POR_INSTANCIA = 15.0


class Schedulling:
    def __init__(self, n_instancias: int = 30, semilla: int = 42):
        """No modificar."""
        self.n_instancias = n_instancias
        self.semilla = semilla
        random.seed(semilla)
        if _HAS_PANDAS:
            import numpy as np
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
        """Resuelve una instancia con presupuesto de tiempo TIEMPO_POR_INSTANCIA."""
        t_inicio = time.perf_counter()
        t_corte = t_inicio + TIEMPO_POR_INSTANCIA

        lotes = instancia["lotes"]
        setup = instancia["setup"]

        names, idx_map, n, tipo_int_arr, proc_flat, setup_3d = _preprocesar(lotes, setup)

        # Fase 1: NEH multi-start
        seq_neh, cmax_neh = _neh_multistart(n, proc_flat, tipo_int_arr, setup_3d)

        # Fase 2: búsqueda local
        t_bl_corte = t_inicio + min(TIEMPO_POR_INSTANCIA * 0.25, 3.0)
        seq_bl, cmax_bl = _busqueda_local(
            seq_neh, tipo_int_arr, proc_flat, setup_3d, t_corte=t_bl_corte
        )

        mejor_seq = seq_bl[:]
        mejor_cmax = cmax_bl

        # Fase 3: ILS
        if time.perf_counter() < t_corte - 0.5:
            seq_ils, cmax_ils = _ils(seq_bl, cmax_bl, tipo_int_arr, proc_flat, setup_3d, t_corte)
            if cmax_ils < mejor_cmax:
                mejor_cmax = cmax_ils
                mejor_seq = seq_ils[:]

        if not _es_factible(mejor_seq, tipo_int_arr) or mejor_cmax == float("inf"):
            mejor_seq = seq_neh[:]
            mejor_cmax = _cmax(seq_neh, tipo_int_arr, proc_flat, setup_3d)

        return {
            "secuencia": [names[i] for i in mejor_seq],
            "valor_objetivo": int(mejor_cmax),
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
            print(f"Instancia {i}: Cmax={resultado['valor_objetivo']}, tiempo={round(t1-t0,4)}s")

        tiempo_total = time.time() - inicio_total

        if _HAS_PANDAS:
            import pandas as pd
            self.consolidado = pd.DataFrame(self.resultados)
            promedio_tiempo = self.consolidado["tiempo_seg"].mean()
            promedio_obj = self.consolidado["valor_objetivo"].mean()
        else:
            promedio_tiempo = sum(self.resultados["tiempo_seg"]) / len(self.resultados["tiempo_seg"])
            promedio_obj = sum(self.resultados["valor_objetivo"]) / len(self.resultados["valor_objetivo"])

        print("\n==== REPORTE DE RENDIMIENTO ====")
        print("Cmax promedio:", round(promedio_obj, 2))
        print("Tiempo promedio por instancia:", round(promedio_tiempo, 4), "seg")
        print("Tiempo total de ejecución:", round(tiempo_total, 2), "seg")
        return self.consolidado if _HAS_PANDAS else self.resultados

    def calcular_makespan_penalizado(self, data={}, secuencia=[f"L{i}" for i in range(1, 16)]):
        """
        Calcula el makespan de una secuencia dada.
        Retorna float('inf') si hay violación N→R.
        Usa la misma lógica que solve() para garantizar consistencia.
        """
        lotes = data["lotes"]
        setup = data["setup"]
        names, idx_map, n, tipo_int_arr, proc_flat, setup_3d = _preprocesar(lotes, setup)
        seq_idx = [idx_map[l] for l in secuencia]
        return _cmax(seq_idx, tipo_int_arr, proc_flat, setup_3d)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    with open("data.json") as f:
        data = json.load(f)
    result = solve(data)
    print(json.dumps(result, indent=2))
