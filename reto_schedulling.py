"""
Reto 01 — Willy Wonka — Flow Shop 3 máquinas
Algoritmo: NEH-Beam + GRASP-ILS + SA adaptativo + Population restart
Todas las estructuras internas usan índices enteros para máxima velocidad.
Formato de salida: {"Cmax": int, "secuencia": list[str], "tiempo": int}
"""

import json
import time
import random
import math

# =============================================================================
# ESTRUCTURAS GLOBALES (se inicializan en solve)
# =============================================================================
_tipos = []          # tipos[i] = 0(S),1(N),2(R)
_proc = []           # proc[i][k] = tiempo lote i en máquina k
_setup_flat = []     # setup_flat[k][ta][tb] = setup máquina k de tipo ta a tb
_lotes_ids = []      # lista de strings L1..L15
_n = 0

TIPO_MAP = {'S': 0, 'N': 1, 'R': 2}
TIPO_INV = ['S', 'N', 'R']
MAQ_NAMES = ['M1', 'M2', 'M3']


def _init_globals(data):
    global _tipos, _proc, _setup_flat, _lotes_ids, _n
    lotes = data['lotes']
    setup = data['setup']
    _lotes_ids = list(lotes.keys())
    _n = len(_lotes_ids)
    _tipos = [TIPO_MAP[lotes[l]['tipo']] for l in _lotes_ids]
    _proc = [[lotes[l]['M1'], lotes[l]['M2'], lotes[l]['M3']] for l in _lotes_ids]
    _setup_flat = [[[0] * 3 for _ in range(3)] for _ in range(3)]
    for k, m in enumerate(MAQ_NAMES):
        for ti, t1 in enumerate(TIPO_INV):
            for tj, t2 in enumerate(TIPO_INV):
                _setup_flat[k][ti][tj] = setup[m][f'{t1}-{t2}']


# =============================================================================
# EVALUACIÓN RÁPIDA (índices enteros, listas planas)
# =============================================================================

def _cmax(seq):
    """Calcula Cmax. Retorna inf si hay N(1)->R(2)."""
    n = len(seq)
    tipos = _tipos
    proc = _proc
    sf = _setup_flat

    # Verificar restricción N->R
    for i in range(n - 1):
        if tipos[seq[i]] == 1 and tipos[seq[i + 1]] == 2:
            return float('inf')

    fin = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(n):
        lote = seq[i]
        ta = tipos[lote]
        ta_ant = tipos[seq[i - 1]] if i > 0 else -1

        for k in range(3):
            ts = sf[k][ta_ant][ta] if (i > 0 and ta_ant != ta) else 0
            lm = fin[i - 1][k] if i > 0 else 0.0
            ll = fin[i][k - 1] if k > 0 else 0.0
            fin[i][k] = max(lm, ll) + ts + proc[lote][k]

    return fin[-1][2]


def _factible(seq):
    tipos = _tipos
    for i in range(len(seq) - 1):
        if tipos[seq[i]] == 1 and tipos[seq[i + 1]] == 2:
            return False
    return True


# =============================================================================
# CONSTRUCCIÓN — NEH con Beam Search
# =============================================================================

def _neh_beam(orden, beam_width=5):
    """NEH con beam search. orden: lista de índices ordenados."""
    beam = [[orden[0]]]
    for lote in orden[1:]:
        candidatos = []
        for seq in beam:
            for pos in range(len(seq) + 1):
                nueva = seq[:pos] + [lote] + seq[pos:]
                c = _cmax(nueva)
                candidatos.append((c, nueva))
        candidatos.sort(key=lambda x: x[0])
        beam = [s for _, s in candidatos[:beam_width]]
    return beam[0], _cmax(beam[0])


def _orden_neh(perturbacion=0.0, rng=None):
    """Orden NEH por suma de tiempos con perturbación aleatoria."""
    if rng is None:
        rng = random
    suma = [_proc[i][0] + _proc[i][1] + _proc[i][2] for i in range(_n)]
    if perturbacion > 0:
        suma = [s + rng.uniform(-perturbacion * s, perturbacion * s) for s in suma]
    return sorted(range(_n), key=lambda i: suma[i], reverse=True)


def _construir_soluciones_iniciales(t_corte, n_aleatorias=15):
    """
    Genera múltiples soluciones iniciales con NEH-Beam y variantes aleatorias.
    Retorna lista de (cmax, seq) ordenada.
    """
    soluciones = []
    rng = random.Random(42)

    # NEH-Beam con orden determinista (beam 3, 5, 8)
    orden_base = _orden_neh()
    for bw in [3, 5, 8]:
        if time.perf_counter() > t_corte:
            break
        try:
            s, c = _neh_beam(orden_base, beam_width=bw)
            soluciones.append((c, s[:]))
        except Exception:
            pass

    # NEH-Beam con órdenes perturbados
    for seed in range(n_aleatorias):
        if time.perf_counter() > t_corte:
            break
        rng2 = random.Random(seed * 7 + 13)
        orden = _orden_neh(perturbacion=0.15, rng=rng2)
        try:
            s, c = _neh_beam(orden, beam_width=3)
            soluciones.append((c, s[:]))
        except Exception:
            pass

    # Ordenar por Cmax
    soluciones.sort(key=lambda x: x[0])
    return soluciones


# =============================================================================
# BÚSQUEDA LOCAL RÁPIDA (first-improvement, índices enteros)
# =============================================================================

def _bl_fast(seq, cmax_actual, t_corte=None):
    """
    Búsqueda local con first-improvement:
    Or-opt(1,2,3) + swap + reverse_segment
    Retorna (mejor_seq, mejor_cmax)
    """
    mejor = seq[:]
    mejor_c = cmax_actual
    n = len(mejor)
    mejorado_global = True

    while mejorado_global:
        if t_corte and time.perf_counter() > t_corte:
            break
        mejorado_global = False

        # --- Or-opt: extraer segmento de tamaño tam y reinsertar ---
        for tam in [1, 2, 3]:
            mejorado = True
            while mejorado:
                if t_corte and time.perf_counter() > t_corte:
                    return mejor, mejor_c
                mejorado = False
                for i in range(n - tam + 1):
                    seg = mejor[i:i + tam]
                    base = mejor[:i] + mejor[i + tam:]
                    ln = len(base)
                    for j in range(ln + 1):
                        if i <= j <= i + tam:  # misma posición efectiva
                            continue
                        cand = base[:j] + seg + base[j:]
                        c = _cmax(cand)
                        if c < mejor_c - 0.5:
                            mejor_c = c
                            mejor = cand
                            mejorado = True
                            mejorado_global = True
                            break
                    if mejorado:
                        break

        # --- Swap (2-opt de posición) ---
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mejor_c
            mejorado = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    mejor[i], mejor[j] = mejor[j], mejor[i]
                    c = _cmax(mejor)
                    if c < mejor_c - 0.5:
                        mejor_c = c
                        mejorado = True
                        mejorado_global = True
                        break
                    else:
                        mejor[i], mejor[j] = mejor[j], mejor[i]
                if mejorado:
                    break

        # --- Inversión de subsecuencias (reverse segment) ---
        mejorado = True
        while mejorado:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mejor_c
            mejorado = False
            for i in range(n - 1):
                for j in range(i + 2, n + 1):
                    cand = mejor[:i] + mejor[i:j][::-1] + mejor[j:]
                    c = _cmax(cand)
                    if c < mejor_c - 0.5:
                        mejor_c = c
                        mejor = cand
                        mejorado = True
                        mejorado_global = True
                        break
                if mejorado:
                    break

    return mejor, mejor_c


# =============================================================================
# PERTURBACIONES
# =============================================================================

def _double_bridge(seq):
    n = len(seq)
    if n < 8:
        nueva = seq[:]
        idxs = random.sample(range(n), min(4, n))
        vals = [nueva[i] for i in idxs]
        random.shuffle(vals)
        for i, idx in enumerate(idxs):
            nueva[idx] = vals[i]
        return nueva
    pos = sorted(random.sample(range(1, n), 3))
    a, b, c = pos
    return seq[0:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturbacion_tipo(seq, k_min=2, k_max=4):
    """Extrae un bloque de lotes del mismo tipo y lo mueve."""
    tipos_en_seq = list({_tipos[l] for l in seq})
    random.shuffle(tipos_en_seq)
    for tipo in tipos_en_seq:
        indices = [i for i, l in enumerate(seq) if _tipos[l] == tipo]
        if len(indices) < 2:
            continue
        k = random.randint(k_min, min(k_max, len(indices)))
        sel = sorted(random.sample(indices, k), reverse=True)
        nuevo = seq[:]
        ext = []
        for idx in sel:
            ext.insert(0, nuevo.pop(idx))
        pos = random.randint(0, len(nuevo))
        nuevo = nuevo[:pos] + ext + nuevo[pos:]
        if _factible(nuevo):
            return nuevo
    return _double_bridge(seq)


def _perturbacion_random_insert(seq, n_moves=3):
    nuevo = seq[:]
    for _ in range(n_moves):
        i = random.randint(0, len(nuevo) - 1)
        l = nuevo.pop(i)
        j = random.randint(0, len(nuevo))
        nuevo.insert(j, l)
    return nuevo if _factible(nuevo) else _double_bridge(seq)


def _perturbacion_segmento_invertido(seq):
    """Invierte un segmento aleatorio de tamaño 3-6."""
    n = len(seq)
    tam = random.randint(3, min(6, n))
    i = random.randint(0, n - tam)
    nuevo = seq[:i] + seq[i:i + tam][::-1] + seq[i + tam:]
    return nuevo if _factible(nuevo) else _double_bridge(seq)


# =============================================================================
# ILS PRINCIPAL
# =============================================================================

def _ils(seq_init, c_init, t_corte, t_bl_max=2.5):
    mejor = seq_init[:]
    mejor_c = c_init
    actual = seq_init[:]
    actual_c = c_init
    sin_mejora = 0
    iteracion = 0

    perturbaciones = [
        _double_bridge,
        _perturbacion_tipo,
        _perturbacion_random_insert,
        _perturbacion_segmento_invertido,
    ]

    while time.perf_counter() < t_corte:
        t_restante = t_corte - time.perf_counter()
        if t_restante < 0.2:
            break

        # Seleccionar perturbación de forma rotativa + aleatoria
        if sin_mejora > 12:
            perturb_fn = _double_bridge  # perturbación fuerte
            sin_mejora = 0
            # Reiniciar desde la mejor global
            actual = mejor[:]
            actual_c = mejor_c
        else:
            perturb_fn = perturbaciones[iteracion % len(perturbaciones)]

        nueva = perturb_fn(actual)
        if not _factible(nueva):
            nueva = _double_bridge(actual)

        # BL con tiempo limitado
        t_bl = min(t_restante * 0.30, t_bl_max)
        s_local, c_local = _bl_fast(nueva, _cmax(nueva),
                                     t_corte=time.perf_counter() + t_bl)

        # Aceptación: solo mejoras (best acceptance)
        if c_local < actual_c:
            actual = s_local[:]
            actual_c = c_local
            sin_mejora = 0
        else:
            sin_mejora += 1

        if actual_c < mejor_c:
            mejor_c = actual_c
            mejor = actual[:]

        iteracion += 1

    return mejor, mejor_c


# =============================================================================
# SIMULATED ANNEALING (exploración de espacio grande)
# =============================================================================

def _sa(seq_init, c_init, t_corte, T0=None, cooling=0.9998):
    """
    SA con temperatura adaptativa y múltiples tipos de movimiento.
    Usa fast-cmax con índices enteros.
    """
    seq = seq_init[:]
    c = c_init
    mejor = seq[:]
    mejor_c = c

    # Temperatura inicial: aceptar ~20% de soluciones que empeoran en ~50 unidades
    if T0 is None:
        T0 = max(30.0, c_init * 0.05)
    T = T0
    n = len(seq)

    while time.perf_counter() < t_corte:
        # Elegir movimiento aleatorio
        op = random.randint(0, 3)

        if op == 0:  # swap
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            if i == j:
                continue
            seq[i], seq[j] = seq[j], seq[i]
            nc = _cmax(seq)
            delta = nc - c
            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                c = nc
            else:
                seq[i], seq[j] = seq[j], seq[i]

        elif op == 1:  # insert
            i = random.randint(0, n - 1)
            l = seq.pop(i)
            j = random.randint(0, n - 1)
            seq.insert(j, l)
            nc = _cmax(seq)
            delta = nc - c
            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                c = nc
            else:
                seq.pop(j)
                seq.insert(i, l)

        elif op == 2:  # reverse segment
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, min(i + 6, n))
            nuevo = seq[:i] + seq[i:j][::-1] + seq[j:]
            nc = _cmax(nuevo)
            delta = nc - c
            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                seq = nuevo
                c = nc

        else:  # or-opt 1
            i = random.randint(0, n - 1)
            l = seq.pop(i)
            j = random.randint(0, n - 1)
            seq.insert(j, l)
            nc = _cmax(seq)
            delta = nc - c
            if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
                c = nc
            else:
                seq.pop(j)
                seq.insert(i, l)

        if c < mejor_c:
            mejor_c = c
            mejor = seq[:]

        T *= cooling

    return mejor, mejor_c


# =============================================================================
# FUNCIÓN PRINCIPAL SOLVE
# =============================================================================

def solve(data: dict) -> dict:
    """
    Parámetro: data — diccionario con 'lotes' y 'setup'
    Retorna: {"Cmax": int, "secuencia": list[str], "tiempo": int}
    """
    t_wall = time.time()
    t0 = time.perf_counter()
    LIMITE = 57.5  # margen de 2.5 seg
    t_corte_global = t0 + LIMITE

    # Inicializar estructuras globales
    _init_globals(data)
    lotes = data['lotes']

    # =========================================================================
    # FASE 1: Construcción — NEH-Beam con múltiples ordenamientos (~5 seg)
    # =========================================================================
    t_construccion = t0 + 6.0
    soluciones = _construir_soluciones_iniciales(t_construccion, n_aleatorias=20)

    if not soluciones:
        # Fallback mínimo
        seq_fallback = list(range(_n))
        c_fallback = _cmax(seq_fallback)
        soluciones = [(c_fallback, seq_fallback)]

    mejor_c, mejor_seq = soluciones[0]

    # =========================================================================
    # FASE 2: Búsqueda local sobre las top-4 soluciones (~10 seg)
    # =========================================================================
    t_bl_fase2 = t0 + 18.0
    candidatos = []
    t_por_candidato = max(1.0, (t_bl_fase2 - time.perf_counter()) / min(4, len(soluciones)))

    for c_init, s_init in soluciones[:4]:
        if time.perf_counter() > t_bl_fase2:
            candidatos.append((c_init, s_init))
            continue
        t_limite_bl = time.perf_counter() + t_por_candidato
        s_bl, c_bl = _bl_fast(s_init, c_init, t_corte=min(t_limite_bl, t_bl_fase2))
        candidatos.append((c_bl, s_bl))

    candidatos.sort(key=lambda x: x[0])
    mejor_c, mejor_seq = candidatos[0]

    # =========================================================================
    # FASE 3: SA rápido para escapar del óptimo local (~8 seg)
    # =========================================================================
    t_sa = t0 + 28.0
    if time.perf_counter() < t_sa - 1:
        # SA desde las 2 mejores soluciones
        for c_seed, s_seed in candidatos[:2]:
            if time.perf_counter() > t_sa:
                break
            t_sa_i = min(time.perf_counter() + (t_sa - time.perf_counter()) / 2,
                         t_corte_global - 5)
            s_sa, c_sa = _sa(s_seed[:], c_seed, t_sa_i,
                             T0=max(20.0, c_seed * 0.04), cooling=0.9997)
            if c_sa < mejor_c:
                mejor_c = c_sa
                mejor_seq = s_sa[:]

    # =========================================================================
    # FASE 4: ILS hasta agotar el tiempo restante
    # =========================================================================
    if time.perf_counter() < t_corte_global - 2:
        t_restante = t_corte_global - time.perf_counter()
        t_bl_ils = min(t_restante * 0.08, 2.5)
        s_ils, c_ils = _ils(mejor_seq[:], mejor_c, t_corte_global, t_bl_max=t_bl_ils)
        if c_ils < mejor_c:
            mejor_c = c_ils
            mejor_seq = s_ils[:]

    # =========================================================================
    # VALIDACIÓN FINAL
    # =========================================================================
    if not _factible(mejor_seq) or mejor_c == float('inf'):
        # Fallback de emergencia: NEH clásico sin perturbación
        orden = _orden_neh()
        s_fb, c_fb = _neh_beam(orden, beam_width=1)
        mejor_seq = s_fb
        mejor_c = c_fb if c_fb != float('inf') else 999999

    # Convertir índices a nombres de lote
    secuencia_nombres = [_lotes_ids[i] for i in mejor_seq]
    elapsed_ms = int((time.time() - t_wall) * 1000)

    return {
        "Cmax": int(mejor_c),
        "secuencia": secuencia_nombres,
        "tiempo": elapsed_ms
    }


# =============================================================================
# EJECUCIÓN LOCAL
# =============================================================================

if __name__ == "__main__":
    import sys

    data_ejemplo = {
        "lotes": {
            "L1":  {"tipo": "S", "M1": 38, "M2": 33, "M3": 19},
            "L2":  {"tipo": "N", "M1": 41, "M2": 27, "M3": 21},
            "L3":  {"tipo": "R", "M1": 29, "M2": 37, "M3": 14},
            "L4":  {"tipo": "S", "M1": 35, "M2": 24, "M3": 22},
            "L5":  {"tipo": "N", "M1": 44, "M2": 38, "M3": 16},
            "L6":  {"tipo": "R", "M1": 27, "M2": 31, "M3": 18},
            "L7":  {"tipo": "S", "M1": 40, "M2": 28, "M3": 25},
            "L8":  {"tipo": "N", "M1": 33, "M2": 40, "M3": 12},
            "L9":  {"tipo": "R", "M1": 31, "M2": 22, "M3": 17},
            "L10": {"tipo": "S", "M1": 39, "M2": 35, "M3": 15},
            "L11": {"tipo": "N", "M1": 42, "M2": 26, "M3": 23},
            "L12": {"tipo": "R", "M1": 28, "M2": 34, "M3": 13},
            "L13": {"tipo": "S", "M1": 36, "M2": 29, "M3": 20},
            "L14": {"tipo": "N", "M1": 45, "M2": 25, "M3": 24},
            "L15": {"tipo": "R", "M1": 30, "M2": 39, "M3": 11},
        },
        "setup": {
            "M1": {"S-S": 0, "S-N": 15, "S-R": 35, "N-S": 25, "N-N": 0, "N-R": 40, "R-S": 20, "R-N": 25, "R-R": 0},
            "M2": {"S-S": 0, "S-N": 10, "S-R": 25, "N-S": 15, "N-N": 0, "N-R": 30, "R-S": 10, "R-N": 20, "R-R": 0},
            "M3": {"S-S": 0, "S-N":  5, "S-R": 10, "N-S":  8, "N-N": 0, "N-R": 12, "R-S":  6, "R-N":  9, "R-R": 0},
        }
    }

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data_ejemplo = json.load(f)

    resultado = solve(data_ejemplo)
    print(json.dumps(resultado, indent=2))

    # Verificación
    lotes = data_ejemplo["lotes"]
    seq = resultado["secuencia"]
    tipo_map_local = {'S': 0, 'N': 1, 'R': 2}
    factible = all(
        not (lotes[seq[i]]["tipo"] == "N" and lotes[seq[i+1]]["tipo"] == "R")
        for i in range(len(seq)-1)
    )
    print(f"\n✓ Secuencia factible: {factible}")
    print(f"✓ Tiempo total:       {resultado['tiempo']} ms")
