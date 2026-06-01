"""
Reto 01 — Willy Wonka | Flow Shop 3 máquinas con setups y restricción N→R
Algoritmo: NEH multi-start + búsqueda local + ILS con elite pool

Versión corregida:
- Sin pandas.
- Sin dependencias no permitidas fuera de librería estándar.
- solve() y Schedulling usan el mismo tiempo por instancia.
- Respeta la restricción N -> R prohibida.
- Retorna el formato requerido por el reto.
"""

import json
import random
import time

# -----------------------------------------------------------------------------
# PARAMETROS
# -----------------------------------------------------------------------------
# Si Gradescope prueba 30 instancias: 30 * 7 = 210 s aprox, menor a 240 s.
# Si quieres aun mas velocidad, baja a 5.0. Si quieres mas busqueda, sube a 8.0.
TIEMPO_POR_INSTANCIA = 7.0
FRACCION_MULTISTART = 0.08
FRACCION_BL = 0.12

MAQUINAS = ["M1", "M2", "M3"]
TIPOS = ["S", "N", "R"]


# -----------------------------------------------------------------------------
# EVALUACION Y FACTIBILIDAD
# -----------------------------------------------------------------------------
def _calcular_cmax(secuencia, lotes, setup):
    """Calcula Cmax en flow shop de 3 maquinas con setup dependiente del tipo."""
    if not secuencia:
        return 0

    for i in range(1, len(secuencia)):
        if lotes[secuencia[i - 1]]["tipo"] == "N" and lotes[secuencia[i]]["tipo"] == "R":
            return float("inf")

    fin = [[0.0] * 3 for _ in range(len(secuencia))]

    for i, lote in enumerate(secuencia):
        tipo_actual = lotes[lote]["tipo"]
        tipo_anterior = lotes[secuencia[i - 1]]["tipo"] if i > 0 else None

        for k, maquina in enumerate(MAQUINAS):
            t_setup = 0
            if tipo_anterior is not None and tipo_anterior != tipo_actual:
                t_setup = setup[maquina][f"{tipo_anterior}-{tipo_actual}"]

            libre_maquina = fin[i - 1][k] if i > 0 else 0.0
            listo_lote = fin[i][k - 1] if k > 0 else 0.0
            fin[i][k] = max(libre_maquina, listo_lote) + t_setup + lotes[lote][maquina]

    return fin[-1][2]


def _es_factible(secuencia, lotes):
    """Verifica que no exista N inmediatamente seguido por R."""
    if len(secuencia) != len(set(secuencia)):
        return False
    for i in range(1, len(secuencia)):
        if lotes[secuencia[i - 1]]["tipo"] == "N" and lotes[secuencia[i]]["tipo"] == "R":
            return False
    return True


# -----------------------------------------------------------------------------
# CONSTRUCCION NEH
# -----------------------------------------------------------------------------
def _neh_una_vez(orden, lotes, setup):
    seq = [orden[0]]
    for lote in orden[1:]:
        mejor_c = float("inf")
        mejor_pos = 0
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [lote] + seq[pos:]
            c = _calcular_cmax(cand, lotes, setup)
            if c < mejor_c:
                mejor_c = c
                mejor_pos = pos
        seq = seq[:mejor_pos] + [lote] + seq[mejor_pos:]
    return seq


def _multi_start_neh(ids, lotes, setup, t_corte):
    suma = {l: lotes[l]["M1"] + lotes[l]["M2"] + lotes[l]["M3"] for l in ids}
    mejores = []

    ordenes_tipo = [
        ["S", "N", "R"],
        ["S", "R", "N"],
        ["R", "S", "N"],
        ["N", "S", "R"],
        ["R", "N", "S"],
        ["N", "R", "S"],
    ]

    for orden_tipos in ordenes_tipo:
        orden = []
        for tipo in orden_tipos:
            grupo = sorted([l for l in ids if lotes[l]["tipo"] == tipo], key=lambda l: suma[l], reverse=True)
            orden.extend(grupo)
        seq = _neh_una_vez(orden, lotes, setup)
        mejores.append((_calcular_cmax(seq, lotes, setup), seq))

    orden_std = sorted(ids, key=lambda l: suma[l], reverse=True)
    seq = _neh_una_vez(orden_std, lotes, setup)
    mejores.append((_calcular_cmax(seq, lotes, setup), seq))

    seed = 1
    while time.perf_counter() < t_corte:
        rng = random.Random(seed)
        noise = min(60, 2 * seed)
        orden = sorted(ids, key=lambda l: suma[l] + rng.uniform(-noise, noise), reverse=True)
        seq = _neh_una_vez(orden, lotes, setup)
        mejores.append((_calcular_cmax(seq, lotes, setup), seq))
        seed += 1

    mejores.sort(key=lambda x: x[0])
    return mejores[0][1], mejores[0][0]


# -----------------------------------------------------------------------------
# BUSQUEDA LOCAL
# -----------------------------------------------------------------------------
def _busqueda_local(seq, lotes, setup, t_corte=None):
    mejor = seq[:]
    mejor_c = _calcular_cmax(mejor, lotes, setup)
    n = len(mejor)
    hubo_mejora_global = True

    while hubo_mejora_global:
        if t_corte and time.perf_counter() >= t_corte:
            break
        hubo_mejora_global = False

        # Or-opt: mover bloques de tamano 1, 2 y 3
        for tam in (1, 2, 3):
            mejora = True
            while mejora:
                if t_corte and time.perf_counter() >= t_corte:
                    return mejor, mejor_c
                mejora = False
                for i in range(n - tam + 1):
                    if mejora:
                        break
                    segmento = mejor[i:i + tam]
                    base = mejor[:i] + mejor[i + tam:]
                    for j in range(len(base) + 1):
                        cand = base[:j] + segmento + base[j:]
                        c = _calcular_cmax(cand, lotes, setup)
                        if c < mejor_c:
                            mejor = cand
                            mejor_c = c
                            mejora = True
                            hubo_mejora_global = True
                            break

        # Swap
        mejora = True
        while mejora:
            if t_corte and time.perf_counter() >= t_corte:
                return mejor, mejor_c
            mejora = False
            for i in range(n - 1):
                if mejora:
                    break
                for j in range(i + 1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = _calcular_cmax(cand, lotes, setup)
                    if c < mejor_c:
                        mejor = cand
                        mejor_c = c
                        mejora = True
                        hubo_mejora_global = True
                        break

        # Reversos cortos
        mejora = True
        while mejora:
            if t_corte and time.perf_counter() >= t_corte:
                return mejor, mejor_c
            mejora = False
            for i in range(n - 1):
                if mejora:
                    break
                for tam in (2, 3, 4):
                    if i + tam > n:
                        continue
                    cand = mejor[:i] + mejor[i:i + tam][::-1] + mejor[i + tam:]
                    c = _calcular_cmax(cand, lotes, setup)
                    if c < mejor_c:
                        mejor = cand
                        mejor_c = c
                        mejora = True
                        hubo_mejora_global = True
                        break

    return mejor, mejor_c


# -----------------------------------------------------------------------------
# ILS
# -----------------------------------------------------------------------------
def _double_bridge(seq):
    n = len(seq)
    if n < 8:
        ns = seq[:]
        i, j = random.sample(range(n), 2)
        ns[i], ns[j] = ns[j], ns[i]
        return ns
    a, b, c = sorted(random.sample(range(1, n), 3))
    return seq[:a] + seq[c:] + seq[b:c] + seq[a:b]


def _perturbacion_tipo(seq, lotes):
    tipos = TIPOS[:]
    random.shuffle(tipos)
    for tipo in tipos:
        idx = [i for i, l in enumerate(seq) if lotes[l]["tipo"] == tipo]
        if len(idx) < 2:
            continue
        k = random.randint(1, min(3, len(idx)))
        seleccionados = sorted(random.sample(idx, k), reverse=True)
        nuevo = seq[:]
        extraidos = []
        for i in seleccionados:
            extraidos.insert(0, nuevo.pop(i))
        pos = random.randint(0, len(nuevo))
        nuevo = nuevo[:pos] + extraidos + nuevo[pos:]
        if _es_factible(nuevo, lotes):
            return nuevo
    return _double_bridge(seq)


def _perturbacion_insert(seq, lotes):
    nuevo = seq[:]
    movimientos = random.randint(1, 3)
    for _ in range(movimientos):
        i = random.randint(0, len(nuevo) - 1)
        lote = nuevo.pop(i)
        j = random.randint(0, len(nuevo))
        nuevo.insert(j, lote)
    return nuevo if _es_factible(nuevo, lotes) else _double_bridge(seq)


def _ils(seq_ini, c_ini, lotes, setup, t_corte):
    mejor = seq_ini[:]
    mejor_c = c_ini
    actual = seq_ini[:]
    actual_c = c_ini
    elite = [(mejor_c, mejor[:])]
    sin_mejora = 0
    it = 0

    while time.perf_counter() < t_corte:
        restante = t_corte - time.perf_counter()
        if restante < 0.08:
            break

        if sin_mejora > 25 and elite:
            _, base = random.choice(elite)
            actual = base[:]
            actual_c = _calcular_cmax(actual, lotes, setup)
            sin_mejora = 0

        if it % 3 == 0:
            perturbada = _double_bridge(actual)
        elif it % 3 == 1:
            perturbada = _perturbacion_tipo(actual, lotes)
        else:
            perturbada = _perturbacion_insert(actual, lotes)

        t_bl = time.perf_counter() + min(0.35, restante * 0.30)
        local, local_c = _busqueda_local(perturbada, lotes, setup, t_corte=t_bl)

        if local_c < actual_c:
            actual = local[:]
            actual_c = local_c
            sin_mejora = 0
        else:
            sin_mejora += 1

        if actual_c < mejor_c:
            mejor = actual[:]
            mejor_c = actual_c
            elite.append((mejor_c, mejor[:]))
            elite.sort(key=lambda x: x[0])
            elite = elite[:5]

        it += 1

    return mejor, mejor_c


# -----------------------------------------------------------------------------
# SOLUCION PRINCIPAL
# -----------------------------------------------------------------------------
def _resolver(data, tiempo_limite):
    inicio = time.perf_counter()
    fin = inicio + tiempo_limite

    lotes = data["lotes"]
    setup = data["setup"]
    ids = list(lotes.keys())

    random.seed(42)

    t_ms = inicio + tiempo_limite * FRACCION_MULTISTART
    seq, cmax = _multi_start_neh(ids, lotes, setup, t_corte=t_ms)

    t_bl = min(fin, time.perf_counter() + tiempo_limite * FRACCION_BL)
    seq, cmax = _busqueda_local(seq, lotes, setup, t_corte=t_bl)

    if time.perf_counter() < fin - 0.1:
        seq_ils, c_ils = _ils(seq, cmax, lotes, setup, t_corte=fin)
        if c_ils < cmax:
            seq = seq_ils[:]
            cmax = c_ils

    if (not _es_factible(seq, lotes)) or cmax == float("inf"):
        orden = sorted(ids, key=lambda l: lotes[l]["M1"] + lotes[l]["M2"] + lotes[l]["M3"], reverse=True)
        seq = _neh_una_vez(orden, lotes, setup)
        cmax = _calcular_cmax(seq, lotes, setup)

    return seq, int(cmax)


def solve(data: dict) -> dict:
    """Formato del reto: retorna Cmax, secuencia y tiempo en milisegundos."""
    inicio = time.perf_counter()
    secuencia, cmax = _resolver(data, TIEMPO_POR_INSTANCIA)
    return {
        "Cmax": cmax,
        "secuencia": secuencia,
        "tiempo": int((time.perf_counter() - inicio) * 1000),
    }


# -----------------------------------------------------------------------------
# CLASE COMPATIBLE CON EL ZIP / AUTOGRADER LOCAL
# -----------------------------------------------------------------------------
class Schedulling:
    def __init__(self, n_instancias: int = 30, semilla: int = 42):
        self.n_instancias = n_instancias
        self.semilla = semilla
        random.seed(semilla)
        self.resultados = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        self.consolidado = None

    def generar_instancia(self, indice):
        tipos = ["S", "N", "R"]
        lotes = [f"L{i}" for i in range(1, 16)]
        rangos = {"M1": (25, 45), "M2": (20, 40), "M3": (10, 25)}

        lotes_dict = {}
        for lote in lotes:
            lotes_dict[lote] = {
                "tipo": random.choice(tipos),
                "M1": random.randint(*rangos["M1"]),
                "M2": random.randint(*rangos["M2"]),
                "M3": random.randint(*rangos["M3"]),
            }

        setup_dict = {}
        for maquina in MAQUINAS:
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

    def resolver_instancia(self, instancia: dict) -> dict:
        secuencia, cmax = _resolver(instancia, TIEMPO_POR_INSTANCIA)
        return {"secuencia": secuencia, "valor_objetivo": int(cmax)}

    def ejecutar_experimentos(self):
        print(f"Ejecutando {self.n_instancias} instancias...\n")
        self.resultados = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        inicio_total = time.time()

        for i in range(1, self.n_instancias + 1):
            instancia = self.generar_instancia(i)
            t0 = time.time()
            resultado = self.resolver_instancia(instancia)
            t1 = time.time()
            tiempo = round(t1 - t0, 4)
            valor = resultado.get("valor_objetivo", 0)

            self.resultados["instancia"].append(i)
            self.resultados["valor_objetivo"].append(valor)
            self.resultados["tiempo_seg"].append(tiempo)
            print(f"Instancia {i}: Cmax={valor} en {tiempo}s")

        tiempo_total = time.time() - inicio_total
        tiempos = self.resultados["tiempo_seg"]
        valores = self.resultados["valor_objetivo"]
        promedio_tiempo = sum(tiempos) / len(tiempos) if tiempos else 0
        promedio_cmax = sum(valores) / len(valores) if valores else 0

        print("\n==== REPORTE DE RENDIMIENTO ====")
        print("Tiempo promedio:", round(promedio_tiempo, 4), "seg")
        print("Cmax promedio:  ", round(promedio_cmax, 1))
        print("Tiempo total:   ", round(tiempo_total, 2), "seg")

        self.consolidado = self.resultados
        return self.resultados

    def calcular_makespan_penalizado(self, data, secuencia):
        if not _es_factible(secuencia, data["lotes"]):
            return float("inf")
        return _calcular_cmax(secuencia, data["lotes"], data["setup"])


if __name__ == "__main__":
    try:
        with open("data.json", encoding="utf-8") as f:
            data = json.load(f)
        print(json.dumps(solve(data), indent=2, ensure_ascii=False))
    except FileNotFoundError:
        sch = Schedulling(n_instancias=3)
        print(sch.ejecutar_experimentos())
