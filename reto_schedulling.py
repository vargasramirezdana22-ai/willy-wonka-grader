"""
Plantilla base para GradeScope — Reto 01 Willy Wonka
Algoritmo: NEH + Búsqueda Local (Or-opt + 2-opt) + ILS (double-bridge)
Solo librerías estándar + numpy
"""

import random
import numpy as np
import time
import pandas as pd
import json
import math


# =============================================================================
# NÚCLEO DEL ALGORITMO — Flow Shop 3 máquinas con setups y restricción N→R
# =============================================================================

def _calcular_cmax(secuencia, lotes, setup):
    for i in range(1, len(secuencia)):
        if (lotes[secuencia[i - 1]]["tipo"] == "N" and
                lotes[secuencia[i]]["tipo"] == "R"):
            return float("inf")

    fin = [[0.0] * 3 for _ in range(len(secuencia))]
    for i, lote in enumerate(secuencia):
        tipo_act = lotes[lote]["tipo"]
        for k, m in enumerate(["M1", "M2", "M3"]):
            t_setup = 0
            if i > 0:
                tipo_ant = lotes[secuencia[i - 1]]["tipo"]
                if tipo_ant != tipo_act:
                    t_setup = setup[m][f"{tipo_ant}-{tipo_act}"]
            lm = fin[i - 1][k] if i > 0 else 0.0
            ll = fin[i][k - 1] if k > 0 else 0.0
            fin[i][k] = max(lm, ll) + t_setup + lotes[lote][m]
    return fin[-1][2]


def _es_factible(secuencia, lotes):
    for i in range(1, len(secuencia)):
        if (lotes[secuencia[i - 1]]["tipo"] == "N" and
                lotes[secuencia[i]]["tipo"] == "R"):
            return False
    return True


def _neh(lotes_ids, lotes, setup):
    suma = {l: lotes[l]["M1"] + lotes[l]["M2"] + lotes[l]["M3"] for l in lotes_ids}
    orden = sorted(lotes_ids, key=lambda l: suma[l], reverse=True)
    secuencia = [orden[0]]
    for lote in orden[1:]:
        mejor_c = float("inf")
        mejor_pos = 0
        for pos in range(len(secuencia) + 1):
            candidata = secuencia[:pos] + [lote] + secuencia[pos:]
            c = _calcular_cmax(candidata, lotes, setup)
            if c < mejor_c:
                mejor_c = c
                mejor_pos = pos
        secuencia = secuencia[:mejor_pos] + [lote] + secuencia[mejor_pos:]
    return secuencia


def _busqueda_local(seq, lotes, setup):
    mejor = seq[:]
    mejor_c = _calcular_cmax(mejor, lotes, setup)
    n = len(mejor)
    mejorado_global = True

    while mejorado_global:
        mejorado_global = False

        for tam in [1, 2, 3]:
            mejorado = True
            while mejorado:
                mejorado = False
                for i in range(n - tam + 1):
                    segmento = mejor[i:i + tam]
                    base = mejor[:i] + mejor[i + tam:]
                    for j in range(len(base) + 1):
                        candidata = base[:j] + segmento + base[j:]
                        c = _calcular_cmax(candidata, lotes, setup)
                        if c < mejor_c:
                            mejor_c = c
                            mejor = candidata[:]
                            mejorado = True
                            mejorado_global = True

        mejorado = True
        while mejorado:
            mejorado = False
            for i in range(n):
                for j in range(i + 1, n):
                    candidata = mejor[:]
                    candidata[i], candidata[j] = candidata[j], candidata[i]
                    c = _calcular_cmax(candidata, lotes, setup)
                    if c < mejor_c:
                        mejor_c = c
                        mejor = candidata[:]
                        mejorado = True
                        mejorado_global = True

        mejorado = True
        while mejorado:
            mejorado = False
            for i in range(n - 1):
                for tam in [2, 3]:
                    if i + tam > n:
                        continue
                    segmento_inv = mejor[i:i + tam][::-1]
                    base = mejor[:i] + mejor[i + tam:]
                    for j in range(len(base) + 1):
                        candidata = base[:j] + segmento_inv + base[j:]
                        c = _calcular_cmax(candidata, lotes, setup)
                        if c < mejor_c:
                            mejor_c = c
                            mejor = candidata[:]
                            mejorado = True
                            mejorado_global = True

    return mejor, mejor_c


def _double_bridge(seq):
    n = len(seq)
    pos = sorted(random.sample(range(1, n), 3))
    a, b, c = pos
    return seq[0:a] + seq[c:n] + seq[b:c] + seq[a:b]


def _perturbacion_guiada(seq, lotes):
    tipos = ["S", "N", "R"]
    random.shuffle(tipos)
    for tipo in tipos:
        indices = [i for i, l in enumerate(seq) if lotes[l]["tipo"] == tipo]
        if len(indices) < 2:
            continue
        k = random.randint(1, min(3, len(indices)))
        seleccionados = sorted(random.sample(indices, k), reverse=True)
        nuevo = seq[:]
        extraidos = []
        for idx in seleccionados:
            extraidos.insert(0, nuevo.pop(idx))
        pos = random.randint(0, len(nuevo))
        nuevo = nuevo[:pos] + extraidos + nuevo[pos:]
        if _es_factible(nuevo, lotes):
            return nuevo
    return _double_bridge(seq)


def _ils(seq_inicial, cmax_inicial, lotes, setup, t_corte):
    mejor = seq_inicial[:]
    mejor_c = cmax_inicial
    actual = seq_inicial[:]
    actual_c = cmax_inicial
    iteracion = 0

    while time.perf_counter() < t_corte:
        if iteracion % 2 == 0:
            perturb = _double_bridge(actual)
        else:
            perturb = _perturbacion_guiada(actual, lotes)

        s_local, c_local = _busqueda_local(perturb, lotes, setup)

        if c_local < actual_c:
            actual = s_local[:]
            actual_c = c_local

        if actual_c < mejor_c:
            mejor_c = actual_c
            mejor = actual[:]

        if iteracion % 20 == 19 and actual_c > mejor_c:
            actual = mejor[:]
            actual_c = mejor_c

        iteracion += 1

    return mejor, mejor_c


# =============================================================================
# CLASE SCHEDULLING — estructura requerida por Gradescope
# =============================================================================

class Schedulling:
    def __init__(self, n_instancias: int = 30, semilla: int = 42):
        """
        Inicializa la clase con número de instancias y semilla.
        No modificar esta función.
        """
        self.n_instancias = n_instancias
        self.semilla = semilla
        random.seed(semilla)
        np.random.seed(semilla)
        self.resultados = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        self.consolidado = None

    def generar_instancia(self, indice):
        """
        Genera una instancia aleatoria del problema de lotes y setups.
        """
        tipos = ["S", "N", "R"]
        lotes = [f"L{i}" for i in range(1, 16)]

        rangos = {
            "M1": (25, 45),
            "M2": (20, 40),
            "M3": (10, 25)
        }

        lotes_dict = {}
        for lote in lotes:
            tipo = random.choice(tipos)
            lotes_dict[lote] = {
                "tipo": tipo,
                "M1": random.randint(*rangos["M1"]),
                "M2": random.randint(*rangos["M2"]),
                "M3": random.randint(*rangos["M3"])
            }

        setup_dict = {}
        for maquina in ["M1", "M2", "M3"]:
            setup_dict[maquina] = {}
            for tipo_ant in tipos:
                for tipo_sig in tipos:
                    if tipo_ant == tipo_sig:
                        tiempo_setup = 0
                    else:
                        if maquina == "M1":
                            tiempo_setup = random.randint(10, 40)
                        elif maquina == "M2":
                            tiempo_setup = random.randint(5, 30)
                        else:
                            tiempo_setup = random.randint(3, 15)
                    setup_dict[maquina][f"{tipo_ant}-{tipo_sig}"] = tiempo_setup

        instancia = {"lotes": lotes_dict, "setup": setup_dict}
        return instancia

    def resolver_instancia(self, instancia: dict):
        """
        Recibe una instancia y retorna la secuencia optimizada y el Cmax.
        Usa NEH + Búsqueda Local (Or-opt + 2-opt) + ILS con double-bridge.
        """
        T_LIMITE = 58.0
        t_inicio = time.perf_counter()
        t_corte = t_inicio + T_LIMITE

        lotes = instancia["lotes"]
        setup = instancia["setup"]
        lotes_ids = list(lotes.keys())

        # Fase 1: NEH
        seq_neh = _neh(lotes_ids, lotes, setup)

        # Fase 2: Búsqueda local exhaustiva
        seq_bl, cmax_bl = _busqueda_local(seq_neh, lotes, setup)

        mejor_seq = seq_bl[:]
        mejor_cmax = cmax_bl

        # Fase 3: ILS hasta agotar el tiempo
        if time.perf_counter() < t_corte - 1.0:
            seq_ils, cmax_ils = _ils(seq_bl, cmax_bl, lotes, setup, t_corte)
            if cmax_ils < mejor_cmax:
                mejor_cmax = cmax_ils
                mejor_seq = seq_ils[:]

        # Validación de seguridad
        if not _es_factible(mejor_seq, lotes) or mejor_cmax == float("inf"):
            mejor_seq = seq_neh
            mejor_cmax = _calcular_cmax(seq_neh, lotes, setup)

        return {
            "secuencia": mejor_seq,
            "valor_objetivo": int(mejor_cmax)
        }

    def ejecutar_experimentos(self):
        """
        Genera y resuelve las instancias, mide tiempos de ejecución
        y consolida los resultados.
        NO modificar esta función.
        """
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

            print(f"Instancia {i} optimizada en {round(t1 - t0, 4)} seg")

        tiempo_total = time.time() - inicio_total
        self.consolidado = pd.DataFrame(self.resultados)
        promedio_tiempo = self.consolidado["tiempo_seg"].mean()

        print("\n==== REPORTE DE RENDIMIENTO ====")
        print("Tiempo promedio por instancia:", round(promedio_tiempo, 4), "seg")
        print("Tiempo total de ejecución:", round(tiempo_total, 2), "seg")

        return self.consolidado

    def calcular_makespan_penalizado(self, data={}, secuencia=[f'L{i}' for i in range(1, 16)]):
        lotes = data["lotes"]
        setup = data["setup"]
        maquinas = ["M1", "M2", "M3"]

        penalizado = False
        for i in range(1, len(secuencia)):
            if lotes[secuencia[i-1]]["tipo"] == "N" and lotes[secuencia[i]]["tipo"] == "R":
                penalizado = True
                break

        fin_maquina = {m: 0 for m in maquinas}
        fin_lote = {l: 0 for l in secuencia}

        for lote in secuencia:
            tipo_actual = lotes[lote]["tipo"]
            for m_idx, m in enumerate(maquinas):
                prev_tipo = None
                if fin_maquina[m] > 0:
                    for l in secuencia:
                        if fin_lote[l] <= fin_maquina[m] and l != lote:
                            prev_tipo = lotes[l]["tipo"]
                t_setup = 0
                if prev_tipo is not None and prev_tipo != tipo_actual:
                    t_setup = setup[m][f"{prev_tipo}-{tipo_actual}"]
                if m_idx == 0:
                    inicio = fin_maquina[m] + t_setup
                else:
                    inicio = max(fin_maquina[m], fin_lote[lote]) + t_setup
                duracion = lotes[lote][m]
                fin = inicio + duracion
                fin_maquina[m] = fin
                fin_lote[lote] = fin

        makespan = max(fin_maquina.values())
        if penalizado:
            makespan = float('inf')
        return makespan


# ----------------------------------------------------------
# Ejecución directa (GradeScope)
# ----------------------------------------------------------
if __name__ == "__main__":
    sch = Schedulling(30)
    df_resultados = sch.ejecutar_experimentos()
    print("\nConsolidado final:")
    print(df_resultados)
