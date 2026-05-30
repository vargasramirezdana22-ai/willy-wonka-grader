"""
Lógica de evaluación confiable para el Autocalificador.
Este archivo es una COPIA de la estructura base, usado por el autocalificador
para generar instancias y calcular puntajes de forma segura, sin importar
qué modifique el estudiante en su archivo 'reto_schedulling.py'.
"""

import random
import numpy as np
import time
import pandas as pd
import json

class Schedulling:
    def __init__(self, n_instancias: int = 30, semilla: int = 42):
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

        # Rangos de tiempos de procesamiento por máquina
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
                        else:  # M3
                            tiempo_setup = random.randint(3, 15)
                    setup_dict[maquina][f"{tipo_ant}-{tipo_sig}"] = tiempo_setup

        instancia = {
            "lotes": lotes_dict,
            "setup": setup_dict
        }
        # No necesitamos guardar el json en disco para el grader interno, pero no estorba
        return instancia

    def calcular_makespan_penalizado(self, data, secuencia):

        lotes = data["lotes"]
        setup = data["setup"]
        maquinas = ["M1", "M2", "M3"]

        # penalización N → R
        for i in range(1, len(secuencia)):
            if lotes[secuencia[i-1]]["tipo"] == "N" and lotes[secuencia[i]]["tipo"] == "R":
                return float('inf')

        # tiempos
        tiempos = {m: [] for m in maquinas}
        fin_maquina = {m: 0 for m in maquinas}
        fin_lote = {l: 0 for l in secuencia}

        for lote in secuencia:
            tipo_actual = lotes[lote]["tipo"]

            for m_idx, m in enumerate(maquinas):

                prev_tipo = tiempos[m][-1]["tipo"] if tiempos[m] else None

                # setup correcto
                t_setup = 0
                if prev_tipo is not None and prev_tipo != tipo_actual:
                    t_setup = setup[m][f"{prev_tipo}-{tipo_actual}"]

                # inicio correcto
                if m_idx == 0:
                    inicio = fin_maquina[m] + t_setup
                else:
                    inicio = max(fin_maquina[m], fin_lote[lote]) + t_setup

                dur = lotes[lote][m]
                fin = inicio + dur

                tiempos[m].append({
                    "lote": lote,
                    "tipo": tipo_actual,
                    "inicio": inicio,
                    "fin": fin
                })

                fin_maquina[m] = fin
                fin_lote[lote] = fin

        return max(fin_maquina.values())