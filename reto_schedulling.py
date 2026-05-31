"""
RETO 01 — Willy Wonka | Flow Shop 3 máquinas
Estrategia: Multi-start NEH + Búsqueda Local intensiva (Or-opt 1/2/3 + 2-opt + reverse + swap)
           + ILS con perturbación double-bridge y pool elite

Toda la lógica está en resolver_instancia(). Sin dependencias externas.
"""

import random
import numpy as np
import time
import pandas as pd
import json

# ─────────────────────────────────────────────────────────────
# PARÁMETROS DE TIEMPO  (30 × 1.8s = 54s << 60s límite)
# ─────────────────────────────────────────────────────────────
T_POR_INSTANCIA   = 1.80   # segundos por instancia
F_MULTISTART      = 0.35   # fracción para multi-start NEH
F_BL_INICIAL      = 0.10   # fracción para BL inicial tras NEH
# resto → ILS

# ─────────────────────────────────────────────────────────────
# NÚCLEO: Cmax con setup dependiente del tipo
# ─────────────────────────────────────────────────────────────

def _cmax(seq, lotes, setup):
    """Calcula Cmax exacto; devuelve inf si hay violación N→R."""
    n = len(seq)
    if n == 0:
        return 0
    # Verificar restricción N→R en O(n)
    tipos_seq = [lotes[l]["tipo"] for l in seq]
    for i in range(n - 1):
        if tipos_seq[i] == "N" and tipos_seq[i + 1] == "R":
            return float("inf")

    # Flow-shop DP: fin[i][k] = fin del lote i en máquina k
    MACS = ("M1", "M2", "M3")
    fin = [[0.0] * 3 for _ in range(n)]
    for i, lote in enumerate(seq):
        ta = tipos_seq[i]
        tp = tipos_seq[i - 1] if i > 0 else None
        for k, m in enumerate(MACS):
            ts = setup[m][f"{tp}-{ta}"] if tp is not None and tp != ta else 0
            prev_mac = fin[i][k - 1] if k > 0 else 0.0
            prev_lot = fin[i - 1][k]  if i > 0 else 0.0
            fin[i][k] = max(prev_mac, prev_lot) + ts + lotes[lote][m]
    return fin[-1][2]


def _factible(seq, lotes):
    tipos = [lotes[l]["tipo"] for l in seq]
    for i in range(len(tipos) - 1):
        if tipos[i] == "N" and tipos[i + 1] == "R":
            return False
    return True


# ─────────────────────────────────────────────────────────────
# NEH
# ─────────────────────────────────────────────────────────────

def _neh(orden, lotes, setup):
    seq = [orden[0]]
    for lote in orden[1:]:
        bc = float("inf")
        bp = 0
        for p in range(len(seq) + 1):
            c = _cmax(seq[:p] + [lote] + seq[p:], lotes, setup)
            if c < bc:
                bc = c
                bp = p
        seq = seq[:bp] + [lote] + seq[bp:]
    return seq


def _multi_start_neh(ids, lotes, setup, t_corte):
    suma = {l: lotes[l]["M1"] + lotes[l]["M2"] + lotes[l]["M3"] for l in ids}
    best_c = float("inf")
    best_seq = None

    # 1) NEH estándar (orden decreciente suma total)
    orden_std = sorted(ids, key=lambda l: suma[l], reverse=True)
    seq = _neh(orden_std, lotes, setup)
    c = _cmax(seq, lotes, setup)
    if c < best_c:
        best_c = c
        best_seq = seq[:]

    # 2) 6 órdenes por tipo (agrupa por tipo, luego decreciente por suma)
    for orden_tipos in [
        ["S", "N", "R"], ["S", "R", "N"], ["R", "S", "N"],
        ["N", "S", "R"], ["R", "N", "S"], ["N", "R", "S"],
    ]:
        s = []
        for t in orden_tipos:
            s.extend(sorted([l for l in ids if lotes[l]["tipo"] == t],
                            key=lambda l: suma[l], reverse=True))
        seq = _neh(s, lotes, setup)
        c = _cmax(seq, lotes, setup)
        if c < best_c:
            best_c = c
            best_seq = seq[:]

    # 3) NEH por cada máquina individualmente
    for mk in ("M1", "M2", "M3"):
        orden_mk = sorted(ids, key=lambda l: lotes[l][mk], reverse=True)
        seq = _neh(orden_mk, lotes, setup)
        c = _cmax(seq, lotes, setup)
        if c < best_c:
            best_c = c
            best_seq = seq[:]

    # 4) Aleatorios con ruido hasta agotar tiempo
    seed = 0
    while time.perf_counter() < t_corte:
        seed += 1
        rng = random.Random(seed)
        noise = min(seed * 4, 120)
        orden = sorted(ids,
                       key=lambda l: suma[l] + rng.uniform(-noise, noise),
                       reverse=True)
        seq = _neh(orden, lotes, setup)
        # BL rápida sobre cada start NEH aleatorio
        seq, c = _bl(seq, lotes, setup, t_corte=min(t_corte, time.perf_counter() + 0.05))
        if c < best_c:
            best_c = c
            best_seq = seq[:]

    return best_seq, best_c


# ─────────────────────────────────────────────────────────────
# BÚSQUEDA LOCAL — first-improve, 5 vecindarios
# ─────────────────────────────────────────────────────────────

def _bl(seq, lotes, setup, t_corte=None):
    mejor = seq[:]
    mc = _cmax(mejor, lotes, setup)
    n = len(mejor)
    mejora_global = True

    while mejora_global:
        if t_corte and time.perf_counter() > t_corte:
            break
        mejora_global = False

        # --- Or-opt: segmentos de tamaño 1, 2, 3 ---
        for tam in (1, 2, 3):
            mejora = True
            while mejora:
                if t_corte and time.perf_counter() > t_corte:
                    return mejor, mc
                mejora = False
                for i in range(n - tam + 1):
                    if mejora:
                        break
                    seg = mejor[i:i + tam]
                    base = mejor[:i] + mejor[i + tam:]
                    for j in range(len(base) + 1):
                        if j == i:   # misma posición → sin cambio
                            continue
                        cand = base[:j] + seg + base[j:]
                        c = _cmax(cand, lotes, setup)
                        if c < mc:
                            mc = c
                            mejor = cand
                            mejora = True
                            mejora_global = True
                            break

        # --- Swap (intercambio de pares) ---
        mejora = True
        while mejora:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejora = False
            for i in range(n):
                if mejora:
                    break
                for j in range(i + 1, n):
                    cand = mejor[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = _cmax(cand, lotes, setup)
                    if c < mc:
                        mc = c
                        mejor = cand
                        mejora = True
                        mejora_global = True
                        break

        # --- Reverse de subsegmentos (2-opt estilo flow-shop) ---
        mejora = True
        while mejora:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejora = False
            for i in range(n - 1):
                if mejora:
                    break
                for l in range(2, min(6, n - i + 1)):
                    cand = mejor[:i] + mejor[i:i + l][::-1] + mejor[i + l:]
                    c = _cmax(cand, lotes, setup)
                    if c < mc:
                        mc = c
                        mejor = cand
                        mejora = True
                        mejora_global = True
                        break

        # --- 3-opt: reverse de segmento largo ---
        mejora = True
        while mejora:
            if t_corte and time.perf_counter() > t_corte:
                return mejor, mc
            mejora = False
            for i in range(n - 2):
                if mejora:
                    break
                for j in range(i + 2, n):
                    cand = mejor[:i] + mejor[i:j + 1][::-1] + mejor[j + 1:]
                    c = _cmax(cand, lotes, setup)
                    if c < mc:
                        mc = c
                        mejor = cand
                        mejora = True
                        mejora_global = True
                        break

    return mejor, mc


# ─────────────────────────────────────────────────────────────
# PERTURBACIONES PARA ILS
# ─────────────────────────────────────────────────────────────

def _double_bridge(seq):
    """Perturbación clásica de ILS para TSP/scheduling."""
    n = len(seq)
    if n < 8:
        i, j = sorted(random.sample(range(1, n), 2))
        c = seq[:]
        c[i], c[j] = c[j], c[i]
        return c
    a, b, c = sorted(random.sample(range(1, n), 3))
    return seq[:a] + seq[c:] + seq[b:c] + seq[a:b]


def _perturb_por_tipo(seq, lotes):
    """Extrae lotes de un tipo y los reinserta en posición aleatoria."""
    tipos = ["S", "N", "R"]
    random.shuffle(tipos)
    for tipo in tipos:
        idx = [i for i, l in enumerate(seq) if lotes[l]["tipo"] == tipo]
        if len(idx) < 2:
            continue
        k = random.randint(1, min(4, len(idx)))
        sel = sorted(random.sample(idx, k), reverse=True)
        nuevo = seq[:]
        extraidos = []
        for i in sel:
            extraidos.insert(0, nuevo.pop(i))
        p = random.randint(0, len(nuevo))
        nuevo = nuevo[:p] + extraidos + nuevo[p:]
        if _factible(nuevo, lotes):
            return nuevo
    return _double_bridge(seq)


def _perturb_bloque(seq, lotes):
    """Mueve un lote a otra posición, respetando factibilidad."""
    tipos = ["S", "N", "R"]
    random.shuffle(tipos)
    for tipo in tipos:
        idx = [i for i, l in enumerate(seq) if lotes[l]["tipo"] == tipo]
        if not idx:
            continue
        i = random.choice(idx)
        nuevo = seq[:]
        elem = nuevo.pop(i)
        p = random.randint(0, len(nuevo))
        nuevo = nuevo[:p] + [elem] + nuevo[p:]
        if _factible(nuevo, lotes):
            return nuevo
    return _double_bridge(seq)


# ─────────────────────────────────────────────────────────────
# ILS CON POOL ELITE
# ─────────────────────────────────────────────────────────────

def _ils(seq_ini, c_ini, lotes, setup, t_corte):
    mejor = seq_ini[:]
    mc = c_ini
    actual = seq_ini[:]
    ac = c_ini
    elite = [(mc, mejor[:])]
    sin_mejora = 0
    it = 0

    while time.perf_counter() < t_corte:
        t_rest = t_corte - time.perf_counter()
        if t_rest < 0.02:
            break

        # Reiniciar desde elite si llevamos muchas iteraciones sin mejora
        if sin_mejora > 25:
            _, base = random.choice(elite)
            actual = base[:]
            ac = _cmax(actual, lotes, setup)
            sin_mejora = 0

        # Elegir perturbación de forma rotativa
        r = it % 3
        if r == 0:
            perturb = _double_bridge(actual)
        elif r == 1:
            perturb = _perturb_por_tipo(actual, lotes)
        else:
            perturb = _perturb_bloque(actual, lotes)

        # BL corta sobre el vecino perturbado
        t_bl = time.perf_counter() + min(t_rest * 0.30, 0.25)
        sl, cl = _bl(perturb, lotes, setup, t_corte=t_bl)

        if cl < ac:
            actual = sl[:]
            ac = cl
            sin_mejora = 0
        else:
            sin_mejora += 1

        if ac < mc:
            mc = ac
            mejor = actual[:]
            elite.append((mc, mejor[:]))
            elite.sort(key=lambda x: x[0])
            elite = elite[:6]   # mantener top-6

        it += 1

    return mejor, mc


# ─────────────────────────────────────────────────────────────
# FALLBACK DETERMINISTA
# ─────────────────────────────────────────────────────────────

def _fallback(ids, lotes, setup):
    """NEH estándar simple como seguro de factibilidad."""
    suma = {l: lotes[l]["M1"] + lotes[l]["M2"] + lotes[l]["M3"] for l in ids}
    orden = sorted(ids, key=lambda l: suma[l], reverse=True)
    return _neh(orden, lotes, setup)


# ─────────────────────────────────────────────────────────────
# CLASE SCHEDULLING  (estructura exigida por GradeScope)
# ─────────────────────────────────────────────────────────────

class Schedulling:
    def __init__(self, n_instancias: int = 30, semilla: int = 42):
        self.n_instancias = n_instancias
        self.semilla = semilla
        random.seed(semilla)
        np.random.seed(semilla)
        self.resultados = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        self.consolidado = None

    # ── NO MODIFICAR ──────────────────────────────────────────
    def generar_instancia(self, indice):
        tipos = ["S", "N", "R"]
        lotes = [f"L{i}" for i in range(1, 16)]
        rangos = {"M1": (25, 45), "M2": (20, 40), "M3": (10, 25)}

        lotes_dict = {}
        for lote in lotes:
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
            for ta in tipos:
                for tb in tipos:
                    if ta == tb:
                        t = 0
                    elif maquina == "M1":
                        t = random.randint(10, 40)
                    elif maquina == "M2":
                        t = random.randint(5, 30)
                    else:
                        t = random.randint(3, 15)
                    setup_dict[maquina][f"{ta}-{tb}"] = t

        instancia = {"lotes": lotes_dict, "setup": setup_dict}
        with open(f"instancia_{indice}.json", "w", encoding="utf-8") as f:
            json.dump(instancia, f, indent=4, ensure_ascii=False)
        return instancia

    # ── RESOLVER ─────────────────────────────────────────────
    def resolver_instancia(self, instancia: dict) -> dict:
        t0 = time.perf_counter()
        t_fin = t0 + T_POR_INSTANCIA

        lotes = instancia["lotes"]
        setup = instancia["setup"]
        ids   = list(lotes.keys())

        # 1) Multi-start NEH
        t_ms = t0 + T_POR_INSTANCIA * F_MULTISTART
        seq, cmax = _multi_start_neh(ids, lotes, setup, t_corte=t_ms)

        # 2) Búsqueda local inicial
        t_bl = t_ms + T_POR_INSTANCIA * F_BL_INICIAL
        seq, cmax = _bl(seq, lotes, setup, t_corte=t_bl)

        # 3) ILS hasta agotar tiempo
        if time.perf_counter() < t_fin - 0.05:
            sq2, c2 = _ils(seq, cmax, lotes, setup, t_corte=t_fin)
            if c2 < cmax:
                cmax = c2
                seq  = sq2[:]

        # Fallback de seguridad
        if not _factible(seq, lotes) or cmax == float("inf"):
            seq  = _fallback(ids, lotes, setup)
            cmax = _cmax(seq, lotes, setup)

        return {
            "secuencia":      seq,
            "valor_objetivo": int(cmax),
        }

    # ── NO MODIFICAR ──────────────────────────────────────────
    def ejecutar_experimentos(self):
        print(f"Ejecutando {self.n_instancias} instancias...\n")
        self.resultados = {"instancia": [], "valor_objetivo": [], "tiempo_seg": []}
        inicio_total = time.time()

        for i in range(1, self.n_instancias + 1):
            instancia = self.generar_instancia(i)
            t0 = time.time()
            resultado = self.resolver_instancia(instancia)
            t1 = time.time()

            self.resultados["instancia"].append(i)
            self.resultados["valor_objetivo"].append(resultado["valor_objetivo"])
            self.resultados["tiempo_seg"].append(round(t1 - t0, 4))
            print(f"Instancia {i}: Cmax={resultado['valor_objetivo']}  |  {round(t1-t0,3)}s")

        tiempo_total = time.time() - inicio_total
        self.consolidado = pd.DataFrame(self.resultados)
        promedio_tiempo = self.consolidado["tiempo_seg"].mean()

        print("\n==== REPORTE DE RENDIMIENTO ====")
        print(f"Cmax promedio:              {self.consolidado['valor_objetivo'].mean():.1f}")
        print(f"Tiempo promedio/instancia:  {promedio_tiempo:.4f} seg")
        print(f"Tiempo total ejecución:     {round(tiempo_total, 2)} seg")
        return self.consolidado

    def calcular_makespan_penalizado(self, data={}, secuencia=[f"L{i}" for i in range(1, 16)]):
        lotes  = data["lotes"]
        setup  = data["setup"]
        return _cmax(secuencia, lotes, setup)


# ─────────────────────────────────────────────────────────────
# Ejecución directa  (GradeScope / prueba local)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys

    # Si existe data.json, ejecutar en modo solve() individual
    if os.path.exists("data.json"):
        with open("data.json", encoding="utf-8") as f:
            data = json.load(f)
        t0  = time.perf_counter()
        sch = Schedulling(1)
        res = sch.resolver_instancia(data)
        ms  = int((time.perf_counter() - t0) * 1000)
        print(json.dumps({"Cmax": res["valor_objetivo"],
                          "secuencia": res["secuencia"],
                          "tiempo": ms}, indent=2))
    else:
        sch = Schedulling(30)
        df  = sch.ejecutar_experimentos()
        print("\nConsolidado final:")
        print(df)
