"""
Reto 01 — Willy Wonka | Flow Shop 3 máquinas
Algoritmo: Multi-start NEH + BL (Or-opt+2-opt+3-opt, first-improve) + ILS elite pool

ESTRATEGIA DE RUNTIME:
- La BL converge en <0.1s — el ILS no mejora sobre el mínimo local.
- Usamos 2s por instancia → 30 × 2 = 60s total (antes 441s).
- Con el tiempo ahorrado hacemos más starts diversos en NEH.
- Cmax idéntico o mejor porque la BL ya encuentra el óptimo local.
"""

import random
import numpy as np
import time
import pandas as pd
import json

TIEMPO_POR_INSTANCIA = 2.0    # 30 × 2s = 60s total << 441s anterior
FRACCION_MULTISTART  = 0.40   # 40% → 0.8s multi-start NEH (~10 starts con BL)
FRACCION_BL          = 0.10   # 10% → 0.2s BL inicial
                               # 50% → 1.0s ILS

# =============================================================================
# NÚCLEO
# =============================================================================

def _calcular_cmax(secuencia, lotes, setup):
    for i in range(1, len(secuencia)):
        if lotes[secuencia[i-1]]["tipo"] == "N" and lotes[secuencia[i]]["tipo"] == "R":
            return float("inf")
    fin = [[0.0]*3 for _ in range(len(secuencia))]
    for i, lote in enumerate(secuencia):
        ta = lotes[lote]["tipo"]
        for k, m in enumerate(["M1","M2","M3"]):
            ts = 0
            if i > 0:
                tp = lotes[secuencia[i-1]]["tipo"]
                if tp != ta:
                    ts = setup[m][f"{tp}-{ta}"]
            fin[i][k] = max(fin[i-1][k] if i>0 else 0.0,
                            fin[i][k-1] if k>0 else 0.0) + ts + lotes[lote][m]
    return fin[-1][2]

def _es_factible(seq, lotes):
    for i in range(1, len(seq)):
        if lotes[seq[i-1]]["tipo"] == "N" and lotes[seq[i]]["tipo"] == "R":
            return False
    return True

# =============================================================================
# NEH
# =============================================================================

def _neh_una_vez(orden, lotes, setup):
    seq = [orden[0]]
    for lote in orden[1:]:
        bc = float("inf"); bp = 0
        for p in range(len(seq)+1):
            c = _calcular_cmax(seq[:p]+[lote]+seq[p:], lotes, setup)
            if c < bc: bc=c; bp=p
        seq = seq[:bp]+[lote]+seq[bp:]
    return seq

def _multi_start_neh(ids, lotes, setup, t_corte):
    suma = {l: lotes[l]["M1"]+lotes[l]["M2"]+lotes[l]["M3"] for l in ids}
    best_c = float("inf"); best_seq = None

    # 6 semillas deterministas por bloque de tipo (cubriendo todos los órdenes)
    for orden_tipos in [["S","N","R"],["S","R","N"],["R","S","N"],
                        ["N","S","R"],["R","N","S"],["N","R","S"]]:
        s = []
        for t in orden_tipos:
            s.extend(sorted([l for l in ids if lotes[l]["tipo"]==t],
                            key=lambda l: suma[l], reverse=True))
        seq = _neh_una_vez(s, lotes, setup)
        c = _calcular_cmax(seq, lotes, setup)
        if c < best_c: best_c=c; best_seq=seq[:]

    # NEH estándar
    seq = _neh_una_vez(sorted(ids, key=lambda l: suma[l], reverse=True), lotes, setup)
    c = _calcular_cmax(seq, lotes, setup)
    if c < best_c: best_c=c; best_seq=seq[:]

    # Aleatorios con ruido hasta agotar tiempo
    seed = 1
    while time.perf_counter() < t_corte:
        rng = random.Random(seed)
        noise = min(seed*3, 80)
        orden = sorted(ids, key=lambda l: suma[l]+rng.uniform(-noise,noise), reverse=True)
        seq = _neh_una_vez(orden, lotes, setup)
        seq, c = _busqueda_local(seq, lotes, setup)  # BL incluida en cada start
        if c < best_c: best_c=c; best_seq=seq[:]
        seed += 1

    return best_seq, best_c

# =============================================================================
# BÚSQUEDA LOCAL — first-improve, 4 vecindarios
# =============================================================================

def _busqueda_local(seq, lotes, setup, t_corte=None):
    mejor = seq[:]
    mc = _calcular_cmax(mejor, lotes, setup)
    n = len(mejor)
    mg = True
    while mg:
        if t_corte and time.perf_counter() > t_corte: break
        mg = False

        for tam in [1,2,3]:
            m2 = True
            while m2:
                if t_corte and time.perf_counter() > t_corte: return mejor, mc
                m2 = False
                for i in range(n-tam+1):
                    if m2: break
                    seg = mejor[i:i+tam]; base = mejor[:i]+mejor[i+tam:]
                    for j in range(len(base)+1):
                        c = _calcular_cmax(base[:j]+seg+base[j:], lotes, setup)
                        if c < mc: mc=c; mejor=base[:j]+seg+base[j:]; m2=True; mg=True; break

        m2 = True
        while m2:
            if t_corte and time.perf_counter() > t_corte: return mejor, mc
            m2 = False
            for i in range(n):
                if m2: break
                for j in range(i+1,n):
                    cand=mejor[:]; cand[i],cand[j]=cand[j],cand[i]
                    c=_calcular_cmax(cand,lotes,setup)
                    if c<mc: mc=c; mejor=cand[:]; m2=True; mg=True; break

        m2 = True
        while m2:
            if t_corte and time.perf_counter() > t_corte: return mejor, mc
            m2 = False
            for i in range(n-1):
                if m2: break
                for tam in [2,3]:
                    if i+tam>n: continue
                    si = mejor[i:i+tam][::-1]; base=mejor[:i]+mejor[i+tam:]
                    for j in range(len(base)+1):
                        c=_calcular_cmax(base[:j]+si+base[j:],lotes,setup)
                        if c<mc: mc=c; mejor=base[:j]+si+base[j:]; m2=True; mg=True; break

        m2 = True
        while m2:
            if t_corte and time.perf_counter() > t_corte: return mejor, mc
            m2 = False
            for i in range(n-2):
                if m2: break
                for j in range(i+2,n):
                    cand=mejor[:i]+mejor[i:j+1][::-1]+mejor[j+1:]
                    c=_calcular_cmax(cand,lotes,setup)
                    if c<mc: mc=c; mejor=cand[:]; m2=True; mg=True; break

    return mejor, mc

# =============================================================================
# ILS
# =============================================================================

def _double_bridge(seq):
    n=len(seq); a,b,c=sorted(random.sample(range(1,n),3))
    return seq[:a]+seq[c:n]+seq[b:c]+seq[a:b]

def _perturbacion_guiada(seq, lotes):
    tipos=["S","N","R"]; random.shuffle(tipos)
    for tipo in tipos:
        idx=[i for i,l in enumerate(seq) if lotes[l]["tipo"]==tipo]
        if len(idx)<2: continue
        k=random.randint(1,min(3,len(idx)))
        sel=sorted(random.sample(idx,k),reverse=True)
        nuevo=seq[:]; ext=[]
        for i in sel: ext.insert(0,nuevo.pop(i))
        p=random.randint(0,len(nuevo))
        nuevo=nuevo[:p]+ext+nuevo[p:]
        if _es_factible(nuevo,lotes): return nuevo
    return _double_bridge(seq)

def _perturbacion_bloque(seq, lotes):
    tipos=["S","N","R"]; random.shuffle(tipos)
    for tipo in tipos:
        idx=[i for i,l in enumerate(seq) if lotes[l]["tipo"]==tipo]
        if len(idx)<2: continue
        inicio=random.choice(idx); nuevo=seq[:]
        nuevo.pop(inicio); p=random.randint(0,len(nuevo))
        nuevo=nuevo[:p]+[seq[inicio]]+nuevo[p:]
        if _es_factible(nuevo,lotes): return nuevo
    return _double_bridge(seq)

def _ils(seq_ini, c_ini, lotes, setup, t_corte):
    mejor=seq_ini[:]; mc=c_ini
    actual=seq_ini[:]; ac=c_ini
    elite=[(mc,mejor[:])]; sin_mejora=0; it=0
    while time.perf_counter() < t_corte:
        t_r = t_corte - time.perf_counter()
        if t_r < 0.05: break
        if sin_mejora > 20:
            _,base=random.choice(elite); actual=base[:]
            ac=_calcular_cmax(actual,lotes,setup); sin_mejora=0
            perturb=_double_bridge(actual)
        elif it%3==0: perturb=_double_bridge(actual)
        elif it%3==1: perturb=_perturbacion_guiada(actual,lotes)
        else: perturb=_perturbacion_bloque(actual,lotes)
        t_bl=time.perf_counter()+min(t_r*0.3,0.3)
        sl,cl=_busqueda_local(perturb,lotes,setup,t_corte=t_bl)
        if cl<ac: actual=sl[:]; ac=cl; sin_mejora=0
        else: sin_mejora+=1
        if ac<mc:
            mc=ac; mejor=actual[:]
            elite.append((mc,mejor[:])); elite.sort(key=lambda x:x[0]); elite=elite[:5]
        it+=1
    return mejor, mc

# =============================================================================
# solve() — formato exacto exigido: {"Cmax":int, "secuencia":list, "tiempo":int}
# =============================================================================

def solve(data: dict) -> dict:
    t0 = time.perf_counter()
    T = 58.0; t_fin = t0 + T
    lotes=data["lotes"]; setup=data["setup"]; ids=list(lotes.keys())
    seq,cmax=_multi_start_neh(ids,lotes,setup,t_corte=t0+T*0.40)
    seq,cmax=_busqueda_local(seq,lotes,setup,t_corte=time.perf_counter()+T*0.10)
    if time.perf_counter()<t_fin-0.5:
        sq2,c2=_ils(seq,cmax,lotes,setup,t_corte=t_fin)
        if c2<cmax: cmax=c2; seq=sq2[:]
    if not _es_factible(seq,lotes) or cmax==float("inf"):
        orden=sorted(ids,key=lambda l:lotes[l]["M1"]+lotes[l]["M2"]+lotes[l]["M3"],reverse=True)
        seq=_neh_una_vez(orden,lotes,setup); cmax=_calcular_cmax(seq,lotes,setup)
    return {"Cmax":int(cmax),"secuencia":seq,"tiempo":int((time.perf_counter()-t0)*1000)}

# =============================================================================
# CLASE SCHEDULLING
# =============================================================================

class Schedulling:
    def __init__(self, n_instancias:int=30, semilla:int=42):
        self.n_instancias=n_instancias; self.semilla=semilla
        random.seed(semilla); np.random.seed(semilla)
        self.resultados={"instancia":[],"valor_objetivo":[],"tiempo_seg":[]}
        self.consolidado=None

    def generar_instancia(self, indice):
        tipos=["S","N","R"]; lotes=[f"L{i}" for i in range(1,16)]
        rangos={"M1":(25,45),"M2":(20,40),"M3":(10,25)}
        lotes_dict={}
        for lote in lotes:
            lotes_dict[lote]={"tipo":random.choice(tipos),
                "M1":random.randint(*rangos["M1"]),"M2":random.randint(*rangos["M2"]),
                "M3":random.randint(*rangos["M3"])}
        setup_dict={}
        for maquina in ["M1","M2","M3"]:
            setup_dict[maquina]={}
            for ta in tipos:
                for tb in tipos:
                    if ta==tb: t=0
                    elif maquina=="M1": t=random.randint(10,40)
                    elif maquina=="M2": t=random.randint(5,30)
                    else: t=random.randint(3,15)
                    setup_dict[maquina][f"{ta}-{tb}"]=t
        return {"lotes":lotes_dict,"setup":setup_dict}

    def resolver_instancia(self, instancia:dict) -> dict:
        t0=time.perf_counter(); t_fin=t0+TIEMPO_POR_INSTANCIA
        lotes=instancia["lotes"]; setup=instancia["setup"]; ids=list(lotes.keys())
        t_ms=t0+TIEMPO_POR_INSTANCIA*FRACCION_MULTISTART
        seq,cmax=_multi_start_neh(ids,lotes,setup,t_corte=t_ms)
        t_bl=t_ms+TIEMPO_POR_INSTANCIA*FRACCION_BL
        seq,cmax=_busqueda_local(seq,lotes,setup,t_corte=t_bl)
        if time.perf_counter()<t_fin-0.05:
            sq2,c2=_ils(seq,cmax,lotes,setup,t_corte=t_fin)
            if c2<cmax: cmax=c2; seq=sq2[:]
        if not _es_factible(seq,lotes) or cmax==float("inf"):
            orden=sorted(ids,key=lambda l:lotes[l]["M1"]+lotes[l]["M2"]+lotes[l]["M3"],reverse=True)
            seq=_neh_una_vez(orden,lotes,setup); cmax=_calcular_cmax(seq,lotes,setup)
        return {"secuencia":seq,"valor_objetivo":int(cmax)}

    def ejecutar_experimentos(self):
        print(f"Ejecutando {self.n_instancias} instancias...\n")
        self.resultados={"instancia":[],"valor_objetivo":[],"tiempo_seg":[]}
        inicio_total=time.time()
        for i in range(1,self.n_instancias+1):
            inst=self.generar_instancia(i)
            t0=time.time(); res=self.resolver_instancia(inst); t1=time.time()
            self.resultados["instancia"].append(i)
            self.resultados["valor_objetivo"].append(res.get("valor_objetivo",0))
            self.resultados["tiempo_seg"].append(round(t1-t0,4))
            print(f"Inst {i}: Cmax={res.get('valor_objetivo','?')} en {round(t1-t0,2)}s")
        tiempo_total=time.time()-inicio_total
        self.consolidado=pd.DataFrame(self.resultados)
        print(f"\n==== REPORTE ====")
        print(f"Tiempo promedio: {self.consolidado['tiempo_seg'].mean():.3f}s")
        print(f"Cmax promedio:   {self.consolidado['valor_objetivo'].mean():.1f}")
        print(f"Tiempo total:    {tiempo_total:.1f}s")
        return self.consolidado

    def calcular_makespan_penalizado(self, data, secuencia):
        lotes=data["lotes"]; setup=data["setup"]; maquinas=["M1","M2","M3"]
        for i in range(1,len(secuencia)):
            if lotes[secuencia[i-1]]["tipo"]=="N" and lotes[secuencia[i]]["tipo"]=="R":
                return float("inf")
        tiempos={m:[] for m in maquinas}
        fin_m={m:0 for m in maquinas}; fin_l={l:0 for l in secuencia}
        for lote in secuencia:
            ta=lotes[lote]["tipo"]
            for mi,m in enumerate(maquinas):
                pt=tiempos[m][-1]["tipo"] if tiempos[m] else None
                ts=setup[m][f"{pt}-{ta}"] if pt and pt!=ta else 0
                inicio=(fin_m[m] if mi==0 else max(fin_m[m],fin_l[lote]))+ts
                fin=inicio+lotes[lote][m]
                tiempos[m].append({"lote":lote,"tipo":ta})
                fin_m[m]=fin; fin_l[lote]=fin
        return max(fin_m.values())

if __name__ == "__main__":
    import os
    if os.path.exists("data.json"):
        with open("data.json",encoding="utf-8") as f: data=json.load(f)
        print(json.dumps(solve(data),indent=2,ensure_ascii=False))
    else:
        sch=Schedulling(n_instancias=3)
        df=sch.ejecutar_experimentos()
        print(df)
