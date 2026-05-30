from grader_logic import Schedulling as GraderSchedulling
from reto_schedulling import Schedulling as StudentSchedulling
import numpy as np
import json
import time
import os

if __name__ == '__main__':
    
    # 1. Instanciar el Generador/Juez (Código confiable del profesor)
    grader = GraderSchedulling()
    
    # 2. Instanciar el Solucionador (Código del estudiante)
    try:
        student = StudentSchedulling()
    except Exception as e:
        print(json.dumps({"score": 0, "output": f"Error al inicializar tu clase Schedulling: {str(e)}"}))
        exit(1)

    # 3. Ejecutar 30 instancias
    total_makespan = 0
    total_time = 0
    n_instancias = 30
    
    print(f"Evaluando {n_instancias} instancias...")
    
    for i in range(1, n_instancias + 1):
        # Generar instancia i
        data = grader.generar_instancia(i)
        
        # Ejecutar solución del estudiante
        t0 = time.time()
        try:
            result = student.resolver_instancia(data)
        except Exception as e:
            print(json.dumps({"score": 0, "output": f"Error en instancia {i}: {str(e)}"}))
            exit(1)
        t1 = time.time()
        
        # Evaluar resultado
        secuencia = result.get("secuencia", [])
        
        # Validar secuencia
        expected_lotes = set(data['lotes'].keys())
        student_lotes = set(secuencia)
        
        if len(secuencia) != len(expected_lotes) or student_lotes != expected_lotes:
             print(f"Instancia {i}: Secuencia inválida.")
             makespan = float('inf')
        else:
            makespan = grader.calcular_makespan_penalizado(data, secuencia)
            
        total_makespan += makespan
        total_time += (t1 - t0)
        
        # print(f"Instancia {i}: Makespan = {makespan}, Tiempo = {t1-t0:.4f}s")

    avg_makespan = total_makespan / n_instancias
    
    results = {
        "score": float(avg_makespan),
        "leaderboard": [
            {"name": "Avg Makespan", "value": float(avg_makespan), "order": "asc"},
            {"name": "Total Runtime", "value": float(np.round(total_time, 4)), "order": "asc"}
        ]
    }

    # Ensure output directory exists and write file
    os.makedirs('/autograder/results', exist_ok=True)
    with open('/autograder/results/results.json', 'w') as f:
        json.dump(results, f)

    print(json.dumps(results))