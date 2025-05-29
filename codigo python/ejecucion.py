import os
from funciones import *
import matplotlib.pyplot as plt
import datetime
import numpy as np
import openpyxl


def guardar_plot_y_scores(data, nombre_base,nombreGraf,n,m):

    nombreGraf = os.path.splitext(os.path.basename(nombreGraf))[0]
    filename = rf"..\DATA\{nombreGraf}_M{m}"

    # Preparar datos para estadística
    all_generaciones = [ [x[0] for x in duplas] for duplas, _ in data ]
    all_valores = [ [x[1] for x in duplas] for duplas, _ in data ]
    min_len = min(len(g) for g in all_generaciones)
    # Recortar a la longitud mínima para alinear
    generaciones = all_generaciones[0][:min_len]
    valores_matrix = np.array([v[:min_len] for v in all_valores])
    media = np.mean(valores_matrix, axis=0)
    std = np.std(valores_matrix, axis=0)
    mejor = np.min(valores_matrix, axis=0)
    peor = np.max(valores_matrix, axis=0)
    valor_optimo = buscar_valor_optimo(nombreGraf,m)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(generaciones, media, label="Media", color="blue")
    plt.fill_between(generaciones, media-std, media+std, color="blue", alpha=0.2, label="±1 std")
    plt.plot(generaciones, mejor, label="Mejor curva", color="green", linestyle="--")
    plt.plot(generaciones, peor, label="Peor curva", color="red", linestyle=":")
    plt.axhline(y=valor_optimo, color='orange', linestyle='--', label=f'Óptimo: {valor_optimo}')
    plt.xlabel("Generación")
    plt.ylabel("Valor")
    plt.title(f"Evolucion con {nombre_base}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename}.png",dpi=300)
    plt.close()

    # Guardar mejores scores
    with open(f"{filename}.txt", "w") as f:
        for i, (_, score) in enumerate(data):
            f.write(f"Run {i+1}: {score}\n")



def buscar_valor_optimo(nombre_grafo, m, archivo=r"..\SALBP.csv"):
    import csv
    nombre_grafo = nombre_grafo.strip().upper()
    with open(archivo, encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            # Saltar filas vacías o de encabezado
            if not row or len(row) < 3:
                continue
            # Buscar filas con nombre de grafo y m
            nombre = str(row[0]).strip().upper()
            try:
                m_val = int(row[1])
            except Exception:
                continue
            if nombre == nombre_grafo and m_val == m:
                # El valor óptimo está en la columna 2 (índice 2)
                optimo = row[2].strip()
                # Si es un rango, devolver como string, si es número, como int
                if '[' in optimo or ']' in optimo:
                    return optimo
                try:
                    return int(optimo)
                except Exception:
                    return optimo
    return None

def graficosYDatos(nombreArch, m):
    selecciones=[seleccion]
    n, task_times,anterioridad=read_in2_data(nombreArch)
    
    plotGeneral1 = []
    for i in range (100):
        pob_init=poblacion_inicial(100,n,m,anterioridad,task_times,0.5)
        print(f"Prueba: {i+1}")
        mejor1,nombre1,tuplas1=genetic(seleccion,pob_init,1000,100,n,m,anterioridad,task_times,1)
        plotGeneral1.append((tuplas1,max(tiempo_grupo(mejor1, task_times)) if condiciones(mejor1, n, m, anterioridad)[0] else -1))
    guardar_plot_y_scores(plotGeneral1,nombre1,nombreArch,n,m)
def main():
    # print(buscar_valor_optimo("lutz1",9))



    graficosYDatos(r"..\precedenceGraphs\lutz1.in2",9)



    # n, task_times,anterioridad=read_in2_data(r"C:\Users\palom\Documents\GitHub\Proyecto-Modelizacion-algoritmos-evolutivos-SALBP\precedenceGraphs\lutz1.in2")
    # m=9
    # pob_init=poblacion_inicial(100,n,m,anterioridad,task_times,0.5)
    # genetic(seleccion,pob_init,1000,100,n,m,anterioridad,task_times,1)
if(__name__=='__main__'):
   main()
