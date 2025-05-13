# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:04:27 2025

@author: Paul Rohel
"""



# - Buscamos equilibrar los tiempos de cada de grupo de montaje (es decir la suma de los tiempos de las etapas que lo componen)
# - La decisión es la división de la línea en m grupos.
# - n etapas y m grupos $\to$ espacio de búsqueda : $\binom{n}{m}$.
# - Restricciones : anterioridad de las etapas.
# - Evaluación : $f(x)$ = suma de las diferencias de tiempo (valor absoluto) entre cada grupo $\to$ minimizar $f$ .
# (si no se cumples las condiciones entonces $f(x)$ = 1000000 (arbitrariamente grande)) \\

# \\

# Información :
#  - n etapas numeradas de 1 a n.
#  - tabla (n x 1) con los tiempos de cada etapa.
#  - matriz (n x n) con las etapas anterior a cada etapa (1 si directamente anterior, 0 sino)

# \\

# Genoma:
# - codifición posicional sobre m enteros (ej. con m = 3 $\to$ 1 | 1 | 3 | 2 | 2 | 3 | 3)
# - candidatos no factibles :
#   - no respetan la condición que, para cada etapa, las etapas anteriores a ella tienen pertenecer al mismo grupo o a otro anterior.
#   - uno de los grupos no aparece nunca en la codificación.
# - operación de mutación : un punto.

# Problema de endogamia

# pip install networkx
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd
import networkx as nx
from matplotlib.lines import Line2D
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from multiprocessing import Process
from bestp import *
# prompt: leer datos de un .IN2 que tiene estas caracteristicas:
# line 1: number n of tasks
# lines 2-n+1: integer task times
# lines n+2,...: direct precedence relations in form "i,j"
# last line: end mark "-1,-1" (optional)

def read_in2_data(filename):
    try:
        with open(filename, 'r',encoding='latin-1') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None

    n = int(lines[0].strip())
    task_times = [int(line.strip()) for line in lines[1:n+1]]
    precedence_relations = []
    for line in lines[n+1:]:
        if line.strip() == "-1,-1":
            break
        i, j = map(int, line.strip().split(','))
        precedence_relations.append((i, j))
    
    return n, task_times, matriz_from_rel(n,precedence_relations)

# def read_alb_file(filename):
    try:
        with open(filename, 'r', encoding='latin-1') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None

    idx = 0
    if "<number of tasks>" in lines[idx]:
        idx += 1
        n = int(lines[idx])
        idx += 1
    else:
        raise ValueError("Format incorrect : <number of tasks> manquant")
    if "<cycle time>" in lines[idx]:
        idx += 1
        cycle_time = int(lines[idx])
        idx += 1
    else:
        cycle_time = None
    if "<order strength>" in lines[idx]:
        idx += 1
        order_strength = float(lines[idx])
        idx += 1
    else:
        order_strength = None
    if "<task times>" in lines[idx]:
        idx += 1
        task_times = [0] * n
        while idx < len(lines) and not lines[idx].startswith("<"):
            task_id, time = map(int, lines[idx].split())
            task_times[task_id - 1] = time
            idx += 1
    else:
        raise ValueError("Format incorrect : <task times> manquant")
    precedence_matrix = [[0 for _ in range(n)] for _ in range(n)]
    if "<precedence relations>" in lines[idx]:
        idx += 1
        while idx < len(lines) and not lines[idx].startswith("<end>"):
            i, j = map(int, lines[idx].split(','))
            precedence_matrix[i - 1][j - 1] = 1
            idx += 1
    return n, task_times, precedence_matrix

def matriz_from_rel(n, relations):
    matriz = [[0] * n for _ in range(n)]
    for i, j in relations:
        matriz[i-1][j-1] = 1
    return matriz

# Función para calcular el tiempo total por grupo
def tiempo_grupo(sec, tiempos):
    suma = defaultdict(int)
    for i in range(len(sec)):
        suma[sec[i]] += tiempos[i]
    if not sec:
        return []
    max_val = max(sec)
    return [suma.get(i, 0) for i in range(max_val + 1)]

#verificar si el individuo cumple las condiciones de anterioridad y la división de la línea en exactamente m grupos
def condiciones(sec, n, m, anterioridad):
    # Validar que sec contiene todos los valores de 0 a m-1
    if sorted(set(sec)) != list(range(m)):
      return False,1e7
    # Comprobación de anterioridad
    bool=True
    n_fallos=0
    for k in range(len(sec)):
        for i in range(len(sec)):
            if anterioridad[k][i] == 1 and sec[k] < sec[i]:
                bool=False
                n_fallos += 1
    return bool, n_fallos

def dispersion(t):
    return max(t) - min(t)
def penal_sobrecarga(t):
   media = sum(t) / len(t)
   res=sum(max(0, load - media) for load in t)
   return res/len(t)
    
def score(sec, n, m, anterioridad, tiempos):
    bool, n_fallos = condiciones(sec, n, m, anterioridad)
    t = tiempo_grupo(sec, tiempos)
    media = sum(t) / len(t)
    return 0.6*max(t) +0.2*penal_sobrecarga(t)+ 0.2*dispersion(t) + (media/10)*n_fallos

#clasificar los individuos de una población del más al menos equilibrado
def clasificacion(poblacion,n,m,anterioridad,tiempos):
  poblacion_clasificada = sorted(poblacion, key=lambda sec: score(sec, n, m,anterioridad,tiempos))
  return poblacion_clasificada

#mutación de cada individuo de la población, cambiando aleatoriamente un bit
def mutacion(poblacion,n,m):
  for i in range(0,len(poblacion)):
    a =  rd.randint(0,n-1)
    b =  rd.randint(poblacion[i][a]-1,poblacion[i][a]+1)
    poblacion[i][a] = b
  return poblacion

def mutacion2(poblacion,n,m):
  for a in range(len(poblacion)):
    for j in range(len(poblacion[a])):
        fac = rd.randint(-m//2,m//2)
        cambio = rd.choices(population=[fac*1, 0], weights=[0.2,0.8], k=1)[0]
        poblacion[a][j] += cambio
  return poblacion

def seleccionCruc2(poblacion,n,m,anterioridad,tiempos,p):
    poblacion = clasificacion(poblacion,n,m,anterioridad,tiempos)
    ord_top = matriz_a_lista_adyacencia(anterioridad)
    poblacion = cruce(poblacion,ord_top)
    mitad = len(poblacion) // 2
    poblacion = clasificacion(poblacion,n,m,anterioridad,tiempos)
    poblacion[mitad:] = poblacion_inicial(len(poblacion)-mitad,n,m,anterioridad,tiempos,p)
    return poblacion


def seleccion_corto(poblacion, n, m, anterioridad, tiempos,p):
    tercio = len(poblacion) // 3
    poblacion[0] = clasificacion(copy.deepcopy(poblacion), n, m, anterioridad, tiempos)[0]
    poblacion[1:] = class_torneo(poblacion[1:],n,m,anterioridad,tiempos,2)
    ord_top = matriz_a_lista_adyacencia(anterioridad)
    poblacion[:tercio] = cruce(poblacion[:tercio], ord_top)
    for i in range(tercio, 2 * tercio):
        poblacion[i] = copy.deepcopy(poblacion[0])
    poblacion[tercio:2*tercio] = mutacion2(poblacion[tercio:2*tercio],n,m)

    poblacion[2 * tercio:] = poblacion_inicial(len(poblacion) - 2 * tercio, n, m, anterioridad, tiempos,p)

    for i in range(len(poblacion)):
        poblacion[1:] = mutacion(copy.deepcopy(poblacion[1:]), n, m)
    
    return poblacion

def seleccion_corto_rueda(poblacion, n, m, anterioridad, tiempos,p):
    tercio = len(poblacion) // 3
    poblacion[0] = clasificacion(copy.deepcopy(poblacion), n, m, anterioridad, tiempos)[0]
    poblacion[1:] = class_rueda(poblacion[1:],n,m,anterioridad,tiempos)
    ord_top = matriz_a_lista_adyacencia(anterioridad)
    poblacion[:tercio] = cruce(poblacion[:tercio], ord_top)
    # for i in range(tercio,2*tercio):
    #     poblacion[i]= copy.deepcopy(poblacion[0])
    poblacion[2 * tercio:] = poblacion_inicial(len(poblacion) - 2 * tercio, n, m, anterioridad, tiempos,p)

    poblacion[1:] = mutacion(copy.deepcopy(poblacion[1:]), n, m)
        
    return poblacion
    

def class_rueda(poblacion, n, m, anterioridad, tiempos):
    scores = [score(ind, n, m, anterioridad, tiempos) for ind in poblacion]
    max_score = max(scores)
    weights = [np.sqrt((max_score - s)) + 1e-6 for s in scores]
    selected_indices = rd.choices(range(len(poblacion)), weights=weights, k=len(poblacion))
    return [poblacion[i] for i in selected_indices]

def class_torneo(poblacion, n, m, anterioridad, tiempos, tam):
    nueva_poblacion = []
    for _ in range(len(poblacion)):
        competidores = rd.sample(poblacion[1:], tam)
        meilleur = min(competidores, key=lambda ind: score(ind, n, m, anterioridad, tiempos))
        nueva_poblacion.append(meilleur)
    return nueva_poblacion



#EN CASO DE GENERACION ALEATORIA DE LA POBLACION INCIAL : verificar que la matriz incial cumple las condiciones del grafo
#(sin etapa aislada del resto,sin cruce, sin vuelta hacia atrás)
def cond_matriz(matriz):
    matriz_np = np.array(matriz)
    G = nx.from_numpy_array(matriz_np, create_using=nx.DiGraph())
    if not nx.is_connected(G.to_undirected()):
        return False
    if any(G.has_edge(n, n) for n in G.nodes()):
        return False
    if not nx.is_directed_acyclic_graph(G):
        return False
    is_planar, _ = nx.check_planarity(G)
    if not is_planar:
        return False
    for i in range(len(matriz)):
        for j in range(i + 1, len(matriz)):
            if matriz[i][j] > 1 or matriz[j][i] > 1:
                return False
    return True

#Cruce2a2
def cruce2a2(ind1,ind2,a,b,ord_top):
  nvind1 = ind1.copy()
  nvind2 = ind2.copy()
  result=[]
  eleccion=rd.choices([1,2,3],weights=[0.2,0.6,0.2],k=1)[0]
  if(eleccion==1):
    for i in range(a,b):
            nvind1[ord_top[i]]=ind2[ord_top[i]]
            nvind2[ord_top[i]]=ind1[ord_top[i]]
  elif(eleccion==2):
    for i in range(b,len(ord_top)):
            nvind1[ord_top[i]]=ind2[ord_top[i]]
            nvind2[ord_top[i]]=ind1[ord_top[i]]
  else:
     for i in range(0,a):
            nvind1[ord_top[i]]=ind2[ord_top[i]]
            nvind2[ord_top[i]]=ind1[ord_top[i]]
  result.append(nvind1)
  result.append(nvind2)
  return result

#Cruce general
#poblacion = total de la poblacion / 2

def cruce(mejores, ord_top):
    resultado = []
    resultado.append(mejores[0])  # Garde le meilleur

    num_cruces = (len(mejores) - 1) // 2  # Nombre de croisements à faire

    for _ in range(num_cruces):
        # Tirer 2 parents distincts au hasard
        i, j = rd.sample(range(len(mejores)), 2)
        a = rd.randint(0, len(ord_top) // 2)
        b = rd.randint(a, len(ord_top))

        hijos = cruce2a2(mejores[i], mejores[j], a, b, ord_top)
        resultado.extend(hijos[:2])  # Ajoute les deux enfants

    return resultado

# prompt:  quiero hacer un grafico con eje x siendo el primer elemento de una dupla y el y el segundo

# Assuming your data is in a list of tuples called 'data'
# Example: data = [(1, 2), (3, 4), (5, 6)]
def plot_tuples(data):
    tuplas = [item[1] for item in data]
    cmap = cm.get_cmap('tab10')
    for i in range(len(tuplas)):
        x_values=[tupla[0] for tupla in tuplas[i]]
        y_values=[tupla[1] for tupla in tuplas[i]]
        plt.plot(x_values, y_values, marker='o', linestyle='-', color=cmap(i % cmap.N))
    
    plt.legend()
    plt.xlabel("X-axis (Numero de generaciones)")
    plt.ylabel("Y-axis (Valor de la peor estacion)")
    plt.title("Comparacion entre los distintos")
    plt.grid(True)
    plt.show()



#EN CASO DE GENERACION ALEATORIA DE LA POBLACION INCIAL : construir la matriz que representa las relaciones de anterioridad entre las etapas
def construccion_matriz(n):
  matriz = [[0] * n for _ in range(n)]
  while cond_matriz(matriz) == False:
    for i in range(1, n):
      for j in range(i-3,i):
        if all(matriz[k][j] == 0 for k in range(j)):
          matriz[i][j] = rd.choice([0, 1])
  return matriz

##EN CASO DE GENERACION ALEATORIA DE LA POBLACION INCIAL : construir la tabla con los tiempos correspondientes a cada etapa
def tiempo_etapas(n,minT,maxT):
  tiempos = []
  for i in range(0,n):
    tiempos.append(rd.randint(minT,maxT))
  return tiempos

#POBLACION INICIAL COMPLETAMENTE ALEATORIA
def poblacion_iniciala(dim_pob,n,m,anterioridad,tiempos):
  poblacion = []
  sec = [0]*n
  for i in range(dim_pob):
    for j in range(n):
        sec[j]= rd.randint(0,m-1)
    
    poblacion.append(sec.copy()) 
    sec = [0]*n
    
  return poblacion

def poblacion_inicial(dim_pob,n,m,anterioridad,tiempos,p):
  poblacion = []
  ord_top = matriz_a_lista_adyacencia(anterioridad)
  for i in range(dim_pob//2):
    sec = [0]*n
    l = 1 # probabilidad
    for j in range(len(ord_top)):
            if l==1:
                if j == 0:
                    sec[ord_top[j]] = m - 1  # primera tarea a última estación
                else:
                    prev_est = sec[ord_top[j - 1]]
                    if prev_est - 1 >= 0:
                        opciones = [prev_est - 1, prev_est,prev_est+1]
                    else:
                        opciones = [0, 0, 1]
                    elegido=rd.choices(opciones, weights=[p,0.99-p,0.01], k=1)[0]
                    sec[ord_top[j]] = elegido
    poblacion.append(sec.copy()) 

  for i in range(dim_pob//2,dim_pob):
    for j in range(n):
        sec[j]= rd.randint(0,m-1)
            
    poblacion.append(sec.copy()) 
    sec = [0]*n
  return poblacion



def grafo(matriz, sec, tiempo, score, m,fig_id):
    plt.figure(fig_id)
    plt.clf()  # Limpiar figura actual
    matriz_np = np.array(matriz)
    tiempos = tiempo_grupo(sec, tiempo)
    num_grupos = max(sec) + 1
    cmap = plt.get_cmap("gist_rainbow", num_grupos)
    colores = [mcolors.to_hex(cmap(i)) for i in range(num_grupos)]
    nudos_color = [colores[g] for g in sec]

    G = nx.from_numpy_array(matriz_np)
    group_times = {color: 0 for color in colores}

    for j, group in enumerate(sec):
        color = colores[group]
        group_times[color] = tiempos[group]

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color=nudos_color,
             node_size=4800//len(matriz), edge_color='gray', font_weight='bold',
             font_size=10 + (4800//len(matriz))/500)

    used_colors = sorted(set(nudos_color), key=lambda c: colores.index(c))
    legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
                     for color in used_colors]
    time_legend = [f"{color}: {group_times[color]}" for color in used_colors]

    plt.legend(legend_labels, time_legend,
               title="Dispersión: " + str(dispersion(tiempos)) +
                     "\nTiempo máximo: " + str(max(tiempos))+
                     "\nValido: " + str(condiciones(sec, 0, m, matriz)[0])+", Nº de fallos:"+str(condiciones(sec, 0, m, matriz)[1]),
               loc="upper left",fontsize=10)
    plt.title("Mejor solución actual",loc="center")
    plt.pause(0.01)  # Mostrar sin bloquear

def genetic(tipo_seleccion, pob_init, no_gen, dim_pob, n, m, anterioridad, tiempos,fig_id):
    plt.ion()  # Activar modo interactivo
    p=find_best_p(n,m,500)[0]
    k = 0
    poblacion = pob_init
    tuplas = []
    best = poblacion[0]
    endo = True
    max_score_prec = 10000
    num_sol = math.comb(n, m)
    delay = 100000
    for i in range(no_gen):
        if poblacion[0] != best or i == no_gen - 1 or i % 100 == 0:
            print("GENERACION:", i)
            print("PUNTUACION: "+str(score(poblacion[0], n, m, anterioridad, tiempos )))
            print("Tiempo maximo:", max(tiempo_grupo(poblacion[0], tiempos)))
            tuplas.append((i, max(tiempo_grupo(poblacion[0], tiempos))))
            best = poblacion[0]
            grafo(anterioridad, poblacion[0], tiempos, score(poblacion[0], n, m, anterioridad, tiempos), m,fig_id)
        
        
        k += dim_pob
        # if i % delay == 0 and i > 0:     
        #     if score(poblacion[0], n, m, anterioridad, tiempos) == max_score_prec:
        #         for i in range(len(poblacion)):
        #             poblacion[1:] = mutacion(poblacion[1:],n,m)
        #         print(k)
        #         endo = False    
        #     max_score_prec = score(poblacion[0], n, m, anterioridad, tiempos)
        poblacion = tipo_seleccion(poblacion, n, m, anterioridad, tiempos,p)
        endo = True
            

    plt.ioff()  # Desactivar modo interactivo
    plt.figure(fig_id)  # Asegura mostrar la figura correcta
    plt.show()
    print("Valor del resultado 1:", score(poblacion[0], n, m, anterioridad, tiempos))
    print("Número de soluciones posibles:", num_sol)
    print("Número de soluciones calculadas:", k)
    return poblacion[0], (tipo_seleccion.__name__, tuplas)

#Matriz a lista de adyaciencia
def matriz_a_lista_adyacencia(matriz):
  G = nx.DiGraph()
    # Añadir aristas a partir de la matriz
  for i in range(len(matriz)):
      for j in range(len(matriz[i])):
          if matriz[i][j] == 1:
              G.add_edge(i, j)
  return  list(nx.topological_sort(G))
