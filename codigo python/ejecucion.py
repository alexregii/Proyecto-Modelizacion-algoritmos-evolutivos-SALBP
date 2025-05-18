from algo_propre import *
from itertools import islice
def n_topological_sorts(G, n):
    return list(islice(nx.algorithms.dag.all_topological_sorts(G), n))
def main():
    # #test problema 1
    # t1 = [10,15,20,12,18,25]
    # A1 = [[0,0,0,0,0,0],[1,0,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,1,0]]
    # n1 = 6
    # m1 = 3
    # dim_pob1 = 6

    # pob_init1 = poblacion_inicial(dim_pob1,n1,m1,A1,t1)
    # genetic(seleccionCruc,pob_init1,6,dim_pob1,n1,m1,A1,t1)
    # print("\n")
    # genetic(seleccionMut2,pob_init1,6,dim_pob1,n1,m1,A1,t1)
    # print("\n")
    # genetic(seleccionMut,pob_init1,6,dim_pob1,n1,m1,A1,t1)
    # print("\n")
    # genetic(seleccionCrucRueda,pob_init1,6,dim_pob1,n1,m1,A1,t1)


    #test problema 2
    t2 = [12,18,22,10,14,20,16,24,30]

    A2 = [[0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,0,1,0]]

    dim_pob2 = 100
    n2 = 9
    m2 = 3
    pob_init2 = poblacion_inicial(dim_pob2,n2,m2,A2,t2,0.5)
    genetic(seleccion_corto_rueda,pob_init2,5000,dim_pob2,n2,m2,A2,t2,1)

    # #test problema 3
    # t3 = [36,26,30,30,24,30,24,12,54,12]
    # A3 = [[0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[1,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0]]
    # n3 = 10
    # m3 = 3
    # dim_pob3 = 100
    # pob_init3 = poblacion_inicial(dim_pob3,n3,m3,A3,t3,0.5)
    # print(genetic(seleccion_corto_rueda,pob_init3,10000,dim_pob3,n3,m3,A3,t3,1))

    # t = tiempo_etapas(50,25,60)
    # A = construccion_matriz(50)

    # #test genetic - t y A aleatorio
    # dim_pob4 = 100
    # n4 = 50
    # m4 = 15
    # pob_init4 = poblacion_inicial(dim_pob4,n4,m4,A,t)
    # print(genetic(seleccionCruc,pob_init4,100,dim_pob4,n4,m4,A,t))
    # print("\n")
    # print(genetic(seleccionMut2,pob_init4,100,dim_pob4,n4,m4,A,t))
    # print("\n")
    # print(genetic(seleccionMut,pob_init4,100,dim_pob4,n4,m4,A,t))
    # print("\n")
    # print(genetic(seleccionCrucRueda,pob_init4,10,dim_pob4,n4,m4,A,t))

    # test genetic desde los datos
    
    # n, task_times, precedence_matrix = read_in2_data(".\precedenceGraphs\SCHOLL.IN2")  
    # # print("DATOS:\n"+"n: "+str(n)+"\ntasktimes: "+str(task_times)+"\nMatriz:"+str(precedence_matrix))  
    # anterioridad = precedence_matrix
    # m=39
    # # pob_init=poblacion_inicial(100,n,m,anterioridad,task_times,0.5)
    # # n,dato1=genetic(seleccionCruc2,pob_init,10000,100,n,m,anterioridad,task_times,1)
    # # plot_tuples([dato1])
    # print(list(n_topological_sorts(matriz_a_lista_adyacencia(anterioridad),10)))
    # #print(genetic2(100,100,n,9,anterioridad,task_times))

if(__name__=='__main__'):
   main()