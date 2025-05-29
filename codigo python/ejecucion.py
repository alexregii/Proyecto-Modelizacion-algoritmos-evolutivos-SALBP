from algo_propre import *
def main():

    n, task_times, precedence_matrix = read_in2_data(".\precedence Graphs\BUXEY.IN2")   
    anterioridad = precedence_matrix
    m=12
    p=find_best_p(n,m,500)[0]
    print(p)
    pob_init=poblacion_inicial(500,n,m,anterioridad,task_times,p)
    n,dato1=genetic(seleccion,pob_init,10000,500,n,m,anterioridad,task_times,p)
    plot_tuples([dato1])

if(__name__=='__main__'):
   main()
