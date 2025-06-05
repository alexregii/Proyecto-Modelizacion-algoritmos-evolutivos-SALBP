from funciones import *
def main():
    n, task_times, precedence_matrix = read_in2_data(".\precedence graphs\BUXEY.IN2")    
    anterioridad = precedence_matrix
    m=10
    p = find_best_p(n, m)[0]
    pob_init=poblacion_inicial(500,n,m,anterioridad,task_times,p)
    n,dato1=genetic(seleccion,pob_init,10000,500,n,m,anterioridad,task_times,1)
    plot_tuples([dato1])

if(__name__=='__main__'):
   main()
