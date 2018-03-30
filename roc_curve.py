#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys, getopt

def main(argv):

   clients_file = ''
   fake_clients_file = ''

   ## Default values
   fn = 0.8 # We find roc curve intercept when 'falses negative' is equal to fn
   fp = 0.8 # We find roc curve intercept when 'falses positive' is equal to fp

   try:
      opts, args = getopt.gnu_getopt(argv,"n:p:",["fn=","fp="])
      print(opts)
      print(args)
   except getopt.GetoptError:
      print('Exception: roc_curve.py <clients_file> <fake_clients_file>  -n <fn> -p <fp>')
      sys.exit(2)

   clients_file = args[0]
   fake_clients_file = args[1]

   for opt, arg in opts:
      if opt in ("-n", "--fn"):
         fn = float(arg)
      elif opt in ("-p", "--fp"):
         fp = float(arg)

   print('Archivo de clientes:', clients_file)
   print('Archivo de impostores: ', fake_clients_file)
   print('Se buscará FP cuando FN es igual a', str(fn))
   print('Se buscará FN cuando FP es igual a', str(fp))

   ## Get data
   data1 = np.genfromtxt(clients_file, delimiter=' ', dtype=None)
   data2 = np.genfromtxt(fake_clients_file, delimiter=' ', dtype=None)

   roc_values, area = getROCCurve(data1, data2)

   f1 = find_fn(np.array(roc_values), fp)
   f2 = find_fp(np.array(roc_values), fn)
   f3 = find_equal_fpfn(np.array(roc_values))
   #print(f1.shape)
   #print(f2.shape)

   print("\n")
   print("FN(FP=" + str(fp) + "): " + "{:.3f}".format(1.0 - f1[2]) + " con UMBRAL= " + "{:.3f}".format(f1[0]))
   print("FP(FN=" + str(fn) + "): " + "{:.3f}".format(f2[1]) + " con UMBRAL= " + "{:.3f}".format(f2[0]))
   print("FP = FN -> FP: " + "{:.3f}".format(f3[1]) + " FN: " + "{:.3f}".format(1.0 - f3[2]) +  " UMBRAL: " + "{:.3f}".format(f3[0]))
   #print("ROC Curve area: " + str(computeROCArea(roc_values)))
   print("ROC Curve area: " + "{:.3f}".format(area))
   computeDPrime(data1, data2)

   pltRocCurve(roc_values, [fp, f1[2]], [f2[1],1-fn], [f3[1], f3[2]])


def column(matrix, i):
    return [row[i] for row in matrix]

def find_equal_fpfn(roc_values):
    idx = (np.abs(roc_values[:,1] - (1-roc_values[:,2]))).argmin()
    return roc_values[idx,:]

def find_fp(roc_values, fn):
    #print(roc_values[:,2].shape)
    idx = (np.abs(roc_values[:,2] - (1-fn))).argmin()
    #print(idx)
    return roc_values[idx,:]

def find_fn(roc_values, fp):
    #print(roc_values[:, 1].shape)
    idx = (np.abs(roc_values[:,1] - fp)).argmin()
    #print(idx)
    return roc_values[idx,:]

def pltRocCurve(roc_values, p_fp, p_fn, fpfn):
    plt.xlabel('FP')
    plt.ylabel('1-FN')
    plt.title('Curva ROC')
    plt.axis([0, 1, 0, 1])
    plt.plot(column(roc_values, 1), column(roc_values, 2), 'r--')
    plt.plot(p_fp[0],p_fp[1],'gs')
    plt.plot(p_fn[0], p_fn[1], 'bs')
    plt.grid(True)
    plt.annotate('(' + "{:.3f}".format(p_fp[0]) + ', ' + "{:.3f}".format(p_fp[1])  + ' )', xy=(p_fp[0], p_fp[1]), xytext=(p_fp[0] + 0.03, p_fp[1] - 0.03) ,
                 arrowprops=dict(arrowstyle = '->', facecolor='black'),
                 )
    plt.annotate('(' + "{:.3f}".format(p_fn[0])  + ', ' + "{:.3f}".format(p_fn[1]) + ' )', xy=(p_fn[0], p_fn[1]), xytext=(p_fn[0] + 0.03, p_fn[1] - 0.03),
                 arrowprops=dict(arrowstyle='->', facecolor='black'),
                 )
    plt.annotate('FP=FN', xy=(fpfn[0], fpfn[1]),
                 xytext=(fpfn[0] + 0.05, fpfn[1] - 0.05),
                 arrowprops=dict(arrowstyle='->', facecolor='black'),
                 )
    plt.show()

def getROCCurve(data_clients, data_noclients):
    ## Get scores
    scoresClientes = np.array([x[1] for x in data_clients])
    n_clientes = len(scoresClientes)
    print("Cantidad de clientes: " + str(n_clientes))
    #print(scoresClientes.shape)
    #print(scoresClientes)

    scoresImpostores = np.array([x[1] for x in data_noclients])
    n_impostores = len(scoresImpostores)
    print("Cantidad de impostores: " + str(n_impostores))
    #print(scoresImpostores.shape)
    #print(scoresImpostores)

    area = computeROCArea2(scoresClientes, scoresImpostores)

    scores = []
    [scores.append((0, x)) for x in scoresClientes]
    [scores.append((1, x)) for x in scoresImpostores]
    dtype = [('ctype', int), ('score', float)]
    scores = np.sort(np.array(scores, dtype=dtype), order='score')

    print("Cantidad de scores totales: " + str(scores.shape[0]))
    #print(scores)

    ## Get threshold values
    threshold = np.unique(np.array([x[1] for x in scores]))

    #print(threshold.shape)
    #print(threshold)

    ##Get FP and FN
    roc_values = []
    for t in threshold:
        n_fp = 0
        n_fn = 0
        for cv in scores:
            type = cv[0]
            score = cv[1]
            # Cliente
            if type == 0 and score < t:
                n_fn += 1
            elif type == 1 and score > t:
                n_fp += 1

        fp = n_fp / n_impostores
        fn = n_fn / n_clientes
        roc_values.append([t,fp, 1.0 - fn])

    #np.savetxt("thresholds", threshold)
    #np.savetxt("results", scores, delimiter=",")
    #np.savetxt("roc_values", roc_values, delimiter=",")

    return roc_values, area

def computeROCArea2(score_c, score_i):
    talla_c = len(score_c)
    talla_i = len(score_i)

    factor = 1.0 / float(talla_c * talla_i)
    acc = 0.0

    for c in score_c:
        for i in score_i:
            acc += H_func(c, i)

    return (acc * factor)

def H_func(c,i):
    if c > i:
        return 1.0
    if c < i:
        return 0.0
    else:
        return 0.5

def computeROCArea(roc_values):
    acc_area = 0.0
    it = 0
    last_fp = 1.0
    for row in roc_values:
        fp = row[1]
        fn_1 = row[2]
        #print("fp: " + str(fp))
        #print("fn_1: " + str(fn_1))
        #print("last_fp: " + str(last_fp))
        carea = (last_fp - fp)*fn_1
        #print("area_acc: " + str(carea))
        acc_area += carea
        last_fp = fp
        it += 1

    return acc_area

def computeDPrime(data_clients, data_impostores):

    #Get scores
    scoresClientes = np.array([x[1] for x in data_clients])
    scoresImpostores = np.array([x[1] for x in data_impostores])

    #Compute means
    meanC = np.mean(scoresClientes)
    meanI = np.mean(scoresImpostores)
    varC = np.var(scoresClientes)
    varI = np.var(scoresImpostores)

    #print("Mean Clientes " + str(meanC))
    #print("Mean Impostores " + str(meanI))
    #print("Var Clientes " + str(varC))
    #print("Var Impostores " + str(varI))

    dprime = (meanC - meanI)/np.sqrt(varC+varI)

    print("D Prime: " + "{:.3f}".format(dprime))

if __name__ == "__main__":
   main(sys.argv[1:])




