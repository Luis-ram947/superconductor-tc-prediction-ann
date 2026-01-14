# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:06:23 2025

@author: giorn
"""

#Funciones para el Pretratamiento de los Superconductores 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    # gráficos


def Normalizar (csv_name,skip_zero=True):
    
    """
    se puede cambiar para trabajar con X y y desep el principio
    """
    
    """
    Esta funcion recive un data frame de pandas en donde las n-1 columnas son las entradas 
    y la columna n son las salidas 
    """
    #Leer el documento con los datos 
    df = pd.read_csv(csv_name, index_col=0)
    R,C=df.shape #Dimensiones de la matriz
    
    X = df.iloc[:,0:-1].abs().fillna(0.0)   # Entradas 
    y = df.iloc[:,-1]     # Salidas 
    
    sums = X.sum(axis=1) #axis = 1 suma las columnas 
    if skip_zero :
        mask = sums >0 # True si sums es mayor a cero y false si sums es cero o menor
        X = X[mask]
        y = y[mask]
        sums = sums[mask]
    Xn = X.div(sums,axis=0) # axis = 0 
    return Xn,y,df

    

def mostrar_resultados(y_true, y_pred, titulo="Conjunto"):
    """
    Esta funcion recibe los datos reales y las predicciones y grafica
    1- la grafica de paridad 
    2- la frecuencia de los residulales 
    """

    resid = y_pred - y_true

    tabla = pd.DataFrame({
        "Tc_real":  y_true,
        "Tc_pred":  y_pred,
        "Error":    resid
    })


    print(f"\n=== {titulo}: primeros 15 ===")
    print(tabla.head(15).round(3))
    
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(5.2,5.2))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot(lims, lims, '--', lw=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Tc real (K)"); plt.ylabel("Tc predicho (K)")
    plt.title(f"Paridad — {titulo}")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(5.8,3.8))
    plt.hist(resid, bins=50, alpha=0.9)
    plt.xlabel("Residual (K)"); plt.ylabel("Frecuencia")
    plt.title(f"Residuales — {titulo}")
    plt.tight_layout(); plt.show()
    
def Eliminar_Comp_repetidas (X,y,keep="first"):
    # hago una copia para no trabajar con los datos originales
    df = X.copy()
    # Lo combiente en una serie si no es asi 
    df["__y__"] = y.values if isinstance(y, pd.Series) else y
    df = df.round(8).drop_duplicates(keep=keep)   # redondea y elimina Renglones duplicados
    X_c = df.drop(columns="__y__") #Elimina la columna y
    y_c = df["__y__"].reset_index(drop=True) #Solo contiene la columna y
    return X_c,y_c
    
    
def Filtro_mejores (X,y_real,y_pred,tol,return_pandas=True):
    # Convertir a arrays para calcular la máscara por posición
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_series = isinstance(y_real, pd.Series)

    X_arr = X.values if X_is_df else np.asarray(X)
    y_arr = y_real.values if y_is_series else np.asarray(y_real)
    ypred_arr = np.asarray(y_pred)

    # Verificaciones rápidas
    if ypred_arr.shape[0] != X_arr.shape[0] or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("Dimensiones no coinciden entre X, y_real y y_pred.")

    # Máscara por posición
    mask = np.abs(y_arr - ypred_arr) <= tol

    if return_pandas and X_is_df:
        X_best = X.iloc[mask]
        y_best = (y_real.iloc[mask] if y_is_series else pd.Series(y_arr[mask], index=X_best.index, name="y"))
        y_best_p = pd.Series(ypred_arr[mask], index=X_best.index, name="y_pred")
    else:
        X_best = X_arr[mask]
        y_best = y_arr[mask]
        y_best_p = ypred_arr[mask]

    return X_best, y_best, y_best_p
    
    
    