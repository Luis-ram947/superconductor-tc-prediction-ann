# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:47:35 2025

@author: giorn
"""
import numpy as np
import pandas as pd
import Funciones_Superconductores as FSC 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import inverse_desing as ID #ChemDomain, build_domain_from_dataset
# Hola
# ===========================================================
CSV_PATH = "Data_base_supercond.csv"



# FUNCIÓN 1: ABS + NORMALIZACIÓN
# ===========================================================

Xn,yn,df = FSC.Normalizar(CSV_PATH)


# FUNCIÓN 2: ELIMINAR COMPOSICIONES REPETIDAS 
# ===========================================================
X,y = FSC.Eliminar_Comp_repetidas(Xn,yn)

FSC.histograma(y,"T(K)")
print(X.shape)
print(max(y))

# SPLIT DE LOS DATOS 
#============================================================
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)


        
# 4) Pipeline RF (imputación por seguridad + RF)
#=============================================================
rf = Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=4,
                    max_features=0.5,
                    bootstrap=True,
                    oob_score=True,        
                    n_jobs=-1,
                    random_state=20,
                    criterion="squared_error",
                ))
            ])
        
# ============================================================

# ENTRENAMIENTO Y R² TEST 
# ============================================================
rf.fit(X_train, y_train)
r2_test = rf.score(X_test, y_test)
print(f"R² (test): {r2_test:.4f}")

# Predicciones 
y_pred_train = rf.predict(X_train)
y_pred_test  = rf.predict(X_test)

#GRAFICOS DE PARIDAD Y RESIDUALES 
#=============================================================

# FSC.mostrar_resultados(y_train, y_pred_train, "train")
# FSC.mostrar_resultados(y_test,  y_pred_test,  "test")



# === NUEVO: filtrar y graficar solo los "buenos" ===
# 6) Filtrar “mejores” por tolerancia de error absoluto
tol = 4  # ajusta a tu criterio
X_new_tr, y_new_tr, _ = FSC.Filtro_mejores(X_train, y_train, y_pred_train, tol)
X_new_te, y_new_te, _ = FSC.Filtro_mejores(X_test,  y_test,  y_pred_test,  tol)

# 7) Unir y reentrenar desde cero
X_retrain = pd.concat([X_new_tr, X_new_te], axis=0).reset_index(drop=True)
y_retrain = pd.concat([y_new_tr, y_new_te], axis=0).reset_index(drop=True)

rf2 = clone(rf)               # nuevo modelo “en blanco”
rf2.fit(X_retrain, y_retrain) # reentrenado SOLO con “mejores”
score_r2=rf2.score(X_retrain,y_retrain)
print(f"R² (retrain): {score_r2:.4f}")

y_new_pred = rf2.predict(X_retrain)

# FSC.histograma(y_new_pred)
print(max(y_new_pred))

#FSC.mostrar_resultados(y_retrain, y_new_pred,"retrain")

print(type(X_train))
print(X_train.shape)
print(len(X_train.columns))
print(X_train.columns[:10])

domain = ID.build_domain_from_dataset(X_train,
                                    q=0.995,
                                    min_frac_active=0.015,
                                    max_active_quantile=0.95,
                                    k_nn=10)

print("N elementos:", len(domain.elements))
print("Ejemplo columnas:", domain.elements[:10])
print("max_active (aprendido):", domain.max_active)

# Para entender qmax (los máximos razonables)
qmax_series = pd.Series(domain.qmax, index=domain.elements).sort_values(ascending=False)
print("\nTop 15 qmax:")
print(qmax_series.head(15).round(3))

# Para entender cuántos activos suelen ocurrir
counts = (X_train.to_numpy() >= domain.min_frac_active).sum(axis=1)
print("\nConteo de activos (quantiles):", np.quantile(counts, [0.5, 0.75, 0.9, 0.95, 0.99]))

v = np.random.rand(85) * 2 - 0.5  # valores con negativos
x = ID.project_to_simplex(v)
x2 = ID.sparsify_and_renorm(x, 0.01)

print("sum x:", x.sum(), "min x:", x.min())
print("sum x2:", x2.sum(), "activos x2:", (x2 >= 0.01).sum())

Tc_target = 120.0
obj_dbg = ID.make_objective_rf_debug(rf2, domain, Tc_target)

x0 = np.random.rand(85)
J0, terms0, x_phys = obj_dbg(x0)

print("J0 =", J0)
print(terms0)

# ver los elementos más importantes de la composición
s = pd.Series(x_phys, index=domain.elements).sort_values(ascending=False)
print("\nTop elementos:")
print(s.head(10).round(4))



from scipy.optimize import differential_evolution

# -----------------------------
Tc_target = 120.0
obj = ID.make_objective_rf(
    rf_model=rf2,
    domain=domain,
    Tc_target=Tc_target,
    tol=5.0,
    w_T=1.0,
    w_qmax=20.0,
    w_active=2.0,
    w_nn=2.0
)

print("obj es:", obj)
print("callable(obj)?", callable(obj))
print("prueba obj(x):", obj(np.random.rand(len(domain.elements))))

# 4) OPTIMIZACIÓN INVERSA (⬅️ AQUÍ VA TU BLOQUE)
# -----------------------------------------------
bounds = [(0.0, 1.0)] * len(domain.elements)

res = differential_evolution(
    obj,
    bounds=bounds,
    maxiter=60,
    popsize=20,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=True,
    seed=42,
    disp=True,
    workers=1,          # <- fuerza serial
    updating="immediate" # <- recomendado cuando workers=1
)

# 5) Post-procesado del mejor candidato
# -------------------------------------
x_best = ID.project_to_simplex(res.x)
x_best = ID.sparsify_and_renorm(x_best, domain.min_frac_active)

Tc_best = float(
    rf2.predict(pd.DataFrame([x_best], columns=domain.elements))[0]
)

print("\nBest J:", res.fun)
print("Tc_hat:", Tc_best)
print("Activos:", int((x_best >= domain.min_frac_active).sum()))

top = pd.Series(x_best, index=domain.elements).sort_values(ascending=False)
print("\nTop 10 elementos:")
print(top.head(10).round(4))