import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors

# Construccion de la clase dominio 
@dataclass
class ChemDomain:
    elements: list
    qmax: np.ndarray
    min_frac_active: float
    max_active: int
    nn_model: NearestNeighbors
    nn_scale: float

def build_domain_from_dataset(
        X: pd.DataFrame,
        q: float = 0.995,
        min_frac_active: float = 0.01,
        max_active_quantile: float = 0.95,
        k_nn: int = 10
)-> ChemDomain:
    
    # Guadra el nombre de los elementos del data set

    elements = list(X.columns)

    # Maximo razonable por elementos
    qmax = X.quantile(q).to_numpy(dtype=float)
    qmax = np.clip(qmax, 0.0, 1.0)

    # Determinar cuantos elementos tipicos tiene un super cond

    X_arr = X.to_numpy(dtype=float)
    active_counts = (X_arr >= min_frac_active).sum(axis=1)
    max_active = int(np.quantile(active_counts, max_active_quantile))
    max_active = max(1, max_active)

    # Entrenar modelo de vecinos cercanos (solo mide distancias)

    nn = NearestNeighbors(n_neighbors=k_nn,metric="euclidean")
    nn.fit(X_arr)

    # Distancia tipica entre composiciones

    dists, _ =nn.kneighbors(X_arr, n_neighbors=k_nn)
    nn_scale = float(np.quantile(dists[:, -1],0.95) + 1e-12)

    return ChemDomain(
        elements=elements,
        qmax=qmax,
        min_frac_active=min_frac_active,
        max_active=max_active,
        nn_model=nn,
        nn_scale=nn_scale
    )

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Proyecta v al simplex: x>=0 y sum(x)=1.
    Esto hace que cualquier candidato de DE se vuelva una composición física válida.
    """
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)

    if v.sum() == 0.0:
        return np.ones_like(v) / len(v)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0]

    if len(rho) == 0:
        return v / v.sum()

    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

def sparsify_and_renorm(x: np.ndarray, min_frac_active: float = 0.01) -> np.ndarray:
    """
    Quita fracciones muy pequeñas (ruido) y renormaliza a suma 1.
    Esto fuerza soluciones con pocos elementos activos (lo que tu dataset muestra como normal).
    """
    x = np.asarray(x, dtype=float).copy()
    x[x < min_frac_active] = 0.0

    s = x.sum()
    if s <= 0:
        # si todo se fue a 0, regresamos a una proyección simple
        return project_to_simplex(x)

    return x / s



def make_objective_rf(
    rf_model,
    domain,
    Tc_target: float,
    tol: float = 5.0,
    w_T: float = 1.0,
    w_qmax: float = 20.0,
    w_active: float = 10.0,
    w_nn: float = 2.0,
    apply_sparsify: bool = True
):
    """
    Regresa una función objective(x_raw)->float para Differential Evolution.
    x_raw: vector (85,) en [0,1].
    """

    elements = domain.elements
    qmax = domain.qmax
    min_frac = domain.min_frac_active

    def objective(x_raw: np.ndarray) -> float:
        # 1) Composición válida
        x = project_to_simplex(x_raw)

        # 2) Quitar fracciones pequeñas (opcional) y renormalizar
        if apply_sparsify:
            x = sparsify_and_renorm(x, min_frac)

        # 3) DataFrame con columnas correctas
        X_df = pd.DataFrame([x], columns=elements)

        # 4) Predicción Tc
        Tc_hat = float(rf_model.predict(X_df)[0])

        # 5) Loss del target con tolerancia ±tol (zona muerta)
        err = abs(Tc_hat - Tc_target)
        L_T = max(0.0, err - tol)

        # 6) Penalización qmax
        over = np.maximum(0.0, x - qmax)
        P_qmax = float(np.sum(over**2))

        # 7) Penalización por demasiados elementos activos
        n_active = int(np.sum(x >= min_frac))
        P_active = float(max(0, n_active - domain.max_active)**2)

        # 8) Penalización por distancia al dataset (kNN)
        dists, _ = domain.nn_model.kneighbors(
            X_df.to_numpy(dtype=float),
            n_neighbors=domain.nn_model.n_neighbors
        )
        d = float(dists[0, -1] / domain.nn_scale)
        P_nn = d**2

        # 9) Objetivo total
        J = (w_T * L_T
             + w_qmax * P_qmax
             + w_active * P_active
             + w_nn * P_nn)

        return float(J)

    return objective

def make_objective_rf_debug(
    rf_model,
    domain,
    Tc_target: float,
    tol: float = 5.0,
    w_T: float = 1.0,
    w_qmax: float = 20.0,
    w_active: float = 10.0,
    w_nn: float = 2.0,
    apply_sparsify: bool = True
):
    elements = domain.elements
    qmax = domain.qmax
    min_frac = domain.min_frac_active

    def objective_with_terms(x_raw: np.ndarray):
        x = project_to_simplex(x_raw)
        if apply_sparsify:
            x = sparsify_and_renorm(x, min_frac)

        X_df = pd.DataFrame([x], columns=elements)
        Tc_hat = float(rf_model.predict(X_df)[0])

        err = abs(Tc_hat - Tc_target)
        L_T = max(0.0, err - tol)

        over = np.maximum(0.0, x - qmax)
        P_qmax = float(np.sum(over**2))

        n_active = int(np.sum(x >= min_frac))
        P_active = float(max(0, n_active - domain.max_active)**2)

        dists, _ = domain.nn_model.kneighbors(X_df.to_numpy(dtype=float), n_neighbors=domain.nn_model.n_neighbors)
        d = float(dists[0, -1] / domain.nn_scale)
        P_nn = d**2

        J = (w_T * L_T
             + w_qmax * P_qmax
             + w_active * P_active
             + w_nn * P_nn)

        terms = {
            "Tc_hat": Tc_hat,
            "L_T": L_T,
            "P_qmax": P_qmax,
            "n_active": n_active,
            "P_active": P_active,
            "P_nn": P_nn,
            "J": J
        }
        return J, terms, x

    return objective_with_terms



