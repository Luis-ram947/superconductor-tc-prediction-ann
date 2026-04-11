import random
import numpy as np
import pandas as pd
import time
import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Chemical_domain import chemical_domain as CD
from deap import base, creator, tools
from GA.ga_fitness import evaluate
from GA.ga_population import create_individual



def mut_mass_transfer(individual):
    x = list(individual)

    activos = [i for i in range(len(x)) if x[i] > 0]

    if len(activos) < 1:
        return individual,

    idx_add = random.randint(0, len(x) - 1)
    idx_sub = random.choice(activos)

    while idx_add == idx_sub:
        idx_add = random.randint(0, len(x) - 1)

    delta_max = min(0.05, x[idx_sub])
    delta = random.uniform(0.0, delta_max)

    x[idx_add] += delta
    x[idx_sub] -= delta

    if x[idx_sub] < 0:
        x[idx_sub] = 0.0

    for i in range(len(x)):
        individual[i] = x[i]

    return individual,


def build_toolbox():
    
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mut_mass_transfer)

    toolbox.register("clone", copy.deepcopy)

    return toolbox


def individual_to_dict(ind, rank=None, run_id=None, seed=None):
    ind_rep = CD.repair(list(ind))
    fit = evaluate(ind_rep)[0]

    neutral, elementos, fracciones, valencias, carga = CD.charge_neutrality(ind_rep)

    comp_dict = {e: float(f) for e, f in zip(elementos, fracciones)}

    return {
        "run_id": run_id,
        "seed": seed,
        "rank": rank,
        "fitness": float(fit),
        "charge_total": float(carga),
        "n_active": len(elementos),
        "sum_frac": float(sum(ind_rep)),
        "feasible": bool(CD.is_feasible_composition(ind_rep)),
        "composition": str(comp_dict)
    }


def run_ga(pop_size=30, ngen=50, cxpb=0.5, mutpb=0.5, seed=42, run_id=1, verbose=True):
    random.seed(seed)
    np.random.seed(seed)

    toolbox = build_toolbox()

    population = toolbox.population(n=pop_size)

    historial = []
    tiempo_acumulado = 0.0

    t0_total = time.perf_counter()

    # evaluar población inicial
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(ngen):
        t0_gen = time.perf_counter()

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # cruza
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)

                child1[:] = CD.repair(list(child1))
                child2[:] = CD.repair(list(child2))

                del child1.fitness.values
                del child2.fitness.values

        # mutación
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                mutant[:] = CD.repair(list(mutant))
                del mutant.fitness.values

        # reevaluar inválidos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fitness_vals = np.array([ind.fitness.values[0] for ind in population])
        best = tools.selBest(population, 1)[0]
        best_fit = best.fitness.values[0]

        t1_gen = time.perf_counter()
        tiempo_gen = t1_gen - t0_gen
        tiempo_acumulado += tiempo_gen

        historial.append({
            "run_id": run_id,
            "seed": seed,
            "generation": gen + 1,
            "best_fitness": float(np.min(fitness_vals)),
            "mean_fitness": float(np.mean(fitness_vals)),
            "std_fitness": float(np.std(fitness_vals)),
            "worst_fitness": float(np.max(fitness_vals)),
            "time_gen_s": tiempo_gen,
            "time_cumulative_s": tiempo_acumulado
        })

        if verbose:
            print(f"Generacion {gen+1}: mejor fitness = {best_fit:.6f} | tiempo = {tiempo_gen:.3f} s")

    t1_total = time.perf_counter()
    tiempo_total = t1_total - t0_total

    top10 = tools.selBest(population, 10)

    resultados_top = []
    for k, ind in enumerate(top10, start=1):
        resultados_top.append(individual_to_dict(ind, rank=k, run_id=run_id, seed=seed))

    resumen = {
        "run_id": run_id,
        "seed": seed,
        "pop_size": pop_size,
        "ngen": ngen,
        "cxpb": cxpb,
        "mutpb": mutpb,
        "best_fitness_final": float(top10[0].fitness.values[0]),
        "time_total_s": tiempo_total
    }

    df_hist = pd.DataFrame(historial)
    df_top = pd.DataFrame(resultados_top)
    df_resumen = pd.DataFrame([resumen])

    return population, df_hist, df_top, df_resumen



# df_hist.to_csv("historial_ga.csv", index=False)
# df_top.to_csv("top10_ga.csv", index=False)
# df_resumen.to_csv("resumen_ga.csv", index=False)
# import random
# import sys
# import os
# import numpy as np



# random.seed(SEED)
# np.random.seed(SEED)
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from Chemical_domain import chemical_domain as CD

# from deap import base, creator, tools

# from ga_fitness import evaluate
# from ga_population import create_individual


# # Parametros del GA 
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)

# POP_SIZE = 30
# NGEN = 50
# CXPB = 0.5
# MUTPB = 0.5


# import random

# def mut_mass_transfer(individual):

#     x = list(individual)

#     # posiciones con fraccion positiva
#     activos = []
#     for i in range(len(x)):
#         if x[i] > 0:
#             activos.append(i)

#     # si no hay suficientes activos, no hacer nada
#     if len(activos) < 1:
#         return individual,

#     # elegir un receptor cualquiera
#     idx_add = random.randint(0, len(x)-1)

#     # elegir un donador que tenga masa
#     idx_sub = random.choice(activos)

#     # evitar que sea el mismo indice
#     while idx_add == idx_sub:
#         idx_add = random.randint(0, len(x)-1)

#     # cantidad a mover
#     delta_max = min(0.05, x[idx_sub])
#     delta = random.uniform(0.0, delta_max)

#     # transferir masa
#     x[idx_add] = x[idx_add] + delta
#     x[idx_sub] = x[idx_sub] - delta

#     # evitar errores numericos
#     if x[idx_sub] < 0:
#         x[idx_sub] = 0.0

#     # copiar de regreso al individuo
#     for i in range(len(x)):
#         individual[i] = x[i]

#     return individual,

# # crar fitness y el individuo

# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)


# toolbox = base.Toolbox()

# #Registrar individuo y población

# toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# # Registrar evaluación

# toolbox.register("evaluate", evaluate)

# # Registrar selección, cruza y mutación
# toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", mut_mass_transfer)
# #toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.06, indpb=0.2)

# #Crear población inicial
# population = toolbox.population(n=POP_SIZE)

# #Evaluar población inicial

# fitnesses = list(map(toolbox.evaluate, population))

# for ind, fit in zip(population, fitnesses):
#     ind.fitness.values = fit

# # Loop  de GA

# for gen in range(NGEN):

#     offspring = toolbox.select(population, len(population))
#     offspring = list(map(toolbox.clone, offspring))

#     for child1, child2 in zip(offspring[::2], offspring[1::2]):
#         if random.random() < CXPB:
#             toolbox.mate(child1, child2)
#             child1[:] = CD.repair(list(child1))
#             child2[:] = CD.repair(list(child2))

#             del child1.fitness.values
#             del child2.fitness.values
#     for mutant in offspring:
#         if random.random() < MUTPB:
#             toolbox.mutate(mutant)
#             mutant[:] = CD.repair(list(mutant))

#             del mutant.fitness.values

#     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#     fitnesses = map(toolbox.evaluate, invalid_ind)

#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit

#     population[:] = offspring

#     best = tools.selBest(population, 1)[0]
#     print(f"Generacion {gen+1}: mejor fitness = {best.fitness.values[0]:.4f}")

# # Mejor individuo

# top10 = tools.selBest(population, 10)

# print("\nTop 10 individuos encontrados:\n")

# for k, ind in enumerate(top10, start=1):

#     ind_rep = CD.repair(list(ind))
#     fit = evaluate(ind_rep)[0]

#     neutral, elementos, fracciones, valencias, carga = CD.charge_neutrality(ind_rep)

#     print(f"--- Individuo {k} ---")
#     print("Fitness:", fit)
#     print("Carga total:", carga)
#     print("Numero de activos:", len(elementos))
#     print("Suma:", sum(ind_rep))
#     print("Factible:", CD.is_feasible_composition(ind_rep))

#     print("Elementos activos:")
#     for e, f, v in zip(elementos, fracciones, valencias):
#         print(f"  {e}: fraccion={f:.6f}, valencia={v}")

#     print()