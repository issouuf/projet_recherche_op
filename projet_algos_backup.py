import numpy as np
import matplotlib.pyplot as plt
from pulp import *
import random
import math
from ipywidgets import IntProgress
from IPython.display import display
from numba import njit
import gc
import time 
import os 
import concurrent.futures
import uuid

# =============================================================================
# 1. GÉNÉRATION D'INSTANCE GARANTIE FAISABLE (TSPTW-PC)
# =============================================================================

def genere_instance_complexe(n, precedence_prob=0.1):
    # 1. Génération de base
    mat_temps = np.random.randint(10, 45, size=(n+1, n+1)).astype(float)
    np.fill_diagonal(mat_temps, 0)
    
    # =========================================================
    # 2. LE CORRECTIF : Algorithme de Floyd-Warshall
    # Transforme le graphe aléatoire en vrai réseau routier.
    # Garantit que le trajet direct A->B n'est jamais plus lent 
    # qu'un détour par C. 
    # =========================================================
    for k in range(n+1):
        for i in range(n+1):
            for j in range(n+1):
                if mat_temps[i][k] + mat_temps[k][j] < mat_temps[i][j]:
                    mat_temps[i][j] = mat_temps[i][k] + mat_temps[k][j]
                    
    # On repasse en entier pour la propreté
    mat_temps = mat_temps.astype(np.int64)

    # 3. Création du chemin secret GARANTI FAISABLE avec les vraies distances
    chemin_secret = list(range(1, n+1))
    random.shuffle(chemin_secret)
    
    e = np.zeros(n+1)
    l = np.full(n+1, 999999.0) 
    s = np.random.randint(5, 15, size=n+1)
    s[0] = 0
    
    t_actuel = 300.0 # Départ à 5h00
    noeud_prec = 0
    
    for noeud in chemin_secret:
        t_actuel += mat_temps[noeud_prec][noeud]
        
        heure_locale = t_actuel % 1440
        if heure_locale > 1320: 
            jours = t_actuel // 1440
            t_actuel = (jours + 1) * 1440 + 300 
            
        jour_actuel = t_actuel // 1440
        debut_journee = jour_actuel * 1440 + 300
        
        # Fenêtres de temps générées SUR LES VRAIES DISTANCES
        e[noeud] = max(debut_journee, t_actuel - random.randint(10, 45))
        l[noeud] = t_actuel + random.randint(45, 120)
        
        t_actuel += s[noeud]
        noeud_prec = noeud
        
    P = []
    for idx, i in enumerate(chemin_secret):
        for j in chemin_secret[idx+1:]:
            if random.random() < precedence_prob:
                P.append((i, j))
                
    return mat_temps, e, l, s, P

def formater_temps(total_mins):
    """Affiche un format Heure propre, ou signale une tournée invalide"""
    if np.isnan(total_mins):
        return "[Non trouvé / Invalide]"
    if total_mins > 2000: # Si la pénalité a explosé le compteur
        return f"[INVALIDE - Score pénalité : {int(total_mins)}]"
        
    heures = int(total_mins // 60)
    mins = int(total_mins % 60)
    return f"{heures}h{mins:02d}"


def genere_instance_pure_aleatoire(n, num_precedences=2):
    # 1. Coordonnées & Distances Euclidiennes
    coords = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(n + 1)}
    mat_temps = np.zeros((n+1, n+1), dtype=np.float64)
    for i in range(n+1):
        for j in range(n+1):
            if i != j: mat_temps[i][j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                
    # 2. Floyd-Warshall (Inégalité triangulaire)
    for k in range(n+1):
        for i in range(n+1):
            for j in range(n+1):
                if mat_temps[i][k] + mat_temps[k][j] < mat_temps[i][j]:
                    mat_temps[i][j] = mat_temps[i][k] + mat_temps[k][j]
    mat_temps = mat_temps.astype(np.int64)

    # 3. Temps de service
    s = np.random.randint(5, 15, size=n+1)
    s[0] = 0
    
    # 4. Fenêtres Périodiques (Heures d'ouverture CHAQUE JOUR)
    e = np.zeros(n+1)
    l = np.full(n+1, 1320.0) # Par défaut, ferme à 22h
    
    for i in range(1, n+1):
        # Ouverture entre 5h (300) et 16h40 (1000)
        e[i] = random.randint(300, 1000) 
        # Reste ouvert entre 1h et 5h (sans dépasser 22h)
        l[i] = min(1320, e[i] + random.randint(60, 300)) 

    # 5. Précédences Aléatoires
    P = []
    villes_dispo = list(range(1, n + 1))
    random.shuffle(villes_dispo)
    for _ in range(min(num_precedences, len(villes_dispo)//2)):
        i = villes_dispo.pop()
        j = villes_dispo.pop()
        P.append((i, j))

    return mat_temps, e, l, s, P

# =============================================================================
# 2. LOGIQUE TEMPORELLE (5h-22h)
# =============================================================================

@njit
def calcul_temps_trajet_reel(temps_actuel, trajet_min):
    """
    Gère la contrainte 22h-5h. Si on dépasse 22h (1320), 
    on attend jusqu'à 5h (300) le lendemain.
    """
    arrivee = temps_actuel + trajet_min
    
    # Heure dans la journée actuelle
    heure_dans_jour = arrivee % 1440
    
    if heure_dans_jour > 1320: # Il est plus de 22h
        # On passe au lendemain 5h du matin
        nb_jours = arrivee // 1440
        arrivee = (nb_jours + 1) * 1440 + 300
        
    return arrivee

def formater_temps(total_mins):
    """Convertit les minutes en Jours, Heures, Minutes"""
    jours = int(total_mins // 1440)
    reste = total_mins % 1440
    heures = int(reste // 60)
    mins = int(reste % 60)
    return f"{jours}j {heures}h {mins}min"

# =============================================================================
# 3. FONCTION DE COÛT (ÉVALUATION AVEC PÉNALITÉS STRICTES)
# =============================================================================
@njit(cache=True)
def evalue_tournee_complexe(path, mat, e, l, s, P_array):
    temps_cumule = 300.0 # Départ à 5h00
    penalite = 0.0
    M = 50000.0 
    
    for i in range(len(path) - 1):
        v = path[i+1]
        temps_cumule += mat[path[i], v]
        
        heure_locale = temps_cumule % 1440
            
        # Gestion Fenêtre Périodique (Le camion peut rouler de nuit, mais attend l'ouverture)
        if heure_locale < e[v]:
            temps_cumule += (e[v] - heure_locale) 
        elif heure_locale > l[v]:
            jours = temps_cumule // 1440
            temps_cumule = (jours + 1) * 1440 + e[v]
            
        temps_cumule += s[v]
        
    temps_cumule += mat[path[-1], path[0]] 
    
    pos = np.empty(len(path), dtype=np.int64)
    for idx in range(len(path)):
        pos[path[idx]] = idx
    for k in range(len(P_array)):
        if pos[P_array[k, 0]] > pos[P_array[k, 1]]:
            penalite += M
            
    return temps_cumule + penalite

# =============================================================================
# 4. MÉTHODES DE RÉSOLUTION
# =============================================================================

def resolution_PuLP_Exact(mat, e, l, s, P, upper_bound=None, chemin_glouton=None, timeout=180):
    n = len(mat) - 1
    
    nom_unique = f"TSPTW_{uuid.uuid4().hex}"
    prob = LpProblem(nom_unique, LpMinimize)
    
    if upper_bound is not None:
        temps_max_absolu = math.ceil(upper_bound)
    else:
        temps_max_absolu = (n + 1) * 1440
        
    borne_sup_jours = int(temps_max_absolu // 1440)
    
    x = LpVariable.dicts("x", (range(n+1), range(n+1)), cat=LpBinary)
    T = LpVariable.dicts("T", range(n+1), lowBound=0, upBound=temps_max_absolu, cat=LpContinuous) 
    D = LpVariable.dicts("D", range(n+1), lowBound=0, upBound=borne_sup_jours, cat=LpInteger) 
    Cmax = LpVariable("Cmax", lowBound=0, upBound=temps_max_absolu, cat=LpContinuous)
    
    # =========================================================
    # LE WARM-START : Injection de la solution Gloutonne
    # =========================================================
    if chemin_glouton is not None:
        # On initialise d'abord tout à 0 pour être propre
        for i in range(n+1):
            D[i].setInitialValue(0)
            for j in range(n+1):
                x[i][j].setInitialValue(0)
                
        t_cumul = 300.0
        T[0].setInitialValue(300.0)
        
        # On simule le trajet du glouton pas à pas pour remplir les variables
        for k in range(len(chemin_glouton) - 1):
            u = chemin_glouton[k]
            v = chemin_glouton[k+1]
            x[u][v].setInitialValue(1) # Le camion passe par cette route
            
            t_cumul += mat[u][v]
            hl = t_cumul % 1440
            if hl < e[v]:
                t_cumul += (e[v] - hl)
            elif hl > l[v]:
                jours = t_cumul // 1440
                t_cumul = (jours + 1) * 1440 + e[v]
            
            D[v].setInitialValue(int(t_cumul // 1440)) # On donne le Jour exact
            T[v].setInitialValue(t_cumul)              # On donne l'Heure exacte
            
            t_cumul += s[v]
            
        # Retour au dépôt
        dernier = chemin_glouton[-1]
        x[dernier][0].setInitialValue(1)
        Cmax.setInitialValue(t_cumul + mat[dernier][0])

    # Objectif
    prob += Cmax 
    
    # Contraintes classiques et Coupes
    for i in range(n+1):
        prob += lpSum(x[i][j] for j in range(n+1) if i != j) == 1
        prob += lpSum(x[j][i] for j in range(n+1) if i != j) == 1
        
        for j in range(i+1, n+1):
            prob += x[i][j] + x[j][i] <= 1 # Anti aller-retour immédiat

    prob += T[0] == 300
    prob += D[0] == 0 
    
    for i in range(n+1):
        prob += T[i] >= (D[i] * 1440) + e[i]
        prob += T[i] <= (D[i] * 1440) + l[i]
        
        for j in range(1, n+1):
            if i != j:
                M_ij = temps_max_absolu + s[i] + mat[i][j] - e[j]
                prob += T[i] + s[i] + mat[i][j] <= T[j] + M_ij * (1 - x[i][j])
                prob += D[j] >= D[i] - (borne_sup_jours + 1) * (1 - x[i][j])

    for i in range(1, n+1):
        M_i0 = temps_max_absolu + s[i] + mat[i][0] - 300
        prob += Cmax >= T[i] + s[i] + mat[i][0] - M_i0 * (1 - x[i][0])

    for (i, j) in P:
        prob += T[i] + s[i] + mat[i][j] <= T[j]
        prob += D[j] >= D[i]

    # =========================================================
    # Lancement du solveur avec warmStart=True
    # =========================================================
    solveur = PULP_CBC_CMD(msg=0, timeLimit=timeout, warmStart=True, keepFiles=True) 
    prob.solve(solveur)
    
    # Extraction sécurisée (Sans se soucier du status si on a une solution !)
    if value(Cmax) is not None:
        succ = np.full(n+1, -1, dtype=np.int64)
        for i in range(n+1):
            for j in range(n+1):
                if i != j and value(x[i][j]) is not None and value(x[i][j]) > 0.5:
                    succ[i] = j
                    break
        if np.any(succ == -1): return np.nan
        path = [0]
        current = 0
        visites = set([0])
        for _ in range(n):
            nxt = int(succ[current])
            if nxt in visites: return np.nan
            path.append(nxt)
            visites.add(nxt)
            current = nxt
        if len(path) != n + 1: return np.nan
        path_np = np.array(path, dtype=np.int64)
        P_array = np.array(P, dtype=np.int64) if len(P) > 0 else np.empty((0, 2), dtype=np.int64)
        return evalue_tournee_complexe(path_np, mat, e, l, s, P_array)
    else:
        return np.nan

def borne_inferieure_TSP(mat):
    n = len(mat) - 1
    prob = LpProblem("Borne_Inf", LpMinimize)
    x = LpVariable.dicts("x", (range(n+1), range(n+1)), lowBound=0, upBound=1, cat=LpContinuous)
    prob += lpSum(mat[i][j] * x[i][j] for i in range(n+1) for j in range(n+1) if i != j)
    for i in range(n+1):
        prob += lpSum(x[i][j] for j in range(n+1) if i != j) == 1
        prob += lpSum(x[j][i] for j in range(n+1) if i != j) == 1
    prob.solve(PULP_CBC_CMD(msg=0))
    # La borne inferieure ne prend pas en compte le service ni les temps d'attente
    # On y ajoute la base du temps (5h = 300) pour être cohérent avec le graphique
    return value(prob.objective) + 300 

# ==========================================
# --- L'HEURISTIQUE GLOUTONNE (Baseline) ---
# ==========================================
def heuristique_gloutonne(mat, e, l, P):
    n = len(mat) - 1
    non_visites = set(range(1, n + 1))
    chemin = [0]
    temps_actuel = 300.0 # Départ 5h00
    
    while non_visites:
        meilleur_candidat = None
        meilleur_score = float('inf')
        
        for ville in non_visites:
            # 1. FILTRE DES PRÉCÉDENCES (On ne visite pas J si I n'est pas encore visité)
            precedence_ok = True
            for (i, j) in P:
                if ville == j and i not in chemin:
                    precedence_ok = False
                    break
            
            if not precedence_ok:
                continue # On ignore cette ville pour le moment
                
            # 2. CALCUL DU TEMPS ET DES FENÊTRES
            temps_trajet = mat[chemin[-1]][ville]
            arrivee = temps_actuel + temps_trajet
            
            heure_locale = arrivee % 1440
                
            if heure_locale < e[ville]:
                arrivee += (e[ville] - heure_locale)
            elif heure_locale > l[ville]:
                jours = arrivee // 1440
                arrivee = (jours + 1) * 1440 + e[ville] 
                
            score = arrivee 
            
            if score < meilleur_score:
                meilleur_score = score
                meilleur_candidat = ville
                
        # 3. SÉCURITÉ ANTI-BLOCAGE
        if meilleur_candidat is None:
            meilleur_candidat = list(non_visites)[0]
            meilleur_score = temps_actuel + mat[chemin[-1]][meilleur_candidat]
            
        chemin.append(meilleur_candidat)
        non_visites.remove(meilleur_candidat)
        
        # 4. MISE À JOUR DE L'HORLOGE (+10min de service moyen)
        temps_actuel = meilleur_score + 10 
            
    return np.array(chemin, dtype=np.int64)

@njit(cache=True)
def recuit_simule_adaptatif_numba(initial_path, mat, e, l, s, P_array, t_init=10000.0, alpha=0.99, iter_plateau=100):
    n = len(initial_path)
    current_path = initial_path.copy()
    current_cost = evalue_tournee_complexe(current_path, mat, e, l, s, P_array)
    
    best_path = current_path.copy()
    best_cost = current_cost
    
    # CRÉATION DU BUFFER (Une seule allocation RAM !)
    new_path = np.empty_like(current_path)
    
    T = t_init
    
    while T > 0.1:
        for _ in range(iter_plateau):
            i = np.random.randint(1, n)
            j = np.random.randint(1, n)
            if i == j: continue
            
            # RECOPIE SANS ALLOCATION RAM
            new_path[:] = current_path[:]
            
            if np.random.random() < 0.7:
                ville = current_path[i]
                if i < j:
                    new_path[i:j] = current_path[i+1:j+1]
                else:
                    new_path[j+1:i+1] = current_path[j:i]
                new_path[j] = ville
            else:
                new_path[i], new_path[j] = new_path[j], new_path[i]
            
            new_cost = evalue_tournee_complexe(new_path, mat, e, l, s, P_array)
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_path[:] = new_path[:]
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_path[:] = current_path[:]
                    best_cost = current_cost
        
        T *= alpha
        
    return best_cost

# =============================================================================
# 5. RECHERCHE TABOU
# =============================================================================
@njit(cache=True)
def recherche_tabou_numba(initial_path, mat, e, l, s, P_array, max_iter=2000, tabu_tenure=10, nb_voisins=50):
    n = len(initial_path)
    current_path = initial_path.copy()
    current_cost = evalue_tournee_complexe(current_path, mat, e, l, s, P_array)
    
    best_path = current_path.copy()
    best_cost = current_cost
    
    max_city_id = np.max(initial_path)
    tabu_matrix = np.zeros((max_city_id + 1, max_city_id + 1), dtype=np.int64)
    
    # BUFFERS MÉMOIRE
    new_path = np.empty_like(current_path)
    best_neighbor_path = np.empty_like(current_path)
    
    for iteration in range(max_iter):
        best_neighbor_cost = np.inf
        best_move = (-1, -1)
        
        for _ in range(nb_voisins):
            i = np.random.randint(1, n - 1)
            j = np.random.randint(1, n - 1)
            if i == j: continue
                
            new_path[:] = current_path[:]
            ville_i = current_path[i]
            ville_j = current_path[j]
            
            if np.random.random() < 0.7:
                ville = current_path[i]
                if i < j:
                    new_path[i:j] = current_path[i+1:j+1]
                else:
                    new_path[j+1:i+1] = current_path[j:i]
                new_path[j] = ville
            else:
                new_path[i], new_path[j] = new_path[j], new_path[i]
                
            cost = evalue_tournee_complexe(new_path, mat, e, l, s, P_array)
            is_tabu = iteration < tabu_matrix[ville_i, ville_j]
            
            if is_tabu and cost < best_cost:
                is_tabu = False
                
            if not is_tabu and cost < best_neighbor_cost:
                best_neighbor_cost = cost
                best_neighbor_path[:] = new_path[:]
                best_move = (ville_i, ville_j)
                
        if best_move[0] != -1:
            current_path[:] = best_neighbor_path[:]
            current_cost = best_neighbor_cost
            
            tabu_matrix[best_move[0], best_move[1]] = iteration + tabu_tenure
            tabu_matrix[best_move[1], best_move[0]] = iteration + tabu_tenure
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_path[:] = current_path[:]
                
    return best_cost

# =============================================================================
# 6. GENETIQUE
# =============================================================================
@njit(cache=True)
def crossover_ox_numba(parent1, parent2):
    n = len(parent1)
    child = np.zeros(n, dtype=np.int64)
    child[0] = 0 # Le dépôt reste toujours à 0
    
    # 1. Sélection de la zone à copier depuis le Parent 1
    p1 = np.random.randint(1, n)
    p2 = np.random.randint(1, n)
    if p1 > p2:
        p1, p2 = p2, p1
        
    # Sécurisation de la taille du tableau 'used'
    max_val = 0
    for x in parent1:
        if x > max_val: max_val = x
    used = np.zeros(max_val + 1, dtype=np.int64)
    used[0] = 1
    
    for i in range(p1, p2 + 1):
        child[i] = parent1[i]
        used[parent1[i]] = 1
        
    # 2. Remplissage avec le Parent 2
    curr_idx = p2 + 1
    if curr_idx >= n: 
        curr_idx = 1
        
    # CORRECTION ICI : On boucle bien sur 'n' éléments pour ne rater aucune ville !
    for i in range(n):
        idx = (p2 + 1 + i) % n
        val = parent2[idx]
        
        # On ignore le dépôt
        if val == 0: 
            continue
            
        # Si la ville n'est pas encore dans l'enfant, on l'ajoute
        if used[val] == 0:
            child[curr_idx] = val
            used[val] = 1
            curr_idx += 1
            if curr_idx >= n: 
                curr_idx = 1
                
    return child

@njit(cache=True)
def algorithme_genetique_numba(initial_path, mat, e, l, s, P_array, pop_size=100, generations=300, mutation_rate=0.1):
    n = len(initial_path)
    
    # 1. Initialisation de la population
    population = np.empty((pop_size, n), dtype=np.int64)
    population[0] = initial_path.copy() # On injecte le Glouton (Élitisme de départ)
    
    # Création d'individus aléatoires (en gardant le dépôt en position 0)
    for i in range(1, pop_size):
        p = initial_path.copy()
        for j in range(1, n):
            rand_idx = np.random.randint(1, n)
            p[j], p[rand_idx] = p[rand_idx], p[j]
        population[i] = p
        
    # Évaluation initiale
    fitness = np.empty(pop_size, dtype=np.float64)
    for i in range(pop_size):
        fitness[i] = evalue_tournee_complexe(population[i], mat, e, l, s, P_array)
        
    best_cost = np.min(fitness)
    best_path = population[np.argmin(fitness)].copy()
    
    new_population = np.empty_like(population)
    
    # 2. Cycle d'évolution
    for gen in range(generations):
        # Élitisme : Le champion survit toujours à la génération suivante
        new_population[0] = best_path.copy()
        
        for i in range(1, pop_size):
            # Sélection par Tournoi (On prend 2 fois 2 individus au hasard et on garde les meilleurs)
            t1, t2 = np.random.randint(pop_size), np.random.randint(pop_size)
            p1 = population[t1] if fitness[t1] < fitness[t2] else population[t2]
            
            t3, t4 = np.random.randint(pop_size), np.random.randint(pop_size)
            p2 = population[t3] if fitness[t3] < fitness[t4] else population[t4]
            
            # Croisement (Crossover)
            child = crossover_ox_numba(p1, p2)
            
            # Mutation (Swap)
            if np.random.random() < mutation_rate:
                m1, m2 = np.random.randint(1, n), np.random.randint(1, n)
                child[m1], child[m2] = child[m2], child[m1]
                
            new_population[i] = child
            
        # Mise à jour et Évaluation de la nouvelle génération
        for i in range(pop_size):
            population[i] = new_population[i]
            fitness[i] = evalue_tournee_complexe(population[i], mat, e, l, s, P_array)
            if fitness[i] < best_cost:
                best_cost = fitness[i]
                best_path = population[i].copy()
                
    return best_cost




import time
import concurrent.futures

def executer_un_run(n, run_id):
    """Calcule 1 Graphe et chronomètre chaque méthode."""
    seed_val = 42 + (n * 100) + run_id
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    mat, e, l, s, P = genere_instance_pure_aleatoire(n, num_precedences=max(1, n//3))
    P_array = np.array(P) if len(P) > 0 else np.empty((0, 2), dtype=np.int64)
    
    # 1. GLOUTON
    t0 = time.time()
    chemin_glouton = heuristique_gloutonne(mat, e, l, P)
    res_glouton = evalue_tournee_complexe(chemin_glouton, mat, e, l, s, P_array)
    t_glouton = time.time() - t0
    
    jours_glouton = int(res_glouton // 1440) + 1 
    
    # 2. PuLP EXACT (Bridé et Warm-Starté par le Glouton)
    t0 = time.time()
    timeout_val = 60 
    res_exact = np.nan
    if n <= 40: 
        # On passe le score ET le chemin du glouton à PuLP
        res_brut = resolution_PuLP_Exact(mat, e, l, s, P, upper_bound=res_glouton, chemin_glouton=chemin_glouton, timeout=timeout_val) 
        
        # Le filet de sécurité
        if not np.isnan(res_brut): 
            res_exact = res_brut
        else:
            res_exact = res_glouton # S'il plante, on garde le glouton
            
    t_exact = time.time() - t0
    
    # 3. RECUIT SIMULÉ
    t0 = time.time()
    dynamic_temp = float(n * 2000) 
    dynamic_alpha = float(1.0 - (0.1 / n)) 
    dynamic_plateau = int(n * 500) 
    res_sa = recuit_simule_adaptatif_numba(chemin_glouton, mat, e, l, s, P_array, t_init=dynamic_temp, alpha=dynamic_alpha, iter_plateau=dynamic_plateau)
    t_sa = time.time() - t0
    
    # 4. RECHERCHE TABOU
    t0 = time.time()
    dyn_iter = n * 250              
    dyn_voisins = n * 40            
    dyn_tenure = max(10, int(n * 0.8)) 
    res_tabou = recherche_tabou_numba(chemin_glouton, mat, e, l, s, P_array, max_iter=dyn_iter, tabu_tenure=dyn_tenure, nb_voisins=dyn_voisins)
    t_tabou = time.time() - t0
    
    # 5. GÉNÉTIQUE
    t0 = time.time()
    dyn_pop = 50 + (n * 2) 
    dyn_gen = n * 500       
    dyn_mut = 0.40         
    res_ga = algorithme_genetique_numba(chemin_glouton, mat, e, l, s, P_array, pop_size=dyn_pop, generations=dyn_gen, mutation_rate=dyn_mut)
    t_ga = time.time() - t0
    
    # On renvoie 2 groupes : (Les Scores) et (Les Temps d'exécution)
    return (res_exact, res_glouton, res_sa, res_tabou, res_ga), (t_exact, t_glouton, t_sa, t_tabou, t_ga)



# =============================================================================
# 7. MAIN BENCHMARK
# =============================================================================


def main():
    plt.close('all') 
    gc.collect()
    
    temps_debut_global = time.time()
    
    sizes = range(5, 31, 5) 
    nb_runs = 10 #5 
    
    results = {"Exact_Plot": [], "Glouton": [], "SA": [], "Tabou": [], "GA": []}
    std_results = {"Exact_Plot": [], "Glouton": [], "SA": [], "Tabou": [], "GA": []}
    
    progress = IntProgress(min=0, max=len(sizes) * nb_runs, description='Calculs:', layout={"width" : "100%"})
    display(progress)
    
    for n in sizes:
        print(f"\n" + "="*60)
        print(f"--- MOYENNE SUR {nb_runs} GRAPHES ALÉATOIRES POUR {n} SOMMETS ---")
        print("="*60)
        
        runs_data = {"Exact_Plot": [], "Glouton": [], "SA": [], "Tabou": [], "GA": []}
        time_data = {"Exact_Plot": [], "Glouton": [], "SA": [], "Tabou": [], "GA": []} # NOUVEAU
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            taches = [executor.submit(executer_un_run, n, run_id) for run_id in range(nb_runs)]
            
            for future in concurrent.futures.as_completed(taches):
                try:
                    # On déballe les scores ET les temps
                    (res_exact, res_glouton, res_sa, res_tabou, res_ga), (t_exact, t_glouton, t_sa, t_tabou, t_ga) = future.result()
                    
                    runs_data["Exact_Plot"].append(res_exact)
                    runs_data["Glouton"].append(res_glouton)
                    runs_data["SA"].append(res_sa)
                    runs_data["Tabou"].append(res_tabou)
                    runs_data["GA"].append(res_ga)
                    
                    time_data["Exact_Plot"].append(t_exact)
                    time_data["Glouton"].append(t_glouton)
                    time_data["SA"].append(t_sa)
                    time_data["Tabou"].append(t_tabou)
                    time_data["GA"].append(t_ga)
                    
                    progress.value += 1
                except Exception as e:
                    print(f"Erreur sur un processus : {e}")
            
        # =========================================================
        # CALCUL DES MOYENNES (SCORES ET TEMPS)
        # =========================================================
        avg_pulp = np.nanmean(runs_data["Exact_Plot"]) if not np.isnan(runs_data["Exact_Plot"]).all() else np.nan
        results["Exact_Plot"].append(avg_pulp)
        
        avg_glouton = np.nanmean(runs_data["Glouton"])
        results["Glouton"].append(avg_glouton)
        
        avg_sa = np.nanmean(runs_data["SA"])
        results["SA"].append(avg_sa)
        
        avg_tabou = np.nanmean(runs_data["Tabou"])
        results["Tabou"].append(avg_tabou)
        
        avg_ga = np.nanmean(runs_data["GA"])
        results["GA"].append(avg_ga)
        
        # Calcul des écarts-types
        std_results["Exact_Plot"].append(np.nanstd(runs_data["Exact_Plot"]) if not np.isnan(runs_data["Exact_Plot"]).all() else np.nan)
        std_results["Glouton"].append(np.nanstd(runs_data["Glouton"]))
        std_results["SA"].append(np.nanstd(runs_data["SA"]))
        std_results["Tabou"].append(np.nanstd(runs_data["Tabou"]))
        std_results["GA"].append(np.nanstd(runs_data["GA"]))
        
        # Moyennes des Temps CPU
        avg_t_pulp = np.nanmean(time_data["Exact_Plot"]) if not np.isnan(time_data["Exact_Plot"]).all() else np.nan
        avg_t_glouton = np.nanmean(time_data["Glouton"])
        avg_t_sa = np.nanmean(time_data["SA"])
        avg_t_tabou = np.nanmean(time_data["Tabou"])
        avg_t_ga = np.nanmean(time_data["GA"])
        
        # --- Affichage des Moyennes avec les temps d'exécution ---
        print(f"MOYENNES POUR {n} VILLES :")
        if not np.isnan(avg_pulp):
            print(f"- Simplexe        : {formater_temps(avg_pulp)} | Temps d'exec : {avg_t_pulp:.2f}s")
            print(f"- Glouton         : {formater_temps(avg_glouton)} (+{((avg_glouton - avg_pulp)/avg_pulp)*100:.2f}%) | Temps d'exec : {avg_t_glouton:.2f}s")
            print(f"- Recuit Simulé   : {formater_temps(avg_sa)} (+{((avg_sa - avg_pulp)/avg_pulp)*100:.2f}%) | Temps d'exec : {avg_t_sa:.2f}s")
            print(f"- Recherche Tabou : {formater_temps(avg_tabou)} (+{((avg_tabou - avg_pulp)/avg_pulp)*100:.2f}%) | Temps d'exec : {avg_t_tabou:.2f}s")
            print(f"- Algorithme Génétique : {formater_temps(avg_ga)} (+{((avg_ga - avg_pulp)/avg_pulp)*100:.2f}%) | Temps d'exec : {avg_t_ga:.2f}s")
        else:
            print(f"- Simplexe        : [100% Impossible / Timeouts]")
            print(f"- Glouton         : {formater_temps(avg_glouton)} | Temps d'exec : {avg_t_glouton:.2f}s")
            print(f"- Recuit Simulé   : {formater_temps(avg_sa)} | Temps d'exec : {avg_t_sa:.2f}s")
            print(f"- Recherche Tabou : {formater_temps(avg_tabou)} | Temps d'exec : {avg_t_tabou:.2f}s")
            print(f"- Algorithme Génétique : {formater_temps(avg_ga)} | Temps d'exec : {avg_t_ga:.2f}s")
            
        gc.collect()

    for fichier in os.listdir('.'):
            if fichier.endswith('.mps') or fichier.endswith('.mst') or fichier.endswith('.sol'):
                try:
                    os.remove(fichier)
                except:
                    pass
    progress.close()
    
    temps_total_global = time.time() - temps_debut_global
    print(f"\n" + "*"*60)
    print(f"TEMPS TOTAL D'EXÉCUTION DU BENCHMARK ({nb_runs} runs/taille) : {temps_total_global:.2f} secondes")
    print("*"*60 + "\n")

    # =============================================================================
    # --- GRAPHIQUE 1 : MOYENNES UNIQUEMENT ---
    # =============================================================================
    plt.figure(figsize=(14, 8))
    
    y_pulp = results["Exact_Plot"]
    y_glouton = results["Glouton"]
    y_sa = results["SA"]
    y_tabou = results["Tabou"]
    y_ga = results["GA"]

    plt.plot(sizes, y_pulp, marker='o', linestyle='-', label="Optimum Mathématique (Moyenne)", color='black', linewidth=2)
    plt.plot(sizes, y_sa, marker='D', linestyle='-', label="Recuit Simulé (Moyenne)", color='purple', linewidth=2)
    plt.plot(sizes, y_tabou, marker='X', linestyle='-', label="Recherche Tabou (Moyenne)", color='green', linewidth=2)
    plt.plot(sizes, y_ga, marker='v', linestyle='-', label="Algorithme Génétique", color='red', linewidth=2)

    if len(y_glouton) > 0:
        plt.plot(sizes, y_glouton, marker='s', linestyle='--', label="Glouton (Moyenne)", color='orange', alpha=0.8)

    for i in range(len(sizes)):
        if not np.isnan(y_pulp[i]):
            plt.annotate(f"{int(y_pulp[i])}m", (sizes[i], y_pulp[i]), textcoords="offset points", xytext=(-15, 10), ha='center', color='black', fontsize=9)
        if len(y_sa) > i and not np.isnan(y_sa[i]):
            plt.annotate(f"{int(y_sa[i])}m", (sizes[i], y_sa[i]), textcoords="offset points", xytext=(15, -15), ha='center', color='purple', fontsize=10, fontweight='bold')
        if len(y_tabou) > i and not np.isnan(y_tabou[i]):
            plt.annotate(f"{int(y_tabou[i])}m", (sizes[i], y_tabou[i]), textcoords="offset points", xytext=(0, -20), ha='center', color='green', fontsize=10, fontweight='bold')
        if len(y_ga) > i and not np.isnan(y_ga[i]):
            plt.annotate(f"{int(y_ga[i])}m", (sizes[i], y_ga[i]), textcoords="offset points", xytext=(0, 20), ha='center', color='red', fontsize=10, fontweight='bold')
            
    plt.xlabel("Nombre de villes (n)", fontsize=12)
    plt.ylabel("Durée totale moyenne (Minutes)", fontsize=12)
    
    valeurs_pulp_valides = [v for v in y_pulp if not np.isnan(v)]
    if valeurs_pulp_valides:
        plafond = max(valeurs_pulp_valides) * 2.0 
        plt.ylim(bottom=0, top=plafond)
        
    plt.title(f"Benchmark TSPTW-PC Aléatoire Pur : Moyennes sur {nb_runs} instances", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    plt.margins(y=0.2)
    plt.show()
    
    # =============================================================================
    # --- GRAPHIQUE 2 : BANDES D'ÉCART-TYPE ---
    # =============================================================================
    plt.figure(figsize=(14, 8))
    
    # Petite fonction utilitaire pour tracer proprement en évitant les crashs liés aux NaN
    def plot_with_fill(x, y, std, color, label, marker, linestyle):
        y_arr = np.array(y, dtype=float)
        std_arr = np.array(std, dtype=float)
        mask = ~np.isnan(y_arr)
        if np.any(mask):
            x_clean = np.array(x)[mask]
            y_clean = y_arr[mask]
            std_clean = std_arr[mask]
            # La courbe centrale de moyenne
            plt.plot(x_clean, y_clean, marker=marker, linestyle=linestyle, label=label, color=color, linewidth=2)
            # La bande transparente d'écart-type
            plt.fill_between(x_clean, 
                             np.subtract(y_clean, std_clean), 
                             np.add(y_clean, std_clean), 
                             color=color, alpha=0.1)

    plot_with_fill(sizes, y_pulp, std_results["Exact_Plot"], 'black', "Optimum Mathématique", 'o', '-')
    plot_with_fill(sizes, y_glouton, std_results["Glouton"], 'orange', "Glouton", 's', '--')
    plot_with_fill(sizes, y_sa, std_results["SA"], 'purple', "Recuit Simulé", 'D', '-')
    plot_with_fill(sizes, y_tabou, std_results["Tabou"], 'green', "Recherche Tabou", 'X', '-')
    plot_with_fill(sizes, y_ga, std_results["GA"], 'red', "Algorithme Génétique", 'v', '-')

    plt.xlabel("Nombre de villes (n)", fontsize=12)
    plt.ylabel("Durée totale (Minutes) ± Écart-type", fontsize=12)
    plt.title(f"Stabilité des Algorithmes : Courbes de moyenne avec bandes d'écart-type sur {nb_runs} runs", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    
    # On maintient le même plafond que le premier graphique pour comparer facilement
    if valeurs_pulp_valides:
        plt.ylim(bottom=0, top=plafond)
        
    plt.show()
    
    
# =============================================================================
    # --- GRAPHIQUE 3 : BOÎTE À MOUSTACHES (BOXPLOT) POUR L'INSTANCE MAXIMALE ---
    # =============================================================================
    plt.figure(figsize=(12, 7))
    
    # On récupère les données de la dernière itération de la boucle (le plus grand 'n')
    # On retire les 'NaN' (Timeouts) pour que matplotlib puisse dessiner la boîte
    data_pulp = [val for val in runs_data["Exact_Plot"] if not np.isnan(val)]
    data_glouton = [val for val in runs_data["Glouton"] if not np.isnan(val)]
    data_sa = [val for val in runs_data["SA"] if not np.isnan(val)]
    data_tabou = [val for val in runs_data["Tabou"] if not np.isnan(val)]
    data_ga = [val for val in runs_data["GA"] if not np.isnan(val)]

    donnees_a_tracer = [data_pulp, data_glouton, data_sa, data_tabou, data_ga]
    labels = ["PuLP Exact\n(Optimum)", "Glouton", "Recuit Simulé", "Rech. Tabou", "Génétique"]

    # Création du Boxplot
    bplot = plt.boxplot(donnees_a_tracer, 
                        labels=labels, 
                        patch_artist=True,  # Permet de colorer l'intérieur
                        showmeans=True,     # Affiche un petit triangle pour la moyenne
                        medianprops={'color': 'black', 'linewidth': 2}, # Ligne de la médiane
                        flierprops={'marker': 'o', 'color': 'red', 'alpha': 0.5}) # Valeurs aberrantes (outliers)

    # Couleurs pour correspondre à tes courbes précédentes
    couleurs = ['lightgrey', 'moccasin', 'thistle', 'lightgreen', 'lightcoral']
    for patch, couleur in zip(bplot['boxes'], couleurs):
        patch.set_facecolor(couleur)
        patch.set_alpha(0.7)

    plt.title(f"Distribution des scores pour l'instance la plus dure ({n} villes, {nb_runs} runs)", fontsize=14, fontweight='bold')
    plt.ylabel("Durée totale de la tournée (Minutes)", fontsize=12)
    
    # On limite l'axe Y pour la lisibilité si le Glouton s'est envolé
    valeurs_propres = data_pulp + data_sa + data_tabou + data_ga
    if valeurs_propres:
        plt.ylim(bottom=0, top=max(valeurs_propres) * 1.5)

    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.show()

# if __name__ == "__main__":
#     main()
    

import itertools

def auto_tune_recuit_simule(n_test=20, nb_runs_par_combo=3):
    print(f"=== DÉMARRAGE DU GRID SEARCH POUR {n_test} VILLES ===")
    
    # 1. On définit nos grilles de tests (les coefficients multiplicateurs)
    grille_temp_coeffs = [2000, 4000, 5000]
    grille_alpha_divs = [0.1, 0.5, 0.9] # Pour faire 1 - (div/n)
    grille_plateau_coeffs = [1000, 3000, 5000]
    
    # Toutes les combinaisons possibles (3x3x3 = 27 scénarios à tester)
    combinaisons = list(itertools.product(grille_temp_coeffs, grille_alpha_divs, grille_plateau_coeffs))
    print(f"{len(combinaisons)} combinaisons à tester...")
    
    meilleur_combo = None
    meilleur_score_moyen = float('inf')
    
    # Pour ne pas générer de nouveaux graphes à chaque combo, on fige les graphes de test
    instances_test = []
    for run in range(nb_runs_par_combo):
        random.seed(999 + run)
        np.random.seed(999 + run)
        instances_test.append(genere_instance_complexe(n_test))

    # 2. On lance la boucle de tests
    for idx, (t_coef, a_div, p_coef) in enumerate(combinaisons):
        scores_combo = []
        
        # Valeurs réelles pour N
        T = float(n_test * t_coef)
        alpha = float(1.0 - (a_div / n_test))
        plateau = int(n_test * p_coef)
        
        # On teste ce combo sur nos graphes pré-générés
        for mat, e, l, s, P in instances_test:
            P_array = np.array(P) if len(P) > 0 else np.empty((0, 2), dtype=np.int64)
            chemin_depart = heuristique_gloutonne(mat, e, l, P)
            
            score = recuit_simule_adaptatif_numba(chemin_depart, mat, e, l, s, P_array, 
                                                  t_init=T, alpha=alpha, iter_plateau=plateau)
            scores_combo.append(score)
            
        score_moyen = np.mean(scores_combo)
        print(f"[{idx+1}/{len(combinaisons)}] T:{t_coef}*n | a:1-({a_div}/n) | plat:{p_coef}*n ---> Score: {formater_temps(score_moyen)}")
        
        # 3. Sauvegarde du meilleur
        if score_moyen < meilleur_score_moyen:
            meilleur_score_moyen = score_moyen
            meilleur_combo = (t_coef, a_div, p_coef)
            
    # 4. Le Verdict
    print("\n" + "="*50)
    print("🏆 MEILLEURS PARAMÈTRES TROUVÉS :")
    print(f"Température Initiale = n * {meilleur_combo[0]}")
    print(f"Vitesse Refroidissement = 1 - ({meilleur_combo[1]} / n)")
    print(f"Plateau = n * {meilleur_combo[2]}")
    print("="*50)
    
    return meilleur_combo


# if __name__ == "__main__":
#     auto_tune_recuit_simule(n_test=20)

# =============================================================================
# 8. ANALYSE DE SENSIBILITÉ GLOBALE
# =============================================================================

def analyse_sensibilite_globale(n_villes=15, nb_runs=5):
    print("\n" + "="*70)
    print(f"DEMARRAGE DE L'ANALYSE DE SENSIBILITE GLOBALE ({n_villes} VILLES, {nb_runs} RUNS)")
    print("="*70)
    
    # 1. GENERATION DES INSTANCES FIGEES
    instances = []
    chemins_initiaux = []
    
    for run in range(nb_runs):
        random.seed(1000 + run)
        np.random.seed(1000 + run)
        mat, e, l, s, P = genere_instance_pure_aleatoire(n_villes, num_precedences=max(1, n_villes//3))
        P_array = np.array(P) if len(P) > 0 else np.empty((0, 2), dtype=np.int64)
        
        chemin_glouton = heuristique_gloutonne(mat, e, l, P)
        instances.append((mat, e, l, s, P_array))
        chemins_initiaux.append(chemin_glouton)
        
    def evaluer_parametre(algo_func, param_name, param_values, kwargs_base):
        scores_mean, scores_std, times_mean = [], [], []
        for val in param_values:
            scores, times = [], []
            for i in range(nb_runs):
                mat, e, l, s, P_array = instances[i]
                kwargs = kwargs_base.copy()
                kwargs[param_name] = val
                
                t0 = time.time()
                score = algo_func(chemins_initiaux[i], mat, e, l, s, P_array, **kwargs)
                times.append(time.time() - t0)
                scores.append(score)
                
            scores_mean.append(np.mean(scores))
            scores_std.append(np.std(scores))
            times_mean.append(np.mean(times))
            
            val_str = f"{val:.3f}" if isinstance(val, float) else f"{val:3d}"
            print(f"  - {param_name} = {val_str} : Score = {int(scores_mean[-1])}m | Temps = {times_mean[-1]:.3f}s")
        return scores_mean, scores_std, times_mean

    # 2. CALCULS : RECUIT ET TABOU
    print("\n[1/5] Analyse du Recuit Simule (Alpha)...")
    alphas = [0.01,0.05,0.1,0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    sa_s_mean, sa_s_std, sa_t_mean = evaluer_parametre(recuit_simule_adaptatif_numba, 'alpha', alphas, {'t_init': n_villes*2000, 'iter_plateau': n_villes*100})

    print("\n[2/5] Analyse de la Recherche Tabou (Tenure)...")
    tenures = [1, 2, 5, 10, 15, 20, 30, 50, 100, 500, 1000]
    tab_s_mean, tab_s_std, tab_t_mean = evaluer_parametre(recherche_tabou_numba, 'tabu_tenure', tenures, {'max_iter': n_villes*150, 'nb_voisins': n_villes*20})

    # 3. CALCULS : GENETIQUE 
    base_pop, base_gen, base_mut = 100, 500, 0.30
    
    print("\n[3/5] Genetique : Impact de la Population (TRES LARGE)...")
    pops = [2, 10, 50, 100, 300, 600, 1000] 
    gap_s_mean, gap_s_std, gap_t_mean = evaluer_parametre(algorithme_genetique_numba, 'pop_size', pops, {'generations': base_gen, 'mutation_rate': base_mut})

    print("\n[4/5] Genetique : Impact de la Mutation (0% a 100%)...")
    mutations = [0.0, 0.05, 0.15, 0.30, 0.60, 0.85, 1.0] # De l'immobilisme au chaos
    gam_s_mean, gam_s_std, gam_t_mean = evaluer_parametre(algorithme_genetique_numba, 'mutation_rate', mutations, {'pop_size': base_pop, 'generations': base_gen})

    print("\n[5/5] Genetique : Impact des Generations...")
    gens = [10, 100, 500, 1000, 3000, 5000]
    gag_s_mean, gag_s_std, gag_t_mean = evaluer_parametre(algorithme_genetique_numba, 'generations', gens, {'pop_size': base_pop, 'mutation_rate': base_mut})

    # 4. TRACAGE
    def tracer_separe(x_data, y_score, y_std, y_time, titre, xlabel):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        color_score, color_time = '#1f77b4', '#d62728'
        
        ax.set_title(titre, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Qualite (Minutes)", color=color_score, fontweight='bold')
        ax.plot(x_data, y_score, marker='o', color=color_score, lw=3, label="Score")
        ax.fill_between(x_data, np.subtract(y_score, y_std), np.add(y_score, y_std), color=color_score, alpha=0.1)
        ax.tick_params(axis='y', labelcolor=color_score)
        ax.grid(True, ls='--', alpha=0.5)
        
        ax2 = ax.twinx()
        ax2.set_ylabel("Temps CPU (Secondes)", color=color_time, fontweight='bold')
        ax2.plot(x_data, y_time, marker='s', ls=':', color=color_time, lw=2, label="Temps")
        ax2.tick_params(axis='y', labelcolor=color_time)
        plt.tight_layout()
        plt.show()

    # Affichage en fenetres separees
    tracer_separe(alphas, sa_s_mean, sa_s_std, sa_t_mean, "Recuit : Alpha", "Alpha")
    tracer_separe(tenures, tab_s_mean, tab_s_std, tab_t_mean, "Tabou : Tenure", "Tenure")
    tracer_separe(pops, gap_s_mean, gap_s_std, gap_t_mean, "Genetique : Population ", "Individus")
    tracer_separe(mutations, gam_s_mean, gam_s_std, gam_t_mean, "Genetique : Mutation ", "Taux mutation")
    tracer_separe(gens, gag_s_mean, gag_s_std, gag_t_mean, "Genetique : Generations (Loi du rendement decroissant)", "Generations")


if __name__ == "__main__":
    main() 
    #analyse_sensibilite_globale(n_villes=15, nb_runs=5)