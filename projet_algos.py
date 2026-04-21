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
    # qu'un détour par C. (Modélise le transit sans service !)
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
        return f"🚨 [INVALIDE - Score pénalité : {int(total_mins)}]"
        
    heures = int(total_mins // 60)
    mins = int(total_mins % 60)
    return f"{heures}h{mins:02d}"



def genere_instance_pure_aleatoire(n, num_precedences=2):
    # 1. Génération des coordonnées (0 à 100)
    coords = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n+1)]
    
    # 2. Calcul de la matrice des distances (Euclidienne)
    mat_temps = np.zeros((n+1, n+1), dtype=np.float64)
    for i in range(n+1):
        for j in range(n+1):
            if i != j:
                mat_temps[i][j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                
    # On applique Floyd-Warshall pour garantir des routes logiques !
    for k in range(n+1):
        for i in range(n+1):
            for j in range(n+1):
                if mat_temps[i][k] + mat_temps[k][j] < mat_temps[i][j]:
                    mat_temps[i][j] = mat_temps[i][k] + mat_temps[k][j]
    
    mat_temps = mat_temps.astype(np.int64)

    # 3. Temps de service
    s = np.random.randint(5, 15, size=n+1)
    s[0] = 0

    # 4. Fenêtres de temps 100% Aléatoires (Réparties sur 3 jours logiques)
    e = np.zeros(n+1)
    l = np.full(n+1, 999999.0)
    for i in range(1, n+1):
        jour = random.randint(0, 2) # L'ouverture peut tomber le Jour 0, 1 ou 2
        heure_ouverture = random.randint(300, 1000) # Entre 5h et 16h40
        
        e[i] = jour * 1440 + heure_ouverture
        l[i] = e[i] + random.randint(60, 300) # La fenêtre reste ouverte entre 1h et 5h

    # 5. Précédences Aléatoires (comme ton code)
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
    M = 1000.0 
    
    # --- 1. Calcul du Temps et Fenêtres (O(N)) ---
    for i in range(len(path) - 1):
        v = path[i+1]
        temps_cumule += mat[path[i], v]
        
        heure_locale = temps_cumule % 1440
        if heure_locale > 1320: # Gestion de la nuit (> 22h)
            jours_ecoules = temps_cumule // 1440
            temps_cumule = (jours_ecoules + 1) * 1440 + 300 
            
        if temps_cumule < e[v]:
            temps_cumule = e[v] # Attente
        elif temps_cumule > l[v]:
            penalite += (temps_cumule - l[v]) * M # Retard
            
        temps_cumule += s[v]
        
    temps_cumule += mat[path[-1], path[0]] # Retour au dépôt
    
    # --- 2. Vérification des Précédences (Ultra-Optimisé O(N)) ---
    # Au lieu de chercher dans tout le tableau à chaque fois, on stocke la position
    # de chaque ville une seule fois ! (C'est ce qui ralentissait tout avant)
    pos = np.empty(len(path), dtype=np.int64)
    for idx in range(len(path)):
        pos[path[idx]] = idx
        
    for k in range(len(P_array)):
        if pos[P_array[k, 0]] > pos[P_array[k, 1]]:
            penalite += M * 50
            
    return temps_cumule + penalite

# =============================================================================
# 4. MÉTHODES DE RÉSOLUTION
# =============================================================================

def resolution_PuLP_Exact(mat, e, l, s, P, timeout=120):
    n = len(mat) - 1
    prob = LpProblem("TSPTW_PC", LpMinimize)
    
    x = LpVariable.dicts("x", (range(n+1), range(n+1)), cat=LpBinary)
    T = LpVariable.dicts("T", range(n+1), lowBound=0, cat=LpContinuous)
    
    prob += lpSum(mat[i][j] * x[i][j] for i in range(n+1) for j in range(n+1) if i != j)
    
    for i in range(n+1):
        prob += lpSum(x[i][j] for j in range(n+1) if i != j) == 1
        prob += lpSum(x[j][i] for j in range(n+1) if i != j) == 1

    M = 10000
    for i in range(n+1):
        prob += T[i] >= e[i]
        prob += T[i] <= l[i]
        for j in range(1, n+1):
            if i != j:
                prob += T[i] + s[i] + mat[i][j] - M*(1 - x[i][j]) <= T[j]

    for (i, j) in P:
        prob += T[i] + s[i] + mat[i][j] <= T[j]

    # Contrainte globale 22h max
    # for i in range(n+1):
    #     prob += T[i] <= 1320

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=timeout))
    
    if prob.status == 1: # Solution Optimale trouvée
        return value(prob.objective)
    else: # Infeasible ou Timeout
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
    """
    Construit un chemin ville par ville en choisissant le meilleur compromis 
    (Trajet + Attente) tout en respectant les précédences.
    """
    n = len(mat) - 1
    non_visites = set(range(1, n + 1))
    chemin = [0]
    temps_actuel = 300.0 # Départ 5h00
    
    while non_visites:
        meilleur_candidat = None
        meilleur_score = float('inf')
        
        for ville in non_visites:
            # 1. Vérifier si on a le droit d'y aller (Précédences)
            precedence_ok = True
            for (avant, apres) in P:
                if apres == ville and avant in non_visites:
                    precedence_ok = False # La ville 'avant' n'a pas encore été visitée !
                    break
            
            if not precedence_ok:
                continue # On passe à la ville suivante
                
            # 2. Simuler le trajet
            temps_trajet = mat[chemin[-1]][ville]
            arrivee = temps_actuel + temps_trajet
            
            # Gestion de la nuit (si on dépasse 22h)
            if (arrivee % 1440) > 1320:
                jours = arrivee // 1440
                arrivee = (jours + 1) * 1440 + 300
                
            # 3. Calcul du "Score" (On veut le score le plus bas)
            score = temps_trajet
            if arrivee < e[ville]:
                score += (e[ville] - arrivee) # On ajoute le temps perdu à attendre
            elif arrivee > l[ville]:
                score += (arrivee - l[ville]) * 100 # Grosse pénalité si on est en retard
                
            # 4. Garder le meilleur
            if score < meilleur_score:
                meilleur_score = score
                meilleur_candidat = ville
                
        # Si aucun candidat valide (sécurité), on prend une ville au hasard
        if meilleur_candidat is None:
            meilleur_candidat = list(non_visites)[0]
            
        chemin.append(meilleur_candidat)
        non_visites.remove(meilleur_candidat)
        
        # Mise à jour approximative du temps pour la prochaine itération
        temps_actuel += mat[chemin[-2]][meilleur_candidat]
        if temps_actuel < e[meilleur_candidat]:
            temps_actuel = e[meilleur_candidat]
            
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


# =============================================================================
# 7. MAIN BENCHMARK
# =============================================================================


def main():
    plt.close('all') 
    gc.collect()
    
    temps_debut_global = time.time()
    
    sizes = range(5, 31, 5) #41
    nb_runs = 3 # Le nombre d'instances différentes à tester par taille de ville !
    
    # Nos listes stockeront désormais les MOYENNES
    results = {"Borne": [], "Exact_Plot": [], "Glouton": [], "SA": [], "Tabou": [], "GA": []}
    
    progress = IntProgress(min=0, max=len(sizes) * nb_runs, description='Calculs:', layout={"width" : "100%"})
    display(progress)
    
    for n in sizes:
        print(f"\n" + "="*60)
        print(f"--- MOYENNE SUR {nb_runs} GRAPHES POUR {n} SOMMETS ---")
        print("="*60)
        
        # Dictionnaire temporaire pour stocker les 5 essais de cette taille
        runs_data = {"Borne": [], "Exact_Plot": [], "Glouton": [], "SA": [], "Tabou": [], "GA": []}
        
        for run in range(nb_runs):
            # La graine change à chaque run, mais reste reproductible 
            seed_val = 42 + (n * 100) + run
            random.seed(seed_val)
            np.random.seed(seed_val)
            
            mat, e, l, s, P = genere_instance_complexe(n, precedence_prob=0.3)
            #mat, e, l, s, P = genere_instance_pure_aleatoire(n, num_precedences=n//2)
            P_array = np.array(P) if len(P) > 0 else np.empty((0, 2), dtype=np.int64)
            
            # --- 1. Borne ---
            res_borne = borne_inferieure_TSP(mat)
            runs_data["Borne"].append(res_borne)
            
            # --- 2. PuLP Exact ---
            timeout_val = 180 
            if n <= 40: 
                t0_pulp = time.time()
                res_exact = resolution_PuLP_Exact(mat, e, l, s, P, timeout=timeout_val) 
                t_pulp_cpu = time.time() - t0_pulp
                if not np.isnan(res_exact):
                    temps_pulp_total = res_exact + 300 + sum(s)
                    runs_data["Exact_Plot"].append(temps_pulp_total) 
                else:
                    runs_data["Exact_Plot"].append(np.nan)
            else:
                runs_data["Exact_Plot"].append(np.nan)
                
            # --- 3. Heuristique Gloutonne ---
            t0_glouton = time.time()  
            chemin_glouton = heuristique_gloutonne(mat, e, l, P)
            res_glouton = evalue_tournee_complexe(chemin_glouton, mat, e, l, s, P_array)
            t_glouton_cpu = time.time() - t0_glouton
            runs_data["Glouton"].append(res_glouton)

            # --- 4. Recuit Simulé ---
            dynamic_temp = float(n * 1000) 
            dynamic_alpha = float(1.0 - (0.1 / n)) #vitesse de refroiddisement 
            dynamic_plateau = int(n * 300) # 150 #tests par température 
            
            t0_sa = time.time()
            res_sa = recuit_simule_adaptatif_numba(chemin_glouton, mat, e, l, s, P_array, 
                                                   t_init=dynamic_temp, 
                                                   alpha=dynamic_alpha, 
                                                   iter_plateau=dynamic_plateau)
            t_sa_cpu = time.time() - t0_sa
            runs_data["SA"].append(res_sa)
            
            
            # --- 5. Recherche Tabou ---
            # Paramètres dynamiques : plus il y a de villes, plus on fouille !
            dyn_iter = n * 250              # Nombre d'itérations
            dyn_voisins = n * 40            # Voisins explorés à CHAQUE itération
            dyn_tenure = max(5, int(n * 0.4)) # Durée de l'interdiction tabou
            
            res_tabou = recherche_tabou_numba(chemin_glouton, mat, e, l, s, P_array,
                                              max_iter=dyn_iter,
                                              tabu_tenure=dyn_tenure,
                                              nb_voisins=dyn_voisins)
            runs_data["Tabou"].append(res_tabou)
            
            # --- 6. Algorithme Génétique ---
            # Paramètres adaptatifs
            dyn_pop = 50 + (n * 2) # Plus il y a de villes, plus on a besoin de population
            dyn_gen = n * 50       # Nombre de générations
            dyn_mut = 0.15         # Taux de mutation
            
            res_ga = algorithme_genetique_numba(chemin_glouton, mat, e, l, s, P_array,
                                                pop_size=dyn_pop,
                                                generations=dyn_gen,
                                                mutation_rate=dyn_mut)
            runs_data["GA"].append(res_ga)
            
            progress.value += 1
            
        # =========================================================
        # CALCUL DES MOYENNES POUR LA TAILLE 'n'
        # np.nanmean calcule la moyenne en ignorant les np.nan (les Timeouts)
        # =========================================================
        
        avg_borne = np.nanmean(runs_data["Borne"])
        results["Borne"].append(avg_borne)
        
        # S'il n'y a QUE des NaN (ex: 5 timeouts de suite), nanmean lève un warning, on le gère :
        if np.isnan(runs_data["Exact_Plot"]).all():
            avg_pulp = np.nan
        else:
            avg_pulp = np.nanmean(runs_data["Exact_Plot"])
        results["Exact_Plot"].append(avg_pulp)
        
        avg_glouton = np.nanmean(runs_data["Glouton"])
        results["Glouton"].append(avg_glouton)
        
        avg_sa = np.nanmean(runs_data["SA"])
        results["SA"].append(avg_sa)
        
        avg_tabou = np.nanmean(runs_data["Tabou"])
        results["Tabou"].append(avg_tabou)
        
        avg_ga = np.nanmean(runs_data["GA"])
        results["GA"].append(avg_ga)
        
        
        # --- Affichage des Moyennes dans la console ---
        print(f"MOYENNES POUR {n} VILLES :")
        print(f"- Borne Idéale    : {formater_temps(avg_borne)}")
        
        if not np.isnan(avg_pulp):
            print(f"- PuLP Exact      : {formater_temps(avg_pulp)}")
            print(f"- Glouton         : {formater_temps(avg_glouton)} (+{((avg_glouton - avg_pulp)/avg_pulp)*100:.2f}%b)")
            print(f"- Recuit Simulé   : {formater_temps(avg_sa)} (+{((avg_sa - avg_pulp)/avg_pulp)*100:.2f}%)")
            print(f"- Recherche Tabou : {formater_temps(avg_tabou)} (+{((avg_tabou - avg_pulp)/avg_pulp)*100:.2f}%)")
            print(f"- Algo  Génétique : {formater_temps(avg_ga)} (+{((avg_ga - avg_pulp)/avg_pulp)*100:.2f}%)")
            print(f"-> Amélioration Recuit vs Glouton : {((avg_glouton - avg_sa)/avg_glouton)*100:.2f}% de temps gagné !")
        else:
            print(f"- PuLP Exact      : [100% de Timeouts]")
            print(f"- Glouton         : {formater_temps(avg_glouton)}")
            print(f"- Recuit Simulé   : {formater_temps(avg_sa)}")
            print(f"- Recherche Tabou : {formater_temps(avg_tabou)}")
            
        gc.collect()

    progress.close()
    
    temps_total_global = time.time() - temps_debut_global
    print(f"\n" + "*"*60)
    print(f"⏳ TEMPS TOTAL D'EXÉCUTION DU BENCHMARK ({nb_runs} runs/taille) : {temps_total_global:.2f} secondes")
    print("*"*60 + "\n")

    # =============================================================================
    # --- AFFICHAGE DU GRAPHIQUE ---
    # =============================================================================
    plt.figure(figsize=(14, 8))
    
    y_borne = results["Borne"]
    y_pulp = results["Exact_Plot"]
    y_glouton = results["Glouton"]
    y_sa = results["SA"]
    y_tabou = results["Tabou"]
    y_ga = results["GA"]

    #plt.plot(sizes, y_borne, marker='^', linestyle=':', label="Borne Inférieure (Moyenne)", color='gray', alpha=0.7)
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
        if len(y_glouton) > i and not np.isnan(y_glouton[i]):
            plt.annotate(f"{int(y_glouton[i])}m", (sizes[i], y_glouton[i]), textcoords="offset points", xytext=(0, 15), ha='center', color='orange', fontsize=9)
        if len(y_tabou) > i and not np.isnan(y_tabou[i]):
            plt.annotate(f"{int(y_tabou[i])}m", (sizes[i], y_tabou[i]), textcoords="offset points", xytext=(0, -20), ha='center', color='green', fontsize=10, fontweight='bold')
        if len(y_ga) > i and not np.isnan(y_ga[i]):
            plt.annotate(f"{int(y_ga[i])}m", (sizes[i], y_ga[i]), textcoords="offset points", xytext=(0, 20), ha='center', color='red', fontsize=10, fontweight='bold')
            
            
    plt.xlabel("Nombre de villes (n)", fontsize=12)
    plt.ylabel("Durée totale moyenne de la tournée (Minutes)", fontsize=12)
    
    # NOUVEAU LIMITATEUR D'AXE Y POUR LE GRAPHIQUE
    valeurs_pulp_valides = [v for v in y_pulp if not np.isnan(v)]
    if valeurs_pulp_valides:
        plafond = max(valeurs_pulp_valides) * 1.5 
        plt.ylim(bottom=min(y_borne) * 0.9, top=plafond)
        
    plt.title(f"Benchmark TSPTW-PC : Moyennes sur {nb_runs} instances indépendantes", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    plt.margins(y=0.2)
    plt.show()


    
    
if __name__ == "__main__":
    main()
    
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