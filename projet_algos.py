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
    mat_temps = np.random.randint(10, 45, size=(n+1, n+1))
    np.fill_diagonal(mat_temps, 0)
    
    chemin_secret = list(range(1, n+1))
    random.shuffle(chemin_secret)
    
    e = np.zeros(n+1)
    # On ne bloque plus l'heure de fin à 1320 ! (On met une valeur très haute par défaut)
    l = np.full(n+1, 999999.0) 
    s = np.random.randint(5, 15, size=n+1)
    s[0] = 0
    
    t_actuel = 300.0 # Départ à 5h00
    noeud_prec = 0
    
    for noeud in chemin_secret:
        t_actuel += mat_temps[noeud_prec][noeud]
        
        # --- ON SIMULE LA NUIT DANS LE GÉNÉRATEUR ---
        heure_locale = t_actuel % 1440
        if heure_locale > 1320: # S'il est plus de 22h
            jours = t_actuel // 1440
            t_actuel = (jours + 1) * 1440 + 300 # On avance à 5h le lendemain
            
        jour_actuel = t_actuel // 1440
        debut_journee = jour_actuel * 1440 + 300
        
        # L'ouverture au plus tôt (ne peut pas être avant 5h du matin DU MÊME JOUR)
        e[noeud] = max(debut_journee, t_actuel - random.randint(10, 45))
        
        # La fermeture n'est plus bridée au Jour 1 !
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
    temps_cumule = 300.0 # Départ jour 0 à 5h00
    penalite = 0.0
    M = 1000.0 
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        temps_cumule += mat[u, v]
        
        # --- GESTION DE LA NUIT ---
        heure_locale = temps_cumule % 1440
        if heure_locale > 1320: # S'il est plus de 22h00
            jours_ecoules = temps_cumule // 1440
            temps_cumule = (jours_ecoules + 1) * 1440 + 300 
            
        # --- CORRECTION DU BUG ICI ---
        # On compare le temps_cumule (absolu) avec les fenêtres e[v] et l[v] (absolues)
        if temps_cumule < e[v]:
            temps_cumule = e[v] # On avance l'horloge directement à l'heure d'ouverture
        elif temps_cumule > l[v]:
            penalite += (temps_cumule - l[v]) * M # La pénalité est calculée sur le vrai retard absolu
            
        temps_cumule += s[v]
        
    # Retour au dépôt
    temps_cumule += mat[path[-1], path[0]]
    
    # Précédences
    for k in range(len(P_array)):
        avant, apres = P_array[k]
        idx_avant, idx_apres = -1, -1
        for idx in range(len(path)):
            if path[idx] == avant: idx_avant = idx
            if path[idx] == apres: idx_apres = idx
        if idx_avant > idx_apres:
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
def recuit_simule_adaptatif_numba(initial_path, mat, e, l, s, P_array, t_init=5000.0, alpha=0.99, iter_plateau=100):
    current_path = initial_path.copy()
    current_cost = evalue_tournee_complexe(current_path, mat, e, l, s, P_array)
    
    best_path = current_path.copy()
    best_cost = current_cost
    
    T = t_init
    
    while T > 0.1:
        for _ in range(iter_plateau):
            idx1 = np.random.randint(1, len(current_path))
            idx2 = np.random.randint(1, len(current_path))
            
            if idx1 == idx2:
                continue
                
            new_path = current_path.copy()
            
            # --- MIX D'OPÉRATEURS (80% Insertion, 20% Swap) ---
            if np.random.random() < 0.8:
                # Mouvement 1 : Insertion (Glissement)
                ville = current_path[idx1]
                if idx1 < idx2:
                    new_path[idx1:idx2] = current_path[idx1+1:idx2+1]
                else:
                    new_path[idx2+1:idx1+1] = current_path[idx2:idx1]
                new_path[idx2] = ville
            else:
                # Mouvement 2 : Swap classique
                new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
            
            new_cost = evalue_tournee_complexe(new_path, mat, e, l, s, P_array)
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_path = new_path
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = current_path.copy()
        
        T *= alpha
            
    return best_cost

# =============================================================================
# 5. MAIN BENCHMARK
# =============================================================================

def main():
    plt.close('all') 
    gc.collect()
    
    # Chronomètre global
    temps_debut_global = time.time()
    
    sizes = range(5, 41, 5) 
    results = {"Borne": [], "Exact": [], "Exact_Plot": [], "SA": [], "Glouton": []}
    
    progress = IntProgress(min=0, max=len(sizes), description='Calculs:', layout={"width" : "100%"})
    display(progress)
    
    for n in sizes:
        random.seed(42 + n)
        np.random.seed(42 + n)
        print(f"\n" + "="*60)
        print(f"--- Test avec {n} sommets ---")
        print("="*60)
        
        mat, e, l, s, P = genere_instance_complexe(n)
        P_array = np.array(P) if len(P) > 0 else np.empty((0, 2), dtype=np.int64)
        
        # --- Borne ---
        res_borne = borne_inferieure_TSP(mat)
        results["Borne"].append(res_borne)
        print(f"Borne Inférieure (Idéal)   : {formater_temps(res_borne)}")
        
        # --- PuLP Exact ---
        timeout_val = 180 
        if n <= 40: 
            t0_pulp = time.time() # Chrono PuLP Start
            res_exact = resolution_PuLP_Exact(mat, e, l, s, P, timeout=timeout_val) 
            t_pulp_cpu = time.time() - t0_pulp # Chrono PuLP End
            
            results["Exact"].append(res_exact)
            
            if not np.isnan(res_exact):
                temps_pulp_total = res_exact + 300 + sum(s)
                results["Exact_Plot"].append(temps_pulp_total) 
                print(f"Optimum Mathématique (PuLP): {formater_temps(temps_pulp_total)} (Calcul: {t_pulp_cpu:.2f}s)")
            else:
                results["Exact_Plot"].append(np.nan)
                print(f"Optimum Mathématique (PuLP): [Timeout (> {timeout_val}s) ou Impossible] (Calcul: {t_pulp_cpu:.2f}s)")
        else:
            results["Exact"].append(np.nan)
            results["Exact_Plot"].append(np.nan)
            print(f"Optimum Mathématique (PuLP): [Ignoré - Instance trop complexe]")
            
            
            
            
        # ==========================================
        # --- L'HEURISTIQUE GLOUTONNE (Baseline) ---
        # ==========================================
        t0_glouton = time.time()
        chemin_glouton = heuristique_gloutonne(mat, e, l, P)
        # On évalue rigoureusement son score avec notre fonction officielle
        res_glouton = evalue_tournee_complexe(chemin_glouton, mat, e, l, s, P_array)
        t_glouton_cpu = time.time() - t0_glouton
        results["Glouton"].append(res_glouton)
        
        if not np.isnan(results["Exact_Plot"][-1]):
            ecart_glouton = ((res_glouton - results["Exact_Plot"][-1]) / results["Exact_Plot"][-1]) * 100
            print(f"Heuristique Gloutonne      : {formater_temps(res_glouton)} | Écart : +{ecart_glouton:.2f}% | (Calcul: {t_glouton_cpu:.4f}s)")
        else:
            print(f"Heuristique Gloutonne      : {formater_temps(res_glouton)} | (Calcul: {t_glouton_cpu:.4f}s)")

        # ==========================================
        # --- RECUIT SIMULÉ (Amélioration) ---
        # ==========================================
        dynamic_temp = float(n * 500) 
        dynamic_alpha = float(1.0 - (0.10 / n)) 
        dynamic_plateau = int(n * 150) 
        
        t0_sa = time.time()
        # On donne le chemin du Glouton comme point de départ au Recuit !
        res_sa = recuit_simule_adaptatif_numba(chemin_glouton, mat, e, l, s, P_array, 
                                               t_init=dynamic_temp, 
                                               alpha=dynamic_alpha, 
                                               iter_plateau=dynamic_plateau)
        t_sa_cpu = time.time() - t0_sa
        
        results["SA"].append(res_sa)
        
        if not np.isnan(results["Exact_Plot"][-1]):
            ecart_sa = ((res_sa - results["Exact_Plot"][-1]) / results["Exact_Plot"][-1]) * 100
            print(f"Recuit Adaptatif (Numba)   : {formater_temps(res_sa)} | Écart : +{ecart_sa:.2f}% | (Calcul: {t_sa_cpu:.2f}s)")
            print(f"-> Amélioration par rapport au Glouton : {((res_glouton - res_sa)/res_glouton)*100:.2f}% de temps gagné !")
        else:
            print(f"Recuit Adaptatif (Numba)   : {formater_temps(res_sa)} | (Calcul: {t_sa_cpu:.2f}s)")
            
            
            
            
            
            
        # # --- Recuit Simulé ---
        # dynamic_temp = float(n * 500) 
        # # Refroidissement un peu plus lent pour mieux chercher
        # dynamic_alpha = float(1.0 - (0.39 / n)) 
        # dynamic_plateau = int(n * 200) 
        
        # villes_sans_depot = list(range(1, n+1))
        # villes_sans_depot.sort(key=lambda x: e[x]) 
        # initial_path = np.array([0] + villes_sans_depot, dtype=np.int64)
        
        # t0_sa = time.time() # Chrono SA Start
        # res_sa = recuit_simule_adaptatif_numba(initial_path, mat, e, l, s, P_array, 
        #                                        t_init=dynamic_temp, 
        #                                        alpha=dynamic_alpha, 
        #                                        iter_plateau=dynamic_plateau)
        # t_sa_cpu = time.time() - t0_sa # Chrono SA End
        
        # results["SA"].append(res_sa)
        
        # if not np.isnan(results["Exact_Plot"][-1]):
        #     ecart = ((res_sa - results["Exact_Plot"][-1]) / results["Exact_Plot"][-1]) * 100
        #     print(f"Recuit Adaptatif (Numba)   : {formater_temps(res_sa)} | Écart : +{ecart:.2f}% | (Calcul: {t_sa_cpu:.2f}s)")
        # else:
        #     print(f"Recuit Adaptatif (Numba)   : {formater_temps(res_sa)} | (Calcul: {t_sa_cpu:.2f}s)")
            
        progress.value += 1
        gc.collect()

    progress.close()
    
    # Affichage du temps total
    temps_total_global = time.time() - temps_debut_global
    print(f"\n" + "*"*60)
    print(f"⏳ TEMPS TOTAL D'EXÉCUTION DU SCRIPT : {temps_total_global:.2f} secondes")
    print("*"*60 + "\n")


# =============================================================================
    # --- AFFICHAGE DU GRAPHIQUE ---
    # =============================================================================
    plt.figure(figsize=(14, 8))
    
    # 1. Récupération directe des listes principales
    y_borne = results["Borne"]
    y_pulp = results["Exact_Plot"]
    y_sa = results["SA"]

    # 2. Tracé des courbes
    plt.plot(sizes, y_borne, marker='^', linestyle=':', label="Borne Inférieure (Plancher théorique)", color='gray', alpha=0.7)
    plt.plot(sizes, y_pulp, marker='o', linestyle='-', label="Optimum Mathématique (PuLP)", color='black', linewidth=2)
    plt.plot(sizes, y_sa, marker='D', linestyle='-', label="Recuit Simulé Adaptatif (Numba)", color='purple', linewidth=2)

    # Correction pour le Glouton : on récupère et on trace SEULEMENT s'il existe
    if "Glouton" in results and len(results["Glouton"]) > 0:
        y_glouton = results["Glouton"]
        # Correction de l'erreur ici : on donne toute la liste 'y_glouton' et non 'y_glouton[i]'
        plt.plot(sizes, y_glouton, marker='s', linestyle='--', label="Heuristique Gloutonne", color='orange', alpha=0.8)

    # 3. Ajout des annotations de temps (en minutes)
    for i in range(len(sizes)):
        if not np.isnan(y_pulp[i]):
            plt.annotate(f"{int(y_pulp[i])}m", (sizes[i], y_pulp[i]), 
                         textcoords="offset points", xytext=(-15, 10), ha='center', color='black', fontsize=9)
        
        # On vérifie que la liste SA a bien été remplie
        if len(y_sa) > i and not np.isnan(y_sa[i]):
            plt.annotate(f"{int(y_sa[i])}m", (sizes[i], y_sa[i]), 
                         textcoords="offset points", xytext=(15, -15), ha='center', color='purple', fontsize=10, fontweight='bold')
            
        # (Optionnel) Annotation pour le Glouton s'il est là
        if "Glouton" in results and len(results["Glouton"]) > i and not np.isnan(results["Glouton"][i]):
            plt.annotate(f"{int(results['Glouton'][i])}m", (sizes[i], results['Glouton'][i]), 
                         textcoords="offset points", xytext=(0, 15), ha='center', color='orange', fontsize=9)

    # 4. Configuration des axes
    plt.xlabel("Nombre de villes (n)", fontsize=12)
    plt.ylabel("Durée totale de la tournée (Minutes)", fontsize=12)
    plt.title("Benchmark TSPTW-PC : Performance des Algorithmes", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")
    
    # Ajustement des limites pour la lisibilité
    plt.margins(y=0.2)
    
    
    # --- LIMITATION DE L'AXE Y (Anti-Explosion du Glouton) ---
    valeurs_pulp_valides = [v for v in y_pulp if not np.isnan(v)]
    if valeurs_pulp_valides:
        plafond = max(valeurs_pulp_valides) * 1.5 # On coupe à +50% du temps max de PuLP
        plt.ylim(bottom=min(y_borne) * 0.9, top=plafond)
    
    plt.show()
    gc.collect()

if __name__ == "__main__":
    main()