# 🧠 RRT Planner for Niryo Ned

## 🎯 Objectif
Ce projet implémente un planificateur de trajectoire basé sur l’algorithme **RRT (Rapidly-exploring Random Tree)** pour le bras robotique **Niryo Ned**.  
Il permet de générer des trajectoires optimisées en espace cartésien, avec gestion d'obstacles et affichage visuel des chemins planifiés.

## 🛠️ Fonctionnalités
- Génération d’un graphe RRT pour explorer l’espace de configuration du robot.
- Calcul de trajectoires optimales entre deux points.
- Évitement d’obstacles via régénération adaptative du graphe.
- Affichage des trajectoires et analyse de la résolution en fonction du nombre d’itérations.
- Mode "RRT dynamique" où l’arbre évolue en temps réel pendant le déplacement du robot.

## ⚙️ Technologies utilisées
- **Python**
- **Matplotlib / Pygame** pour la visualisation
- **API Niryo Ned (pyniryo)** pour le contrôle du bras robotique
- Structures de données : arbres, graphes
- Algorithmes : RRT, recherche de plus court chemin
