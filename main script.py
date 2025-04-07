# main_script.py
import numpy as np
import time
from pyniryo import * # Importation de la bibliothèque pour contrôler le robot
#from niryo_robot_python_ros_wrapper import *
from RRT_algorithm_with_obstacles import RRT  # Importation de l'algorithme RRT

# Connexion au robot Niryo (remplacer l'adresse IP par celle du robot)
robot = NiryoRobot("10.10.10.10")  # Adresse IP du robot
robot.calibrate_auto()  # Calibrer le robot automatiquement pour commencer
robot.update_tool()  # Mise à jour de l'outil du robot

# Définir les paramètres de l'environnement
start = [0, 0]  # Point de départ en 2D (x, y)
goal = [8, 8]   # Point objectif en 2D
x_range = [0, 10]  # Plage des coordonnées x
y_range = [0, 10]  # Plage des coordonnées y
z_value = 0  # La coordonnée z est fixe, le robot travaille en 2D
step_size = 0.5  # Taille du pas pour l'algorithme RRT
max_iter = 500  # Nombre maximum d'itérations de l'algorithme

# Définir des obstacles sous forme de cercles (centre, rayon)
obstacles = [([3, 3], 1), ([6, 6], 1.5), ([5, 8], 0.8)]

# Initialiser l'algorithme RRT
rrt = RRT(start, goal, x_range, y_range, z_value, step_size, max_iter, obstacles)

# Construire l'arbre RRT
rrt.build_tree()

# Afficher l'arbre et la trajectoire générée
rrt.plot()

# Récupérer la trajectoire générée par l'algorithme RRT
path = rrt.reconstruct_path()


