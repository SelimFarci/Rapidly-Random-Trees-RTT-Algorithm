import numpy as np
import matplotlib.pyplot as plt
import cv2

class DynamicRRT:
    def __init__(self, start, goal, x_range, y_range, obstacles, step_size=0.5, max_iter=100, prune_threshold=50, video_filename="rrt_simulation.avi"):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.x_range = x_range
        self.y_range = y_range
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.start]  # Initialise l’arbre avec le point de départ
        self.edges = []  # Liste des connexions entre les nœuds
        self.prune_threshold = prune_threshold  # Seuil de suppression des nœuds
        self.parents = {tuple(self.start): None}  # Stocke les parents de chaque nœud
        self.frames = []  # Liste des images pour la vidéo

        # Initialisation de la vidéo avec OpenCV
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(video_filename, self.fourcc, 10, (800, 800))

    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def random_point(self):
        """Génère un point aléatoire dans l'espace cartésien défini."""
        x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = np.random.uniform(self.y_range[0], self.y_range[1])
        return np.array([x, y])

    def nearest_node(self, random_point):
        """Trouve le nœud le plus proche dans l’arbre."""
        distances = [self.distance(node, random_point) for node in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]

    def steer(self, from_node, to_point):
        """Crée un nouveau nœud dans la direction du point donné."""
        direction = to_point - from_node
        distance = np.linalg.norm(direction)
        direction = direction / distance
        step = min(self.step_size, distance)
        new_node = from_node + direction * step
        return new_node

    def collision_free(self, point1, point2):
        """Vérifie si la trajectoire entre point1 et point2 est libre d'obstacles."""
        for (ox, oy, radius) in self.obstacles:
            # Vérifie si la ligne entre point1 et point2 traverse un obstacle
            dx, dy = point2 - point1
            dist = np.linalg.norm([dx, dy])
            steps = int(dist / self.step_size)
            for i in range(steps):
                x = point1[0] + (i / steps) * dx
                y = point1[1] + (i / steps) * dy
                if np.linalg.norm([x - ox, y - oy]) <= radius:
                    return False
        return True

    def prune_tree(self):
        """Prune les nœuds trop éloignés de l'objectif et inutilement proches des racines."""
        pruned_tree = []
        pruned_edges = []
        for node in self.tree:
            if self.distance(node, self.goal) <= self.prune_threshold:
                pruned_tree.append(node)

        # Vérifie si chaque arête connecte uniquement des nœuds dans l'arbre élagué
        for edge in self.edges:
            p1, p2 = edge
            if any(np.array_equal(p1, n) for n in pruned_tree) and any(np.array_equal(p2, n) for n in pruned_tree):
                pruned_edges.append(edge)

        self.tree = pruned_tree
        self.edges = pruned_edges

    def build_tree(self):
        """Construit l’arbre RRT de manière dynamique et continue."""
        reached_goal = False
        for _ in range(self.max_iter):
            random_point = self.random_point()
            nearest_node = self.nearest_node(random_point)
            new_node = self.steer(nearest_node, random_point)

            # Vérifie si la nouvelle connexion est libre d'obstacles
            if self.collision_free(nearest_node, new_node):
                self.tree.append(new_node)
                self.edges.append((nearest_node, new_node))
                self.parents[tuple(new_node)] = tuple(nearest_node)
                
                # Si l'objectif est atteint, on arrête la construction
                if self.distance(new_node, self.goal) <= self.step_size:
                    self.tree.append(self.goal)
                    self.edges.append((new_node, self.goal))
                    self.parents[tuple(self.goal)] = tuple(new_node)
                    reached_goal = True

            # Ajoute l'état actuel à la vidéo
            self.capture_frame()

            # Optionnel : on peut également supprimer les nœuds inutiles ici pour garder l'arbre dynamique
            self.prune_tree()

            # Si l'objectif a été atteint, on continue à ajouter des frames pour visualiser l'atteinte
            if reached_goal:
                # Ajouter quelques images supplémentaires pour montrer que l'objectif a été atteint
                for _ in range(30):  # Ajoute quelques frames après avoir atteint l'objectif
                    self.capture_frame()

                break  # Arrêter la construction après avoir atteint l'objectif

    def capture_frame(self):
        """Capture l'état actuel du RRT et enregistre l'image dans la vidéo."""
        frame = np.ones((800, 800, 3), dtype=np.uint8) * 255  # Fond blanc
        
        # Trace les obstacles
        for (ox, oy, radius) in self.obstacles:
            cv2.circle(frame, (int(ox * 80), int(oy * 80)), int(radius * 80), (0, 0, 255), -1)

        # Trace les connexions (arêtes)
        for edge in self.edges:
            p1, p2 = edge
            cv2.line(frame, (int(p1[0] * 80), int(p1[1] * 80)), (int(p2[0] * 80), int(p2[1] * 80)), (255, 0, 0), 2)

        # Trace les nœuds
        for node in self.tree:
            cv2.circle(frame, (int(node[0] * 80), int(node[1] * 80)), 3, (0, 0, 255), -1)

        # Trace les points de départ et d’arrivée
        cv2.circle(frame, (int(self.start[0] * 80), int(self.start[1] * 80)), 5, (0, 255, 0), -1)  # Départ
        cv2.circle(frame, (int(self.goal[0] * 80), int(self.goal[1] * 80)), 5, (0, 0, 255), -1)  # Objectif

        # Ajoute le frame à la vidéo
        self.video_writer.write(frame)

    def close_video(self):
        """Ferme le fichier vidéo après avoir enregistré toutes les images."""
        self.video_writer.release()

# Paramètres
start = [0, 0]  # Point de départ
goal = [8, 8]  # Point objectif
x_range = [0, 10]
y_range = [0, 10]
obstacles = [(3, 3, 0.5), (6, 6, 0.5), (7, 2, 0.5)]  # Liste des obstacles (x, y, rayon)
step_size = 0.5
max_iter = 500
prune_threshold = 100  # Seuil pour la suppression des nœuds trop éloignés

# Exécution de l'algorithme RRT dynamique
rrt = DynamicRRT(start, goal, x_range, y_range, obstacles, step_size, max_iter, prune_threshold)
rrt.build_tree()

# Ferme le fichier vidéo après l'exécution
rrt.close_video()

print("Simulation terminée, vidéo générée.")
