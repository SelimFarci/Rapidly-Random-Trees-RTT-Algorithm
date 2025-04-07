# RRT_algorithm_with_obstacles.py
import numpy as np
import matplotlib.pyplot as plt

class RRT:
    def __init__(self, start, goal, x_range, y_range, z_value, step_size=0.5, max_iter=100, obstacles=[]):
        """
        Initialisation des paramètres pour l'algorithme RRT avec obstacles
        """
        self.start = np.array(start)  # Point de départ en 2D
        self.goal = np.array(goal)    # Point objectif en 2D
        self.z_value = z_value       # La troisième dimension est fixe
        self.x_range = x_range       # Plage des coordonnées x
        self.y_range = y_range       # Plage des coordonnées y
        self.step_size = step_size   # Taille du pas dans l'algorithme RRT
        self.max_iter = max_iter     # Nombre maximum d'itérations
        self.tree = [self.start]     # Initialise l'arbre avec le point de départ
        self.edges = []              # Liste des connexions entre les nœuds
        self.parents = {tuple(self.start): None}  # Stocke les parents de chaque nœud
        self.obstacles = obstacles   # Liste des obstacles

    def distance(self, point1, point2):
        """Calcule la distance entre deux points 2D"""
        return np.linalg.norm(point1 - point2)

    def random_point(self):
        """Génère un point aléatoire dans l'espace cartésien 2D."""
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

    def check_collision(self, point):
        """Vérifie si un point entre en collision avec des obstacles."""
        for (obstacle_center, obstacle_radius) in self.obstacles:
            if np.linalg.norm(point - obstacle_center) < obstacle_radius:
                return True  # Le point est en collision avec un obstacle
        return False

    def check_edge_collision(self, from_node, to_node):
        """Vérifie si l'arête entre deux nœuds entre en collision avec des obstacles."""
        num_steps = int(self.distance(from_node, to_node) / self.step_size)
        for i in range(num_steps + 1):
            intermediate_point = from_node + i * (to_node - from_node) / num_steps
            if self.check_collision(intermediate_point):
                return True  # Collision détectée
        return False

    def build_tree(self):
        """Construit l’arbre RRT en évitant les obstacles."""
        for _ in range(self.max_iter):
            random_point = self.random_point()
            nearest_node = self.nearest_node(random_point)
            new_node = self.steer(nearest_node, random_point)

            # Vérifie si le nouveau nœud est dans un obstacle
            if not self.check_collision(new_node):
                # Vérifie s'il n'y a pas de collision avec des obstacles entre les nœuds
                if not self.check_edge_collision(nearest_node, new_node):
                    self.tree.append(new_node)
                    self.edges.append((nearest_node, new_node))
                    self.parents[tuple(new_node)] = tuple(nearest_node)

                    # Vérifie si on a atteint l'objectif
                    if self.distance(new_node, self.goal) <= self.step_size:
                        self.tree.append(self.goal)
                        self.edges.append((new_node, self.goal))
                        self.parents[tuple(self.goal)] = tuple(new_node)  # Stocke le parent de l'objectif
                        break

    def reconstruct_path(self):
        """Reconstruit le chemin de l’objectif au départ."""
        path = []
        current = tuple(self.goal)
        while current is not None:
            path.append(np.array(current))
            current = self.parents.get(current)
        path.reverse()  # Inverse le chemin pour partir du départ
        return path

    def plot(self):
        """Affiche l’arbre, les obstacles, et la trajectoire spécifique en rouge."""
        plt.figure(figsize=(8, 8))
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)

        # Trace les obstacles (cercles)
        for (obstacle_center, obstacle_radius) in self.obstacles:
            obstacle = plt.Circle(obstacle_center, obstacle_radius, color='gray', alpha=0.5)
            plt.gca().add_artist(obstacle)

        # Trace les connexions (arêtes) en bleu
        for edge in self.edges:
            p1, p2 = edge
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=0.5)

        # Trace les nœuds
        for node in self.tree:
            plt.plot(node[0], node[1], 'bo', markersize=3)

        # Points de départ et d’arrivée
        plt.plot(self.start[0], self.start[1], 'go', label="Départ")  # Départ
        plt.plot(self.goal[0], self.goal[1], 'ro', label="Objectif")  # Objectif

        # Reconstruit et surligne le chemin atteignant l'objectif en rouge
        path = self.reconstruct_path()
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, label="Trajectoire atteignant l'objectif" if i == 0 else "")

        # Affiche les légendes et les options de tracé
        plt.legend()
        plt.grid(True)
        plt.title("Arbre RRT avec obstacles")
        plt.show()
