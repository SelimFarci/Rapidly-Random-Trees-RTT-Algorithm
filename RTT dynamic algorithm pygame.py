import pygame
import numpy as np
import random

# Classe DynamicRRT
class DynamicRRT:
    def __init__(self, start, goal, x_range, y_range, obstacles, step_size=0.5, max_iter=100, prune_freq=10):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.x_range = x_range
        self.y_range = y_range
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.prune_freq = prune_freq  # Fréquence de suppression des anciens nœuds
        self.tree = [self.start]  # Initialise l'arbre avec le point de départ
        self.edges = []  # Liste des connexions entre les nœuds
        self.parents = {tuple(self.start): None}  # Stocke les parents de chaque nœud
        self.clock = pygame.time.Clock()  # Horloge pour contrôler la vitesse de la simulation

    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def random_point(self):
        """Génère un point aléatoire dans l'espace cartésien défini."""
        x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = np.random.uniform(self.y_range[0], self.y_range[1])
        return np.array([x, y])

    def nearest_node(self, random_point):
        """Trouve le nœud le plus proche dans l’arbre."""
        if len(self.tree) == 0:
            return None  # Si l'arbre est vide, retourner None
        distances = [self.distance(node, random_point) for node in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]

    def steer(self, from_node, to_point):
        """Crée un nouveau nœud dans la direction du point donné."""
        if from_node is None:
            return None  # Si from_node est None, retourner None
        
        direction = to_point - from_node
        distance = np.linalg.norm(direction)
        direction = direction / distance
        step = min(self.step_size, distance)
        new_node = from_node + direction * step
        return new_node

    def collision_free(self, point1, point2):
        """Vérifie si la trajectoire entre point1 et point2 est libre d'obstacles."""
        for (ox, oy, radius) in self.obstacles:
            dx, dy = point2 - point1
            dist = np.linalg.norm([dx, dy])
            steps = int(dist / self.step_size)  # Plus de points pour vérifier la collision
            for i in range(steps):
                x = point1[0] + (i / steps) * dx
                y = point1[1] + (i / steps) * dy
                if np.linalg.norm([x - ox, y - oy]) <= radius:
                    return False  # Collision détectée
        return True

    def prune_tree(self):
        """Prune les nœuds trop anciens (racines) à une certaine fréquence."""
        if len(self.tree) > self.prune_freq:
            # Supprimer les anciens nœuds
            removed_node = self.tree.pop(0)
            self.edges = [edge for edge in self.edges if not np.array_equal(removed_node, edge[0]) and not np.array_equal(removed_node, edge[1])]
            del self.parents[tuple(removed_node)]  # Supprimer l'entrée du parent pour removed_node

    def build_tree(self, screen):
        """Construit l’arbre RRT de manière dynamique et continue."""
        reached_goal = False
        for _ in range(self.max_iter):
            random_point = self.random_point()
            nearest_node = self.nearest_node(random_point)

            if nearest_node is None:
                continue  # Si nearest_node est None, on saute cette itération

            new_node = self.steer(nearest_node, random_point)

            # Vérifier si la nouvelle connexion est libre d'obstacles
            if self.collision_free(nearest_node, new_node):
                self.tree.append(new_node)
                self.edges.append((nearest_node, new_node))
                self.parents[tuple(new_node)] = tuple(nearest_node)

                # Si l'objectif est atteint
                if self.distance(new_node, self.goal) <= self.step_size:
                    self.tree.append(self.goal)
                    self.edges.append((new_node, self.goal))
                    self.parents[tuple(self.goal)] = tuple(new_node)
                    reached_goal = True

            self.prune_tree()  # Élaguer l'arbre à chaque itération

            self.render(screen)  # Rendre l'état actuel sur l'écran
            pygame.display.flip()  # Mettre à jour l'affichage
            self.clock.tick(30)  # Limiter la vitesse de la simulation

            # Si l'objectif a été atteint, on ajoute quelques frames supplémentaires
            if reached_goal:
                for _ in range(30):
                    self.render(screen)
                    pygame.display.flip()
                    self.clock.tick(30)
                break  # Terminer la simulation une fois l'objectif atteint

    def render(self, screen):
        """Affiche l'état actuel de l'arbre et des obstacles."""
        screen.fill((255, 255, 255))  # Fond blanc

        # Afficher les obstacles
        for (ox, oy, radius) in self.obstacles:
            pygame.draw.circle(screen, (255, 0, 0), (int(ox * 80), int(oy * 80)), int(radius * 80))

        # Afficher les connexions entre les nœuds
        for edge in self.edges:
            p1, p2 = edge
            pygame.draw.line(screen, (0, 0, 255), (int(p1[0] * 80), int(p1[1] * 80)), (int(p2[0] * 80), int(p2[1] * 80)), 2)

        # Afficher les nœuds
        for node in self.tree:
            pygame.draw.circle(screen, (0, 0, 255), (int(node[0] * 80), int(node[1] * 80)), 3)

        # Afficher le point de départ et d'arrivée
        pygame.draw.circle(screen, (0, 255, 0), (int(self.start[0] * 80), int(self.start[1] * 80)), 5)  # Départ
        pygame.draw.circle(screen, (255, 0, 0), (int(self.goal[0] * 80), int(self.goal[1] * 80)), 5)  # Objectif

        # Afficher le chemin trouvé (si trouvé)
        if any(np.array_equal(self.goal, node) for node in self.tree):
            path = [self.goal]
            current = tuple(self.goal)
            while current != tuple(self.start):
                current = self.parents[current]
                path.append(np.array(current))
            path.reverse()
            for i in range(1, len(path)):
                pygame.draw.line(screen, (255, 0, 0), (int(path[i-1][0] * 80), int(path[i-1][1] * 80)), 
                                 (int(path[i][0] * 80), int(path[i][1] * 80)), 2)

# Fonction principale
def main():
    # Initialiser pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("RRT Dynamique")

    # Paramètres
    start = [0, 0]  # Point de départ
    goal = [8, 8]  # Point objectif
    x_range = [0, 10]
    y_range = [0, 10]
    obstacles = [(3, 3, 0.5), (6, 6, 0.5), (7, 2, 0.5)]  # Liste des obstacles (x, y, rayon)
    step_size = 0.2  # Réduire le pas pour une meilleure résolution
    max_iter = 500
    prune_freq = 20  # Fréquence de suppression des anciens nœuds

    # Créer une instance de l'algorithme RRT dynamique
    rrt = DynamicRRT(start, goal, x_range, y_range, obstacles, step_size, max_iter, prune_freq)
    
    # Lancer la construction de l'arbre
    rrt.build_tree(screen)

    pygame.quit()

# Exécution de la fonction principale
if __name__ == "__main__":
    main()
