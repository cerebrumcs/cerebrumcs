'''
Created on 11.05.2018
'''

import random
import numpy as np
from heapq import *
from clustering.distances import * 
from _heapq import heappush

class KMeans:
    
    def __init__(self, n_clusters):
        self.dist = self._eucl_dist
        self.n_clusters = n_clusters

    def _eucl_dist(self, x1,x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))


    def assign_to_clusters(self, X, centers):
        x_arr = np.array(X)
        c_arr = np.array(centers)
        distances = np.array([[self.dist(xa,ca) for ca in c_arr] for xa in x_arr])
        clusters = np.argsort(distances, axis = 1)[:,0]
        return clusters

    
    def update_centers(self, X, clusters):
        centers = [np.mean(X[np.where(clusters == c)], axis = 0) for c in range(self.n_clusters)]
        centers = np.array(centers)
        return centers
        
    
    def fit_predict(self, X):
        centers = random.choices(X, k = self.n_clusters)
        
        # repeat until the clusters do not change
        clusters, prev_clusters = None, None
        while True:
            prev_clusters = clusters
            clusters = self.assign_to_clusters(X, centers)
            centers  = self.update_centers(X, clusters)
                
            if np.array_equal(clusters, prev_clusters):
                break
            
        return clusters


class OPTICS:

    def __init__(self, minpts, epsilon):
        self.minpts = minpts
        self.eps = epsilon
    

    def get_core_distance(self, distances):
        # get all neighbors located in the core-distance-neighborhood
        core_distance = distances[np.argsort(distances)[self.minpts]]
        core_distance = core_distance if core_distance <= self.eps else np.Infinity
        
        return core_distance


    def process_point(self, point_id, X, density_connected_points, reachability_distances, unvisited_points):
        point = X[point_id]
        # determine pairwise distances from point to get all neighbors located in the epsilon-neighborhood
        distances = np.sqrt(np.sum(np.square(X - point), axis = 1))    
        eps_neighbors_idx  = np.where(distances <= self.eps)[0]
        eps_neighbors_dist = distances[eps_neighbors_idx]
        
        # get all neighbors located in the core-distance-neighborhood
        core_distance = self.get_core_distance(distances)
        point_is_core_point = True if not core_distance is np.Infinity else False
        
        if point_is_core_point:
            # core_neighbors_idx = np.where(distances <= core_distance)
            eps_neighbors = zip(eps_neighbors_idx, eps_neighbors_dist)
            self.update_dense_points(eps_neighbors, core_distance, density_connected_points, reachability_distances, unvisited_points)


    def update_dense_points(self, eps_neighbors, core_distance, density_connected_points, reachability_distances, unvisited_points):
                    
        for eps_neighbor, dist_neighbor in eps_neighbors:
                            
            if eps_neighbor in unvisited_points:
                                
                reachability_distance = max(core_distance, dist_neighbor)
                                 
                # 1. remove from queue 
                if (reachability_distances[eps_neighbor], eps_neighbor) in density_connected_points:                                        
                    density_connected_points.remove((reachability_distances[eps_neighbor], eps_neighbor))
                    
                # 2. update best distance
                best_distance = min(reachability_distances[eps_neighbor], reachability_distance)
                
                # 3. update priority queue and lookup table                   
                heappush(density_connected_points, (best_distance, eps_neighbor))
                reachability_distances[eps_neighbor] = best_distance
    
    
    def fit_predict(self, X):
        
        visited_points_idx, unvisited_points_idx = [], list(range(len(X)))
        
        density_connected_points = []         
        reachability_distances =  {id : np.Infinity for id in unvisited_points_idx}
        
        while len(unvisited_points_idx) > 0:
            # choose point randomly
            point_id = random.choice(unvisited_points_idx)
            visited_points_idx.append(point_id)
            unvisited_points_idx.remove(point_id)
            self.process_point(point_id, X, density_connected_points, reachability_distances, unvisited_points_idx)
            
            while len(density_connected_points) > 0:
                reachability_dist, point_id = heappop(density_connected_points)
                visited_points_idx.append(point_id)  
                unvisited_points_idx.remove(point_id)
                self.process_point(point_id, X, density_connected_points, reachability_distances, unvisited_points_idx)
                
        # output list of pairs (point, min reachability), where every output[i] was processed before output[i+1]            
        output = [(vpi, reachability_distances[vpi]) for vpi in visited_points_idx]
        return output  