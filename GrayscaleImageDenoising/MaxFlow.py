import maxflow
import numpy as np
from Edger import get_neighbours_shift, set_edge_weight, get_edge_weight


class MaxFlow():
    def __init__(self, noised_image, lamda, sigma, iterations):
        # Parameters
        self.l = lamda
        self.s = sigma
        self.iterations = iterations

        # Image
        self.noised = noised_image
        self.denoised = noised_image.copy()
        self.height, self.width = noised_image.shape

        # Alpha expansion
        self.nodes = np.zeros((self.height, self.width, 2))
        self.edges = np.zeros((self.height, self.width, 8))
        self.labels = [k for k in range(256)]
        self.kk_weight = None
        self.compute_nodes_weights()
        self.a = None

    def compute_nodes_weights(self):
        weights = np.zeros((256, 256))
        for y in range(256):
            for x in range(256):
                weights[y, x] = (y - x)**2
        self.kk_weight = weights

    def update_alpha(self):
        """ sets random alpha from current labels list and removes from it """
        self.a = np.random.choice(self.labels)
        self.labels.remove(self.a)

    def edge_weight(self, k, k_):
        return self.l * np.log((1 + self.kk_weight[k, k_]) / (2 * self.s**2))

    def set_graph_weights(self):
        self.nodes = np.zeros((self.height, self.width, 2))
        self.edges = np.zeros((self.height, self.width, 8))

        for y in range(self.height):
            for x in range(self.width):
                k = self.denoised[y, x]
                self.nodes[y, x, 0] = self.kk_weight[self.noised[y, x], k]
                self.nodes[y, x, 1] = self.kk_weight[self.noised[y, x], self.a]
                nbs = get_neighbours_shift(self.height, self.width, y, x)
                for nb in nbs:
                    k_ = self.denoised[y + nb[0], x + nb[1]]
                    # 0 - 0: k - k`
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 0, 0,
                                    self.edge_weight(k, k_))
                    # 0 - 1: k - alpha
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 0, 1,
                                    self.edge_weight(k, self.a))
                    # 1 - 0: aplha - k`
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 1, 0,
                                    self.edge_weight(self.a, k_))
                    # 1 - 1: aplha - aplha
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 1, 1,
                                    self.edge_weight(self.a, self.a))

    def null_paraller_edges(self):
        """ null parallel edges """
        for y in range(self.height):
            for x in range(self.width):
                k = self.denoised[y, x]
                nbs = get_neighbours_shift(self.height, self.width, y, x)

                for nb in nbs:
                    ny = y + nb[0]
                    nx = x + nb[1]
                    k_ = self.denoised[ny, nx]

                    a = self.edge_weight(k, k_)
                    b = self.edge_weight(self.a, k_)
                    c = self.edge_weight(k, self.a)
                    d = self.edge_weight(self.a, self.a)

                    self.nodes[y, x, 1] = d - c
                    self.nodes[ny, nx, 0] = a
                    self.nodes[ny, nx, 1] = c

                    # 0 - 0: 0
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 0, 0, 0)
                    # 0 - 1: 0
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 0, 1, 0)
                    # 1 - 0: b + c - a - d
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 1, 0,
                                    b + c - a - d)
                    # 1 - 1: 0
                    set_edge_weight(self.edges, y, x, nb[0], nb[1], 1, 1, 0)

    def aplha_max_flow(self):
        self.set_graph_weights()
        self.null_paraller_edges()

        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                # add edges
                nbs = get_neighbours_shift(self.height, self.width, y, x)
                for nb in nbs:
                    ny = y + nb[0]
                    nx = x + nb[1]
                    edge_0_1 = get_edge_weight(self.edges, y, x, nb[0], nb[1],
                                               0, 1)
                    edge_1_0 = get_edge_weight(self.edges, y, x, nb[0], nb[1],
                                               1, 0)

                    g.add_edge(nodeids[y, x], nodeids[ny, nx], edge_0_1,
                               edge_1_0)
                # add vertices
                g.add_tedge(nodeids[y, x], self.nodes[y, x, 0],
                            self.nodes[y, x, 1])

        # MaxFlow
        g.maxflow()
        segments = g.get_grid_segments(nodeids)

        # denoising
        for y in range(self.height):
            for x in range(self.width):
                if np.int_(np.logical_not(segments[y, x])) == 1:
                    self.denoised[y, x] = self.a

    def alpha_expansion(self):
        for i in range(self.iterations):
            print("Iteration: ", i + 1)

            self.labels = [k for k in range(256)]
            while len(self.labels) > 0:
                self.update_alpha()
                print("Alphas left {0}, selected value: {1}".format(
                    len(self.labels), self.a))
                self.aplha_max_flow()

        return self.denoised
