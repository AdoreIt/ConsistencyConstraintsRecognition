import numpy as np


def get_edge_index(ar, y, x, dy, dx, m1, m2):
    # dx, dy are either -1 or 1
    # if dx != 0 then dy == 0 and vice versa
    # m1, m2 are either 0 or 1

    h = ar.shape[0]
    w = ar.shape[1]

    if dx < 0:
        if x == 0:
            return None
        x -= 1
        dx = -dx
        m1, m2 = m2, m1

    if dy < 0:
        if y == 0:
            return None
        y -= 1
        dy = -dy
        m1, m2 = m2, m1

    if dx > 0 and x == w - 1:
        return None

    if dy > 0 and y == h - 1:
        return None

    b0 = m1
    b1 = m2
    b2 = 0 if dx > 0 else 1

    return x, y, (b0 << 0) | (b1 << 1) | (b2 << 2)


def get_edge_weight(ar, y, x, dy, dx, m1, m2):
    idx = get_edge_index(ar, y, x, dy, dx, m1, m2)
    if idx:
        x, y, z = idx
        return ar[y, x, z]


def set_edge_weight(ar, y, x, dy, dx, m1, m2, value):
    idx = get_edge_index(ar, y, x, dy, dx, m1, m2)
    if idx:
        x, y, z = idx
        ar[y, x, z] = value


def get_neighbours_shift(h, w, py, px):
    #           y   x
    # left:   ( 0, -1 )
    # up:     (-1,  0 )
    # right:  ( 0,  1 )
    # bottom: ( 1,  0 )

    neighbours = []
    for nb in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
        if px + nb[1] >= 0 and px + nb[1] < w and py + nb[0] >= 0 and py + nb[
                0] < h:
            neighbours.append(nb)
    return neighbours

if __name__ == "__main__":
    edges = np.full((3, 3, 8), 999)

    #           y   x
    # left:   ( 0, -1 )
    # up:     (-1,  0 )
    # right:  ( 0,  1 )
    # bottom: ( 1,  0 )

    # arr, y, x, dy, dx, m1, m2, value
    # left
    set_edge_weight(edges, 1, 1, 0, -1, 0, 0, 6)
    set_edge_weight(edges, 1, 1, 0, -1, 0, 1, 5)
    set_edge_weight(edges, 1, 1, 0, -1, 1, 0, 15)
    set_edge_weight(edges, 1, 1, 0, -1, 1, 1, 14)
    # up
    set_edge_weight(edges, 1, 1, -1, 0, 0, 0, 7)
    set_edge_weight(edges, 1, 1, -1, 0, 0, 1, 0)
    set_edge_weight(edges, 1, 1, -1, 0, 1, 0, 9)
    set_edge_weight(edges, 1, 1, -1, 0, 1, 1, 8)
    # right
    set_edge_weight(edges, 1, 1, 0, 1, 0, 0, 1)
    set_edge_weight(edges, 1, 1, 0, 1, 0, 1, 2)
    set_edge_weight(edges, 1, 1, 0, 1, 1, 0, 10)
    set_edge_weight(edges, 1, 1, 0, 1, 1, 1, 11)
    # bottom
    set_edge_weight(edges, 1, 1, 1, 0, 0, 0, 3)
    set_edge_weight(edges, 1, 1, 1, 0, 0, 1, 4)
    set_edge_weight(edges, 1, 1, 1, 0, 1, 0, 12)
    set_edge_weight(edges, 1, 1, 1, 0, 1, 1, 13)

    # left
    assert 6 == get_edge_weight(edges, 1, 1, 0, -1, 0, 0), "left 0 0"
    assert 5 == get_edge_weight(edges, 1, 1, 0, -1, 0, 1), "left 0 1"
    assert 15 == get_edge_weight(edges, 1, 1, 0, -1, 1, 0), "left 1 0"
    assert 14 == get_edge_weight(edges, 1, 1, 0, -1, 1, 1), "left 1 1"
    # up
    assert 7 == get_edge_weight(edges, 1, 1, -1, 0, 0, 0), "up 0 0"
    assert 0 == get_edge_weight(edges, 1, 1, -1, 0, 0, 1), "up 0 1"
    assert 9 == get_edge_weight(edges, 1, 1, -1, 0, 1, 0), "up 1 0"
    assert 8 == get_edge_weight(edges, 1, 1, -1, 0, 1, 1), "up 1 1"
    # right
    assert 1 == get_edge_weight(edges, 1, 1, 0, 1, 0, 0), "right 0 0"
    assert 2 == get_edge_weight(edges, 1, 1, 0, 1, 0, 1), "right 0 1"
    assert 10 == get_edge_weight(edges, 1, 1, 0, 1, 1, 0), "right 1 0"
    assert 11 == get_edge_weight(edges, 1, 1, 0, 1, 1, 1), "right 1 1"
    # bottom
    assert 3 == get_edge_weight(edges, 1, 1, 1, 0, 0, 0), "bottom 0 0"
    assert 4 == get_edge_weight(edges, 1, 1, 1, 0, 0, 1), "bottom 0 1"
    assert 12 == get_edge_weight(edges, 1, 1, 1, 0, 1, 0), "bottom 1 0"
    assert 13 == get_edge_weight(edges, 1, 1, 1, 0, 1, 1), "bottom 1 1"

    # edges = np.full((6, 6, 8), None)
    image = np.zeros((3, 3))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            print(y, x)
            nbs = get_neighbours_shift(image.shape[0], image.shape[1], y, x)
            for nb in nbs:
                print("NB: ", nb)
                print("0 0",get_edge_weight(edges, y, x, nb[0], nb[1], 0, 0))
                print("0 1", get_edge_weight(edges, y, x, nb[0], nb[1], 0, 1))
                print("1 0",get_edge_weight(edges, y, x, nb[0], nb[1], 1, 0))
                print("1 1",get_edge_weight(edges, y, x, nb[0], nb[1], 1, 1))
            print(" ")
                

