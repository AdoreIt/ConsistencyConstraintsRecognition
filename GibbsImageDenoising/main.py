class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = []

    def show(self):
        """ showing image """


class Pixel:
    def __init__(self):
        # edges
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None

        self.label = None  # label - color
        self.weight = None


class Edge:
    def __init__(self):
        self.weight = None


def generate_image(width, height):
    for col in range(width):
        for row in range(height):
            if col == 0:
                print("left_most_col")
            if col == width - 1:
                print("right_most_col")
            if row == 0:
                print("top_most_row")
            if row == height - 1:
                print("bottom_most_row")
