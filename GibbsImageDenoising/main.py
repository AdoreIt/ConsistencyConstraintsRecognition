import numpy as np


class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = [[Pixel(np.random.randint(2))
                       for j in range(width)] for i in range(height)]

    def show(self):
        """ showing image """

    def at(self, y_h_i, x_w_j):
        return self.image[y_h_i][x_w_j]

    def at_value(self, y_h_i, x_w_j):
        return self.image[y_h_i][x_w_j].label

    def print(self):
        for row in range(self.height):
            for col in range(self.width):
                print(self.at_value(row, col))
            print("\n")


class Pixel:
    def __init__(self, label):
        # k = 0 - black
        # k = 1 - white
        # top, bottom, right, left
        self.black_edges_weights = [None, None, None, None]  # k = 0
        self.white_edges_weights = [None, None, None, None]  # k = 1

        self.label = label  # label = color = 0 or 1
        self.weight = None


# def generate_image(width, height):
#     for col in range(width):  # i
#         for row in range(height):  # j
#             Image.
#             if col == 0:
#                 print("left_most_col")
#             if col == width - 1:
#                 print("right_most_col")
#             if row == 0:
#                 print("top_most_row")
#             if row == height - 1:
#                 print("bottom_most_row")

if __name__ == "__main__":
    image = Image(5, 2)
    image.print()
