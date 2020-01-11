import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image

def add_image_figure(figure, name, location, image):
    fig_image = figure.add_subplot(location)
    fig_image.set_title(name)
    fig_image.imshow(image, cmap=plt.get_cmap('gray'))
    fig_image.axis('off')
    return fig_image

def get_neighbours(h, w, py, px):
    """ returns array of neighbors
    [0,0,0,None]
    --------------- 
    left:   (0, -1)
    up:     (-1, 0)
    right:  (0, 1)
    bottom: (1, 0)
    """
    neighbours = []
    for nb in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
        if px + nb[1] >= 0 and px + nb[1] < w and py + nb[0] >= 0 and py + nb[0]<h:
            neighbours.append(nb)
    return neighbours

def generate_image(height, width, beta, iterations):
    # generate k from {0,1}
    rnd_image = np.random.randint(2, size=(height, width))
    image = rnd_image.copy()

    if iterations % 2 != 0:
        iterations += 1
    for iteration in range(iterations):

        for y in range(height):
            for x in range(width):
                k_zero_weights_sum = 0
                k_one_weights_sum = 0

                neighbours = get_neighbours(height, width, y, x)
                for nb in neighbours:
                    # edges weights summing
                    k_zero_weights_sum += 0 if rnd_image[nb[0], nb[1]] == 0 else beta
                    k_one_weights_sum += 0 if rnd_image[nb[0], nb[1]] == 1 else beta
               
                t = math.exp(-k_zero_weights_sum) /(math.exp(-k_zero_weights_sum) +  math.exp(-k_one_weights_sum))
                image[y, x] = int(np.random.uniform() >= t) # {0, 1}
        # after iteration swap images
        image, rnd_image = rnd_image, image

    # after all iteration return generated image
    return image

def noise_image(image, epsilon):
    noised_image = image.copy()
    for y in range(noised_image.shape[0]):
        for x in range(noised_image.shape[1]):
            if np.random.uniform() < epsilon:
                noised_image[y, x] = 1 - image[y, x]
    return noised_image



if __name__ == "__main__":
    # image = np.asarray(Image.open("input_image.jpg").convert('L'))
    # print(image.shape)
    # print(image)
    # image = binarize(image, 128)

    mpl.rcParams['toolbar'] = 'None'
    figure = plt.figure()
    figure.canvas.set_window_title('Histogram creator. Image binarizator')
    specs = gridspec.GridSpec(ncols=3, nrows=1, figure=figure)

    # fig_binary_image = add_image_figure(figure, "Binarized image", specs[0, 0],
    #                                     image)
    gen_image = generate_image(100, 100, 1, 100)
    noised_image = noise_image(gen_image, 0.5)
    fig_generated_image = add_image_figure(figure, "Generated image", specs[0, 0],
                                        gen_image)
    fig_noised_image = add_image_figure(figure, "Noised image", specs[0, 1],
                                        noised_image)

    plt.show()
