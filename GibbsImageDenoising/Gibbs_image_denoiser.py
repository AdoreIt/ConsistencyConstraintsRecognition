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
        if px + nb[1] >= 0 and px + nb[1] < w and py + nb[0] >= 0 and py + nb[
                0] < h:
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
                    k_zero_weights_sum += 0 if rnd_image[nb[0],
                                                         nb[1]] == 0 else beta
                    k_one_weights_sum += 0 if rnd_image[nb[0],
                                                        nb[1]] == 1 else beta

                t = math.exp(-k_zero_weights_sum) / (
                    math.exp(-k_zero_weights_sum) +
                    math.exp(-k_one_weights_sum))
                image[y, x] = int(np.random.uniform() >= t)  # {0, 1}
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


def q_t(k, k_original, epsilon):
    """
    k - zero_or_one
    k_original - pixel from original_image
    """
    return -math.log(1 - epsilon) if k == k_original else -math.log(epsilon)


def g_tt(k, k_, beta):
    """
    k - zero_or_one
    k_ - pixel from noised_image
    """
    return 0 if k == k_ else beta


def sum_g_tt(zero_or_one, noised_image, y, x, beta):
    neighbours = get_neighbours(noised_image.shape[0], noised_image.shape[1],
                                y, x)
    edges_sum = 0
    for nb in neighbours:
        edges_sum += g_tt(zero_or_one, noised_image[nb[0], nb[1]], beta)
    # print(edges_sum)
    return edges_sum


def calc_images_changes(noised_image, denoised_image):
    # for y in range(noised_image.shape[0]):
    #     for x in range
    print(np.mean(noised_image != denoised_image))
    return np.mean(noised_image != denoised_image)


def Gibbs(original_image, noised_image, epsilon, beta, threshold):
    print("Gibbsing . . .")
    noised_image_copy = noised_image.copy()
    denoised_image = np.random.randint(2,
                                       size=(original_image.shape[0],
                                             original_image.shape[1]))
    while True:
        for y in range(noised_image.shape[0]):
            for x in range(noised_image.shape[1]):
                denoised_image[y, x] = (math.exp(
                    -q_t(0, original_image[y, x], epsilon) -
                    sum_g_tt(0, noised_image, y, x, beta))) / (
                        (math.exp(-q_t(0, original_image[y, x], epsilon) -
                                  sum_g_tt(0, noised_image, y, x, beta))) +
                        (math.exp(-q_t(1, original_image[y, x], epsilon) -
                                  sum_g_tt(1, noised_image, y, x, beta))))

        if calc_images_changes(noised_image_copy, denoised_image) < threshold:
            return denoised_image
        else:
            denoised_image, noised_image_copy = noised_image_copy, denoised_image


if __name__ == "__main__":
    # image = np.asarray(Image.open("input_image.jpg").convert('L'))
    # print(image.shape)
    # print(image)
    # image = binarize(image, 128)
    gen_iterations = 100
    epsilon = 0.5
    beta = 0.8
    img_h = 100
    img_w = 100
    threshold = 0.01

    gen_image = generate_image(img_h, img_w, beta, gen_iterations)
    noised_image = noise_image(gen_image, epsilon)
    denoised_image = Gibbs(gen_image, noised_image, epsilon, beta, threshold)

    mpl.rcParams['toolbar'] = 'None'
    figure = plt.figure()
    figure.canvas.set_window_title('Histogram creator. Image binarizator')
    specs = gridspec.GridSpec(ncols=3, nrows=1, figure=figure)

    # fig_binary_image = add_image_figure(figure, "Binarized image", specs[0, 0],
    #                                     image)

    fig_generated_image = add_image_figure(figure, "Generated image",
                                           specs[0, 0], gen_image)
    fig_noised_image = add_image_figure(figure, "Noised image", specs[0, 1],
                                        noised_image)
    fig_denoised_image = add_image_figure(figure, "Denoised image",
                                          specs[0, 2], denoised_image)

    plt.show()
