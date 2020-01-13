from math import exp, log
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image

import maxflow


def add_image_figure(figure, name, location, image):
    fig_image = figure.add_subplot(location)
    fig_image.set_title(name)
    fig_image.imshow(image, cmap=plt.get_cmap('bone'))
    fig_image.axis('off')
    return fig_image


def binarizate(image, threshold):
    binarized_image = image.copy()

    for h_pixel in range(binarized_image.shape[0]):
        for w_pixel in range(binarized_image.shape[1]):
            if binarized_image[h_pixel, w_pixel] >= threshold:
                binarized_image[h_pixel, w_pixel] = 1
            else:
                binarized_image[h_pixel, w_pixel] = 0

    return binarized_image


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
            neighbours.append((py + nb[0], px + nb[1]))
    return neighbours


def generate_image(height, width, beta, iterations):
    # generate k from {0,1}
    rnd_image = np.random.randint(2, size=(height, width))

    if iterations % 2 != 0:
        iterations += 1
    for iteration in range(iterations):
        for y in range(height):
            for x in range(width):
                k_zero_weights_sum = sum_g_tt(0, rnd_image, y, x, beta)
                k_one_weights_sum = sum_g_tt(1, rnd_image, y, x, beta)

                t = exp(-k_zero_weights_sum) / (exp(-k_zero_weights_sum) +
                                                exp(-k_one_weights_sum))
                rnd_image[y, x] = int(np.random.uniform() >= t)  # {0, 1}

    # after all iteration return generated image
    return rnd_image


def noise_image(image, epsilon):
    noised_image = image.copy()
    for y in range(noised_image.shape[0]):
        for x in range(noised_image.shape[1]):
            if np.random.uniform() < epsilon:
                noised_image[y, x] = 1 - image[y, x]
    return noised_image


def q_t(k, k_, epsilon):
    """
    k: 0 or 1
    k_: pixel value
    """
    return -log(1 - epsilon) if k == k_ else -log(epsilon)


def g_tt(k, k_, beta):
    """
    k: 0 or 1
    k_: pixel value
    """
    return 0 if k == k_ else beta


def sum_g_tt(zero_or_one, image, y, x, beta):
    neighbours = get_neighbours(image.shape[0], image.shape[1], y, x)
    edges_sum = 0
    for nb in neighbours:
        edges_sum += g_tt(zero_or_one, image[nb[0], nb[1]], beta)
    return edges_sum


def calc_images_changes(noised_image, denoised_image):
    return np.mean(noised_image != denoised_image, dtype=np.float64)


def most_probable_image(zeros_count, ones_count):
    h = zeros_count.shape[0]
    w = zeros_count.shape[1]
    result_image = np.zeros((h, w), dtype=int)

    for y in range(h):
        for x in range(w):
            if zeros_count[y, x] > ones_count[y, x]:
                result_image[y, x] = 0
            else:
                result_image[y, x] = 1
    return result_image


def Gibbs(original_image, noised_image, epsilon, beta, threshold):
    print("\n--- Gibbs sampling denoising . . .")
    denoised_image = np.random.randint(2,
                                       size=(original_image.shape[0],
                                             original_image.shape[1]))
    denoised_image_prev = denoised_image.copy()

    # Sum count of zeros and ones after each 2nd iteration
    zeros_count = np.zeros((noised_image.shape[0], noised_image.shape[1]),
                           dtype=int)
    ones_count = np.zeros((noised_image.shape[0], noised_image.shape[1]),
                          dtype=int)

    iteration = 0
    # Gibbsing
    while True:
        iteration += 1
        for y in range(noised_image.shape[0]):
            for x in range(noised_image.shape[1]):
                zero_q = q_t(0, noised_image[y, x], epsilon)
                one_q = q_t(1, noised_image[y, x], epsilon)

                zero_edges_sum = sum_g_tt(0, noised_image, y, x, beta)
                one_edges_sum = sum_g_tt(1, noised_image, y, x, beta)

                t = exp(-zero_q -
                        zero_edges_sum) / (exp(-zero_q - zero_edges_sum) +
                                           exp(-one_q - one_edges_sum))

                denoised_image[y, x] = int(np.random.uniform() >= t)

                if iteration > 5 and iteration % 2 == 0:
                    if denoised_image[y, x] == 0:
                        zeros_count[y, x] += 1
                    else:
                        ones_count[y, x] += 1

        if calc_images_changes(denoised_image_prev,
                               denoised_image) < threshold:
            return most_probable_image(zeros_count, ones_count)

        else:
            if iteration % 100 == 0:
                print("iteration {0}: {1}".format(
                    iteration,
                    calc_images_changes(denoised_image_prev, denoised_image)))

            if iteration % 1000 == 0:
                result_image_tmp = most_probable_image(zeros_count, ones_count)
                error_tmp = calc_images_changes(denoised_image_prev,
                                                denoised_image)

                print("Saving iteration_{0}__{1}.png".format(
                    iteration, error_tmp))
                plt.imsave("iteration_{0}__{1}.png".format(
                    iteration, error_tmp, result_image_tmp),
                           cmap=mpl.cm.gray)
                result_image_tmp = None

            denoised_image_prev = denoised_image.copy()


def MaxFlow(noised_image, epsilon, beta):
    print("\n--- MaxFlow denoising . . .")
    g = maxflow.Graph[float]()
    height, width = noised_image.shape
    nodeids = g.add_grid_nodes((height, width))
    for y in range(height):
        for x in range(width):
            # add edges
            nbs = get_neighbours(height, width, y, x)
            for nb in nbs:
                edge_weight = g_tt(noised_image[y, x],
                                   noised_image[nb[0], nb[1]], beta)
                g.add_edge(nodeids[y, x], nodeids[nb[0], [1]], edge_weight,
                           edge_weight)
            # add vertices
            g.add_tedge(nodeids[y, x], q_t(0, noised_image[y, x], epsilon),
                        q_t(1, noised_image[y, x], epsilon))

    # MaxFlow
    g.maxflow()
    segments = g.get_grid_segments(nodeids)
    denoised_maxflow = np.int_(np.logical_not(segments))
    return denoised_maxflow


if __name__ == "__main__":
    # Read from file or generate image
    read_from_file = True

    input_image_path = "bold_tree_300x300.png"
    image = binarizate(np.asarray(Image.open(input_image_path).convert('L')),
                       128)

    # Generation parameters
    generation_iterations = 10000
    image_height = 10
    image_width = 10

    # Denoising parameters
    epsilon = 0.09
    beta = 0.9
    threshold = 0.05

    if read_from_file:
        gen_image = image
    else:
        gen_image = generate_image(image_height, image_width, beta,
                                   generation_iterations)

    noised_image = noise_image(gen_image, epsilon)

    print("Saving input image...")
    plt.imsave("binary_bold_tree_300x300.png", gen_image, cmap=mpl.cm.bone)
    print("Saving noised image...")
    plt.imsave("noised_image.png", noised_image, cmap=mpl.cm.bone)

    denoised_maxflow = MaxFlow(noised_image, epsilon, beta)
    print("Saving denoised with MaxFlow image...")
    plt.imsave("denoised_MaxFlow_MaxFlow.png",
               denoised_maxflow,
               cmap=mpl.cm.bone)

    denoised_gibbs = Gibbs(gen_image, noised_image, epsilon, beta, threshold)
    print("Saving denoised with Gibbs sampler image...")
    plt.imsave("denoised_Gibbs.png", denoised_gibbs, cmap=mpl.cm.bone)

    # Plotting resutsMaxFlow
    print("\n\nPlotting results... ")
    mpl.rcParams['toolbar'] = 'None'
    figure = plt.figure()
    figure.canvas.set_window_title('Histogram creator. Image binarizator')
    specs = gridspec.GridSpec(ncols=4, nrows=1, figure=figure)

    fig_generated_image = add_image_figure(figure, "Read or generated image",
                                           specs[0, 0], gen_image)
    fig_noised_image = add_image_figure(figure, "Noised", specs[0, 1],
                                        noised_image)
    fig_denoised_image = add_image_figure(figure, "Gibbs", specs[0, 2],
                                          denoised_gibbs)
    fig_denoised_maxflow = add_image_figure(figure, "MaxFlow", specs[0, 3],
                                            denoised_maxflow)

    plt.show()
