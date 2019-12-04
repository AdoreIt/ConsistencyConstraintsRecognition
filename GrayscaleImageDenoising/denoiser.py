import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


def rgb2grayscale(image):
    r_coeff = 0.36
    g_coeff = 0.53
    b_coeff = 0.11

    if len(image.shape) > 2:
        return np.dot(image[..., :3],
                      [r_coeff, g_coeff, b_coeff]).round().astype(int)
    else:
        return image


def add_image_figure(figure, name, location, image):
    fig_image = figure.add_subplot(location)
    fig_image.set_title(name)
    fig_image.imshow(image, cmap=plt.get_cmap('gray'))
    # fig_image.axis('off')
    fig_image.set_yticks([])
    fig_image.set_yticklabels([])
    fig_image.set_xticks([])
    fig_image.set_xticklabels([])
    return fig_image


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Histogram creator. Image binarizator')
    argparser.add_argument(
        '-i',
        '--input_image_path',
        type=str,
        help="path to input image",
        default="input_image.jpg")

    args = argparser.parse_args()

    image = mpimg.imread(args.input_image_path)
    gray_image = rgb2grayscale(image)

    figure = plt.figure()

    figure.canvas.set_window_title('Image denoizer')
    specs = gridspec.GridSpec(ncols=3, nrows=1, figure=figure)

    fig_input_image = add_image_figure(figure, "Original image", specs[0, 0],
                                       gray_image)
    fig_noised_image = add_image_figure(figure, "Noised image", specs[0, 1],
                                        gray_image)
    fig_denoised_image = add_image_figure(figure, "Denoised image", specs[0, 2],
                                          gray_image)

    plt.show()
