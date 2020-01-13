import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def plot(original, gaussian, laplace, salt_pepper, den_gaussian, den_laplace,
         den_sp):
    mpl.rcParams['toolbar'] = 'None'
    figure = plt.figure()

    figure.canvas.set_window_title('Image denoizer')
    specs = gridspec.GridSpec(ncols=3, nrows=3, figure=figure)

    fig_original = add_image_figure(figure, "Original image", specs[0, 1],
                                    original)
    fig_g = add_image_figure(figure, "Gaussian", specs[1, 0], gaussian)
    fig_l = add_image_figure(figure, "Laplace", specs[1, 1], laplace)
    fig_sp = add_image_figure(figure, "Salt and Pepper", specs[1, 2],
                              salt_pepper)

    fig_dg = add_image_figure(figure, "Denoised Gaussian", specs[2, 0],
                              den_gaussian)
    fig_dl = add_image_figure(figure, "Denoised Laplace", specs[2, 1],
                              den_laplace)
    fig_dsp = add_image_figure(figure, "Denoised Salt and Pepper", specs[2, 2],
                               den_sp)

    plt.show()
