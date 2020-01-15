from PIL import Image

from Plotter import plot
from Noiser import *
from MaxFlow import MaxFlow

if __name__ == "__main__":
    input_image_path = "Images/flower_100.jpg"

    # Parameters
    lamda = 2.
    sigma = 10.
    number_of_iterations = 1

    #-- gaussian noise
    loc_gaussian = 0.
    scale_gaussian = 10.

    #-- laplacian noise
    loc_laplace = 0.
    scale_laplace = 10.

    #-- salt and pepper noise
    sp_probability = 0.05

    image = np.asarray(Image.open(input_image_path).convert('L'))

    noised_gaussian = noise_gaussian(image, loc_gaussian, scale_gaussian)
    noised_laplace = noise_laplace(image, loc_laplace, scale_laplace)
    noised_salt_and_pepper = noise_salt_and_pepper(image, sp_probability)

    # max_flow = MaxFlow(noised_gaussian, lamda, sigma, number_of_iterations)
    # den_gaussian = max_flow.alpha_expansion()

    # max_flow = MaxFlow(noised_laplace, lamda, sigma, number_of_iterations)
    # den_laplace = max_flow.alpha_expansion()

    max_flow = MaxFlow(noised_gaussian, lamda, sigma, number_of_iterations)
    den_sp = max_flow.alpha_expansion()

    # Plot results
    plot(image, noised_gaussian, noised_laplace, noised_salt_and_pepper,
         den_sp, den_sp, den_sp)
    # plot(image, noised_gaussian, noised_laplace, noised_salt_and_pepper,
    #      den_gaussian, den_laplace, den_sp)
