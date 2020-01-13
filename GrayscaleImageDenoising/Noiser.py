import numpy as np

def noise_gaussian(image, loc=0.0, scale=1.0):
    noised_image = image + np.random.normal(loc, scale, image.shape)
    return np.array(clip(noised_image, 0, 255), dtype=int)

def noise_laplace(image, loc=0.0, scale=1.0):
    noised_image = image + np.random.laplace(loc, scale, image.shape)
    return np.array(clip(noised_image, 0, 255), dtype=int)

def noise_salt_and_pepper(image, probability):
    noised_image = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            rnd_uniform = np.random.random()

            if rnd_uniform < probability:
                noised_image[y][x] = 0

            elif rnd_uniform > 1 - probability:
                noised_image[y][x] = 255

    return noised_image
