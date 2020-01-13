from PIL import Image

from Plotter import plot
from Noiser import *
# from MaxFlow import MaxFlow

if __name__ == "__main__":
    input_image_path = "Viktor100.jpg"

    image = np.asarray(Image.open(input_image_path).convert('L'))

    # Plot results
    plot(image, image, image, image, image, image, image)
