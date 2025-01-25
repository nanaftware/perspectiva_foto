import numpy as np
from PIL import Image


# Cargar la imagen
image = Image.open("c1.png")


def create_perspective_coefs(src, dst):
    """ Calcula coeficientes para transformacion de perspectiva. """
    matrix = []
    for p1, p2 in zip(src, dst):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    a = np.array(matrix, dtype=np.float32)
    b = np.array(dst).reshape(8)
    res = np.linalg.solve(a, b)
    return res


def apply_perspective_transform(image, coefs):
    """ Aplica perspectiva transformacion a la imagen. """
    return image.transform((image.width, image.height), Image.PERSPECTIVE, coefs, Image.BICUBIC)


# Define source and destination points for the transformation
width, height = image.size
src_points = [(0, 0), (width, 0), (width, height), (0, height)]
dst_points = [
    (width * 0.4, height * 0.1),  # Top-left
    (width * 1.9, height * 0.2),  # Top-right
    (width * 1.8, height * 0.7),  # Bottom-right
    (width * -0.4, height * 0.9),  # Bottom-left
]

# Calcula el coeficientes de perspectiva 
coefs = create_perspective_coefs(src_points, dst_points)


# Aplica la transformation de perspectiva 
perspective_image = apply_perspective_transform(image, coefs)


# Savar y mostrar imagen
output_path = r"ruta"
perspective_image.save(output_path)
perspective_image.show()
