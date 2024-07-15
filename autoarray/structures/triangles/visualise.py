from matplotlib import pyplot as plt
import numpy as np


def visualize(triangles):
    """
    Visualize the triangles.
    """
    plt.figure(figsize=(8, 8))
    for triangle in triangles:
        triangle = np.append(triangle, [triangle[0]], axis=0)  # Close the triangle
        plt.plot(triangle[:, 0], triangle[:, 1], "o-")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Triangles")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
