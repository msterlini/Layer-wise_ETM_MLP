import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def ellipsoid_plot_3D(P, plot=True, ax=None, color=None, legend=None):
  """
  Plots a 3D ellipsoid defined by the positive definite matrix P.
  Parameters:
  P (numpy.ndarray): A 3x3 positive definite matrix defining the ellipsoid.
  plot (bool, optional): If True, the plot will be displayed. If False, the figure and axis will be returned. Default is True.
  ax (matplotlib.axes._subplots.Axes3DSubplot, optional): A 3D axis object to plot on. If None, a new figure and axis will be created. Default is None.
  color (str, optional): Color of the ellipsoid surface. Default is 'r'.
  legend (str, optional): Legend label for the ellipsoid. If None, no legend will be added. Default is None.
  Returns:
  matplotlib.figure.Figure, matplotlib.axes._subplots.Axes3DSubplot: If plot is False and ax is None, returns the figure and axis objects.
  matplotlib.axes._subplots.Axes3DSubplot: If plot is False and ax is provided, returns the axis object.
  None: If plot is True, displays the plot and returns None.
  """

  if color is None:
    color = 'r'

  eigvals, eigvecs = np.linalg.eigh(P)
  axis_length = 1 / np.sqrt(eigvals)
  
  phi = np.linspace(0, 2 * np.pi, 100)
  theta = np.linspace(0, np.pi, 100)
  phi, theta = np.meshgrid(phi, theta)

  x = np.sin(theta) * np.cos(phi)
  y = np.sin(theta) * np.sin(phi)
  z = np.cos(theta)
  
  unit_sphere = np.stack((x, y, z), axis=-1)
  ellipsoid_points = unit_sphere @ np.diag(axis_length) @ eigvecs.T
  
  x_ellipsoid = ellipsoid_points[:, :, 0] * 180 / np.pi
  y_ellipsoid = ellipsoid_points[:, :, 1]
  z_ellipsoid = ellipsoid_points[:, :, 2]
  
  if ax is None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    if legend:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=3, cstride=4, color=color, alpha=0.3, linewidth=0, label=legend)
    else:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=3, cstride=4, color=color, alpha=0.3, linewidth=0)

    ax.set_xlabel(r'$\theta$ (deg)', fontsize=14)
    ax.set_ylabel(r'$\dot\theta$ (rad/s)', fontsize=14)
    ax.set_zlabel('z', fontsize=14)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    if plot:
      plt.show()
    else:
      return fig, ax
  else:
    if legend:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=4, cstride=4, color='r', alpha=0.3, linewidth=0, label=legend)
    else:
      ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=4, cstride=4, color='r', alpha=0.3, linewidth=0)
    return ax

def ellipsoid_plot_2D_projections(P, plane='xy', offset=0, ax=None, color=None, legend=None):
    """
    Plots 2D projections of a 3D ellipsoid defined by the positive definite matrix P onto a specified plane.

    Parameters:
    P (numpy.ndarray): A 3x3 positive definite matrix defining the ellipsoid.
    plane (str): The plane onto which the ellipsoid is projected. Must be one of 'xy', 'xz', or 'yz'. Default is 'xy'.
    offset (float): The offset along the axis perpendicular to the projection plane. Default is 0.
    ax (matplotlib.axes._subplots.Axes3DSubplot, optional): A 3D axis object to plot on. If None, a new figure and axis are created.
    color (str, optional): The color of the ellipsoid projection. Default is 'r'.
    legend (str, optional): The legend label for the ellipsoid projection. Default is None.

    Raises:
    ValueError: If the plane parameter is not one of 'xy', 'xz', or 'yz'.

    Returns:
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if color is None:
        color = 'r'

    if plane == 'xy':
        indices = [0, 1]
        xlabel, ylabel = 'X', 'Y'
    elif plane == 'xz':
        indices = [0, 2]
        xlabel, ylabel = 'X', 'Z'
    elif plane == 'yz':
        indices = [1, 2]
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'.")

    P_sub = P[np.ix_(indices, indices)]
    eigvals, eigvecs = np.linalg.eigh(P_sub)
    axis_length = 1 / np.sqrt(eigvals)

    # Generate a unit circle and transform it into the ellipsoid's projection
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    projection = unit_circle @ np.diag(axis_length) @ eigvecs.T


    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if plane == 'xz' or plane == 'xy':
      mul = 180/np.pi
    else:
      mul = 1
    if plane == 'xy':
      verts = [list(zip(projection[:, 0] * mul, projection[:, 1], offset * np.ones_like(projection[:, 0])))]
    elif plane == 'xz':
      verts = [list(zip(projection[:, 0] * mul, offset * np.ones_like(projection[:, 0]), projection[:, 1]))]
    elif plane == 'yz':
      verts = [list(zip(offset * np.ones_like(projection[:, 0]), projection[:, 0] * mul, projection[:, 1]))]
    poly = Poly3DCollection(verts, color=color, alpha=0.3, label=legend)
    ax.add_collection3d(poly)

    if ax is None:
        plt.show()