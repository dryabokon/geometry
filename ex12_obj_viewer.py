import pyvista
# ----------------------------------------------------------------------------------------------------------------------
def example_export(filename_in,filename_out):
    mesh = pyvista.read(filename_in)

    plotter = pyvista.BackgroundPlotter()
    plotter.add_mesh(mesh)
    plotter.export_obj(filename_out)
    return
# ----------------------------------------------------------------------------------------------------------------------
def example_plot(filename_in):
    mesh = pyvista.read(filename_in)
    mesh.plot()
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    #example_export('./images/ex_GL/face/male_head.obj','./images/ex_GL/face/male_head_exp')
    example_plot('./images/ex_GL/rock/TheRock2.obj')
