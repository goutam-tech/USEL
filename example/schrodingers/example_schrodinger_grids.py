from usel.schrodinger.grids import SpatialGrid


grid = SpatialGrid(
    xmin=-5,
    xmax=5,
    points=1000
)


print(grid.dx)
print(grid.x)