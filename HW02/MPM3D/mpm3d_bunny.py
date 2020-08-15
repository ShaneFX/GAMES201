import taichi as ti
import numpy as np
from plyImporter import PlyImporter


ti.init(arch=ti.gpu)

ply3 = PlyImporter("bunny.ply")
dim = 3
n_particles = ply3.get_count()
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2.0e-4
p_vol = (dx * 0.5)**3
p_rho = 1
p_mass = p_vol * p_rho
E = 400

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
J = ti.var(dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid, n_grid))
img = ti.Vector(2, dt=ti.f32, shape=n_particles)


@ti.kernel
def substep():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
        affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]]) + p_mass * C[p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (offset.cast(float) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    grid_v[base + offset].atomic_add(
                        weight * (p_mass * v[p] + affine @ dpos))
                    grid_m[base + offset].atomic_add(weight * p_mass)

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            bound = 3
            inv_m = 1 / grid_m[i, j, k]
            grid_v[i, j, k] = inv_m * grid_v[i, j, k]
            grid_v[i, j, k][1] -= dt * 9.8
            if i < bound and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0
            if i > n_grid - bound and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
            if j < bound and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0
            if j > n_grid - bound and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
            if k < bound and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if k > n_grid - bound and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [
            0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2
        ]
        new_v = ti.Vector.zero(ti.f32, 3)
        new_C = ti.Matrix.zero(ti.f32, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.Vector([i, j, k]).cast(float) - fx
                    g_v = grid_v[base + ti.Vector([i, j, k])]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def to_img():
    for i in img:
        img[i] = ti.Vector([x[i][0], x[i][1]])


def save_ply(frame1):
    series_prefix = "bunny.ply"
    num_vertices = n_particles
    np_pos = np.reshape(x.to_numpy(), (num_vertices, 3))
    writer = ti.PLYWriter(num_vertices=num_vertices)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.export_frame_ascii(frame1, series_prefix)


def initial_state():
    x.from_numpy(ply3.get_array())
    J.fill(1)


initial_state()
for frame in range(100):
    for s in range(200):
        grid_v.fill([0, 0, 0])
        grid_m.fill(0)
        substep()

    # save_ply(frame)
