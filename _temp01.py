import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import magpylib.func as func

N = 5000
eabs = np.linspace(-3,3,31)
eacs = np.linspace(-3,3,31)


if False:
    DAT = np.zeros((len(eabs), len(eacs)))

    for i, eab in enumerate(eabs):
        for j, eac in enumerate(eacs):

            # create dimensions - longest side = 1
            ab = 10**eab
            ac = 10**eac
            dim = np.array((1, 1/ab, 1/ac))
            dim = dim / max(dim)

            # create random polarizations of length 1
            J = np.random.uniform(0, 1, (N, 3))
            J = J/np.linalg.norm(J, axis=1)[:, None]

            # create observation points at increasing distance 10 to 10**6
            cloud = np.random.uniform(-1, 1, (N, 3))
            cloud /= np.linalg.norm(cloud, axis=1)[:, None]
            distances = np.logspace(1, 6, N)
            cloud *= distances[:, None]

            B0 = func.cuboid_field(
                "B",
                observers=cloud,
                dimensions=dim,
                polarizations=J,
            )
            B1 = func.dipole_field(
                "B",
                observers=cloud,
                moments=J *  dim.prod() / magpy.mu_0,
            )

            errors = np.linalg.norm(B0-B1, axis=1)/np.linalg.norm(B1, axis=1)

            coeff = np.polyfit(np.log10(distances), np.log10(errors), deg=15)
            poly = np.poly1d(coeff)
            fit = poly(np.log10(distances))
            thresh = distances[np.argmin(fit)]

            DAT[i, j] = np.log10(thresh)

            # print(eab)
            # print(eac)
            # print(np.log10(thresh))
            # plt.plot(distances, errors, ls='', marker='.')
            # plt.plot(distances, 10**fit, 'k--')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()
            # import sys
            # sys.exit()
    xg, yg = np.meshgrid(eabs, eacs, indexing='ij')
    np.save('_temp99.npy', [xg, yg, DAT])


from time import perf_counter as pf

t0 = pf()
from scipy.interpolate import RegularGridInterpolator
t1 = pf()
xg, yg, dat = np.load('_temp99.npy')
t2 = pf()
interp = RegularGridInterpolator((eabs, eacs), dat, bounds_error=False, fill_value=None)
t3 = pf()
interp((1,2))
t4 = pf()
interp([(2,3)]*100)
t5 = pf()
interp((1,2))
t6 = pf()

print(t1 - t0)
print(t2 - t1)
print(t3 - t2)
print(t4 - t3)
print(t5 - t4)
print(t6 - t5)

# eabs = np.linspace(-3,3,31)
# eacs = np.linspace(-3,3,31)
# xg, yg = np.meshgrid(eabs, eacs, indexing='ij')
# z = interp((xg, yg))

# plt.contourf(eabs, eacs, z.T, levels=20, cmap='viridis')
# cp = plt.contour(eabs, eacs, z.T, levels=20, colors='k')
# plt.clabel(cp, inline=True, fontsize=8)

# plt.show()




# cube = magpy.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 0.5, 0.25))



# for _ in range(100):

#     dim=np.random.uniform(0, 1, 3)
#     dim = dim/max(dim)





#     distances = np.logspace(1, 6, 1000)
#     cloud *= distances[:, None]

#     B0 = cube.getB(cloud)
#     B1 = dipole.getB(cloud)
#     errors = np.linalg.norm(B0-B1, axis=1)/np.linalg.norm(B1, axis=1)

#     coeff = np.polyfit(np.log10(distances), np.log10(errors), deg=15)
#     fit = np.poly1d(coeff)

#     #plt.plot(distances, errors)
#     plt.plot(distances, 10**fit(np.log10(distances)), 'k--')


# plt.xlabel('Distance (m)')
# plt.ylabel('Relative Error')
# plt.title('Dipole VS Cube')
# plt.grid()
# plt.show()