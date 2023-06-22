import unittest
import pytest
import numpy as np

from scipy.constants import mu_0, epsilon_0
from geoana.em import fdem
import discretize

# from SimPEG.EM import FDEM
# from SimPEG import Maps


def E_from_EDWS(
    XYZ, srcLoc, sig, f, current=1., length=1., orientation='X', kappa=0.,
    epsr=1., t=0.
):
    """E_from_EDWS
    Computing the analytic electric fields (E) from an electrical dipole in
    a wholespace
    - You have the option of computing E for multiple frequencies at a single
    reciever location
      or a single frequency at multiple locations

    :param numpy.ndarray XYZ: reciever locations at which to evaluate E
    :param float epsr: relative permitivitty value (unitless),  default is 1.0
    :rtype: numpy.ndarray
    :return: Ex, Ey, Ez: arrays containing all 3 components of E evaluated at the specified locations and frequencies.
    """

    mu = mu_0*(1+kappa)
    epsilon = epsilon_0*epsr
    sig_hat = sig + 1j*fdem.omega(f)*epsilon

    XYZ = discretize.utils.as_array_n_by_dim(XYZ, 3)

    dx = XYZ[:, 0] - srcLoc[0]
    dy = XYZ[:, 1] - srcLoc[1]
    dz = XYZ[:, 2] - srcLoc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*pi*f*mu*sig )
    k = np.sqrt(
        fdem.omega(f)**2. * mu * epsilon - 1j * fdem.omega(f) * mu * sig
    )

    front = current * length / (4.*np.pi*sig_hat * r**3) * np.exp(-1j*k*r)
    mid = -k**2 * r**2 + 3*1j*k*r + 3

    if orientation.upper() == 'X':
        Ex = front*((dx**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r-1.))
        Ey = front*(dx*dy / r**2)*mid
        Ez = front*(dx*dz / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Ey = front*((dy**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r-1.))
        Ez = front*(dy*dz / r**2)*mid
        Ex = front*(dy*dx / r**2)*mid
        return Ex, Ey, Ez

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Ez = front*((dz**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r-1.))
        Ex = front*(dz*dx / r**2)*mid
        Ey = front*(dz*dy / r**2)*mid
        return Ex, Ey, Ez


class TestFDEMdipole(unittest.TestCase):

    def test_defaults(self):
        frequency = 1
        edws = fdem.ElectricDipoleWholeSpace(frequency)
        assert(edws.sigma == 1)
        assert(edws.mu == mu_0)
        assert(edws.epsilon == epsilon_0)
        assert(np.all(edws.orientation == np.r_[1., 0., 0.]))
        assert(edws.length == 1.)
        assert(np.all(edws.location == np.r_[0., 0., 0.]))
        assert(edws.frequency == 1.)

    def test_errors(self):
        edws = fdem.ElectricDipoleWholeSpace(1)
        with pytest.raises(TypeError):
            edws.current = "box"
        with pytest.raises(TypeError):
            edws.length = "box"
        with pytest.raises(ValueError):
            edws.length = -2
        with pytest.raises(TypeError):
            edws.sigma = "box"
        with pytest.raises(ValueError):
            edws.sigma = -2
        with pytest.raises(TypeError):
            edws.epsilon = "box"
        with pytest.raises(ValueError):
            edws.epsilon = -2

    def compare_fields(name, field, ftest):

        def check_component(name, f, ftest):
            geoana_norm = np.linalg.norm(f)
            test_norm = np.linalg.norm(ftest)
            diff = np.linalg.norm(f-ftest)
            passed = np.allclose(f, ftest)
            print(
                "Testing {} ... geoana: {:1.4e}, compare: {:1.4e}, "
                "diff: {:1.4e}, passed?: {}".format(
                    name, geoana_norm, test_norm, diff, passed
                )
            )
            return passed

        passed = []
        for i, orientation in enumerate(['x', 'y', 'z']):
            for component in ['real', 'imag']:
                passed.append(check_component(
                    orientation + '_' + component,
                    getattr(field[:, i], component),
                    getattr(ftest[:, i], component)
                ))
        return all(passed)

    def electric_dipole_e(self, orientation):
        sigma = np.random.random_integers(1)
        frequency = np.random.random_integers(1)
        edws = fdem.ElectricDipoleWholeSpace(
            orientation=orientation,
            sigma=sigma,
            frequency=frequency
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        extest, eytest, eztest = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation=orientation.upper()
        )

        e = edws.electric_field(xyz)
        print(
            "\n\nTesting Electric Dipole {} orientation\n".format(orientation)
        )

        passed = self.compare_fields(e, np.vstack([extest, eytest, eztest]).T)
        self.assertTrue(passed)

    def test_electric_dipole_x_e(self):
        self.electric_dipole_e("x")

    def test_electric_dipole_y_e(self):
        self.electric_dipole_e("y")

    def test_electric_dipole_z_e(self):
        self.electric_dipole_e("z")

    def test_electric_dipole_tilted_e(self):

        frequency = 1.0
        orientation = np.random.rand(3)
        orientation = orientation / np.linalg.norm(orientation)

        edws = fdem.ElectricDipoleWholeSpace(
            frequency, orientation=orientation
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)

        xyz = discretize.utils.ndgrid([x, y, z])

        extest0, eytest0, eztest0 = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation='X'
        )
        extest1, eytest1, eztest1 = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation='Y'
        )
        extest2, eytest2, eztest2 = E_from_EDWS(
            xyz, edws.location, edws.sigma, edws.frequency,
            orientation='Z'
        )

        extest = (
            orientation[0]*extest0 + orientation[1]*extest1 + orientation[2]*extest2
        )
        eytest = (
            orientation[0]*eytest0 + orientation[1]*eytest1 + orientation[2]*eytest2
        )
        eztest = (
            orientation[0]*eztest0 + orientation[1]*eztest1 + orientation[2]*eztest2
        )

        e = edws.electric_field(xyz)
        print(
            "\n\nTesting Electric Dipole {} orientation\n".format("45 degree")
        )

        self.compare_fields(e, np.vstack([extest, eytest, eztest]).T)


# class TestFDEMdipole_SimPEG(unittest.TestCase):
#
#     tol = 1e-1 # error must be an order of magnitude smaller than results
#
#     # put the source at the center
#
#     def getFaceSrc(self, mesh):
#         csx = mesh.hx.min()
#         csz = mesh.hz.min()
#         srcInd = (
#             (mesh.gridFz[:, 0] < csx) &
#             (mesh.gridFz[:, 2] < csz/2.) & (mesh.gridFz[:, 2] > -csz/2.)
#         )
#
#         src_vecz = np.zeros(mesh.nFz, dtype=complex)
#         src_vecz[srcInd] = 1.
#
#         return np.hstack(
#             [np.zeros(mesh.vnF[:2].sum(), dtype=complex), src_vecz]
#         )
#
#     def getProjections(self, mesh):
#         ignore_inside_radius = 5*mesh.hx.min()
#         ignore_outside_radius = 40*mesh.hx.min()
#
#         def ignoredGridLocs(grid):
#             return (
#                 (
#                     grid[:, 0]**2 + grid[:, 1]**2 + grid[:, 2]**2  <
#                     ignore_inside_radius**2
#                 ) | (
#                     grid[:, 0]**2 + grid[:, 1]**2 + grid[:, 2]**2 >
#                     ignore_outside_radius**2
#                 )
#             )
#
#         # Faces
#         ignore_me_Fx = ignoredGridLocs(mesh.gridFx)
#         ignore_me_Fz = ignoredGridLocs(mesh.gridFz)
#         ignore_me = np.hstack([ignore_me_Fx, ignore_me_Fz])
#         keep_me = np.array(~ignore_me, dtype=float)
#         Pf = discretize.utils.sdiag(keep_me)
#
#         # Edges
#         ignore_me_Ey = ignoredGridLocs(mesh.gridEy)
#         keep_me_Ey = np.array(~ignore_me_Ey, dtype=float)
#         Pe = discretize.utils.sdiag(keep_me_Ey)
#
#         return Pf, Pe
#
#     def test_e_dipole_v_SimPEG(self):
#
#         def compare_w_SimPEG(name, geoana, simpeg):
#
#             norm_geoana = np.linalg.norm(geoana)
#             norm_simpeg = np.linalg.norm(simpeg)
#             diff = np.linalg.norm(geoana - simpeg)
#             passed = diff < self.tol * 0.5 * (norm_geoana + norm_simpeg)
#             print(
#                 "  {} ... geoana: {:1.4e}, SimPEG: {:1.4e}, diff: {:1.4e}, "
#                 "passed?: {}".format(
#                     name, norm_geoana, norm_simpeg, diff, passed
#                 )
#             )
#
#             return passed
#
#         print("\n\nComparing Electric dipole with SimPEG")
#
#         sigma_back = 1e-1
#         freqs = np.r_[10., 100.]
#
#         csx, ncx, npadx = 0.5, 40, 20
#         ncy = 1
#         csz, ncz, npadz = 0.5, 40, 20
#
#         hx = discretize.utils.meshTensor(
#             [(csx, ncx), (csx, npadx, 1.2)]
#         )
#         hy = 2*np.pi / ncy * np.ones(ncy)
#         hz = discretize.utils.meshTensor(
#             [(csz, npadz, -1.2), (csz, ncz), (csz, npadz, 1.2)]
#         )
#
#         mesh = discretize.CylMesh([hx, hy, hz], x0='00C')
#
#         s_e = self.getFaceSrc(mesh)
#         prob = FDEM.Problem3D_h(mesh, sigmaMap=Maps.IdentityMap(mesh))
#         srcList = [FDEM.Src.RawVec_e([], f, s_e) for f in freqs]
#         survey = FDEM.Survey(srcList)
#
#         prob.pair(survey)
#
#         fields = prob.fields(sigma_back*np.ones(mesh.nC))
#
#         length = mesh.hz.min()
#         current = np.pi * csx**2
#
#         edws = fdem.ElectricDipoleWholeSpace(
#             sigma=sigma_back, length=length, current=current, orientation="z"
#         )
#
#         Pf, Pe = self.getProjections(mesh)
#
#         j_passed = []
#         h_passed = []
#         for i, f in enumerate(freqs):
#             edws.frequency = f
#
#             j_xz = []
#             for j, component in zip([0, 2], ['x', 'z']):
#                 grid = getattr(mesh, "gridF{}".format(component))
#                 j_xz.append(
#                     edws.current_density(grid)[:, j
#                     ]
#                 )
#             j_geoana = np.hstack(j_xz)
#             h_geoana = edws.magnetic_field(mesh.gridEy)[:, 1]
#
#             P_j_geoana = Pf*j_geoana
#             P_j_simpeg = Pf*discretize.utils.mkvc(fields[srcList[i], 'j'])
#
#             P_h_geoana = Pe*h_geoana
#             P_h_simpeg = Pe*discretize.utils.mkvc(fields[srcList[i], 'h'])
#
#             print("Testing {} Hz".format(f))
#
#             for comp in ['real', 'imag']:
#                 j_passed.append(compare_w_SimPEG(
#                     'J {}'.format(comp),
#                     getattr(P_j_geoana, comp),
#                     getattr(P_j_simpeg, comp)
#                 ))
#                 h_passed.append(compare_w_SimPEG(
#                     'H {}'.format(comp),
#                     getattr(P_h_geoana, comp),
#                     getattr(P_h_simpeg, comp)
#                 ))
#         assert(all(j_passed))
#         assert(all(h_passed))


if __name__ == '__main__':
    unittest.main()
