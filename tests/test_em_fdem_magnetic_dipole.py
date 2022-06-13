import unittest
import pytest
import numpy as np
import discretize

from scipy.constants import mu_0, epsilon_0
from geoana.em import fdem


def H_from_MagneticDipoleWholeSpace(
    XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=0,
    epsr=1., t=0.
):

    assert current == 1
    assert loopArea == 1
    assert np.all(srcLoc == np.r_[0., 0., 0.])
    assert kappa == 0

    mu = mu_0 * (1+kappa)
    epsilon = epsilon_0 * epsr
    m = current * loopArea

    assert m == 1
    assert mu == mu_0
    assert epsilon == epsilon_0

    omega = lambda f: 2*np.pi*f

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)
    # Check

    dx = XYZ[:, 0]-srcLoc[0]
    dy = XYZ[:, 1]-srcLoc[1]
    dz = XYZ[:, 2]-srcLoc[2]

    r = np.sqrt(dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k = np.sqrt(omega(f)**2. * mu * epsilon - 1j*omega(f)*mu*sig)

    front = m / (4. * np.pi * r**3) * np.exp(-1j*k*r)
    mid   = - k**2 * r**2 + 3*1j*k*r + 3

    if orientation.upper() == 'X':
        Hx = front*((dx**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
        Hy = front*(dx*dy  / r**2)*mid
        Hz = front*(dx*dz  / r**2)*mid

    elif orientation.upper() == 'Y':
        #  x--> y, y--> z, z-->x
        Hy = front * ((dy**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
        Hz = front * (dy*dz  / r**2)*mid
        Hx = front * (dy*dx  / r**2)*mid

    elif orientation.upper() == 'Z':
        # x --> z, y --> x, z --> y
        Hz = front*((dz**2 / r**2)*mid + (k**2 * r**2 - 1j*k*r - 1.))
        Hx = front*(dz*dx  / r**2)*mid
        Hy = front*(dz*dy  / r**2)*mid

    return Hx, Hy, Hz


def B_from_MagneticDipoleWholeSpace(
    XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=0,
    epsr=1., t=0.
):

    mu = mu_0 * (1+kappa)

    Hx, Hy, Hz = H_from_MagneticDipoleWholeSpace(
        XYZ, srcLoc, sig, f, current=current, loopArea=loopArea,
        orientation=orientation, kappa=kappa, epsr=epsr
    )
    Bx = mu * Hx
    By = mu * Hy
    Bz = mu * Hz
    return Bx, By, Bz


def E_from_MagneticDipoleWholeSpace(
    XYZ, srcLoc, sig, f, current=1., loopArea=1., orientation='X', kappa=0.,
    epsr=1., t=0.
):

    mu = mu_0 * (1+kappa)
    epsilon = epsilon_0 * epsr
    m = current * loopArea

    omega = lambda f: 2 * np.pi * f

    XYZ = discretize.utils.asArray_N_x_Dim(XYZ, 3)

    dx = XYZ[:, 0]-srcLoc[0]
    dy = XYZ[:, 1]-srcLoc[1]
    dz = XYZ[:, 2]-srcLoc[2]

    r  = np.sqrt( dx**2. + dy**2. + dz**2.)
    # k  = np.sqrt( -1j*2.*np.pi*f*mu*sig )
    k  = np.sqrt( omega(f)**2. *mu*epsilon -1j*omega(f)*mu*sig )

    front = (
        ((1j * omega(f) * mu * m) / (4.* np.pi * r**2)) *
        (1j * k * r + 1) * np.exp(-1j*k*r)
    )

    if orientation.upper() == 'X':
        Ey = front * (dz / r)
        Ez = front * (-dy / r)
        Ex = np.zeros_like(Ey)

    elif orientation.upper() == 'Y':
        Ex = front * (-dz / r)
        Ez = front * (dx / r)
        Ey = np.zeros_like(Ex)

    elif orientation.upper() == 'Z':
        Ex = front * (dy / r)
        Ey = front * (-dx / r)
        Ez = np.zeros_like(Ex)

    return Ex, Ey, Ez


class TestFDEMdipole(unittest.TestCase):

    def test_defaults(self):
        TOL = 1e-15
        frequency = 1.0
        mdws = fdem.MagneticDipoleWholeSpace(frequency)
        assert(mdws.sigma == 1.)
        assert(mdws.mu == mu_0)
        assert(mdws.epsilon == epsilon_0)
        assert(np.all(mdws.orientation == np.r_[1., 0., 0.]))
        assert(mdws.moment == 1.)
        assert(np.all(mdws.location == np.r_[0., 0., 0.]))
        assert(mdws.frequency == 1.)
        assert(mdws.omega == 2.*np.pi*1.)
        assert(mdws.quasistatic is False)
        assert np.linalg.norm(
            mdws.wavenumber - np.sqrt(
                mu_0 * epsilon_0 * (2*np.pi)**2  - 1j * mu_0 * 1. * 2*np.pi
            )
        ) <= TOL
        assert np.linalg.norm(
            mdws.wavenumber**2 - (
                mu_0 * epsilon_0 * (2*np.pi)**2  - 1j * mu_0 * 1. * 2*np.pi
            )
        ) <= TOL

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

    def magnetic_dipole_b(self, orientation):
        sigma = 1
        frequency = 1.
        mdws = fdem.MagneticDipoleWholeSpace(
            orientation=orientation,
            sigma=sigma,
            frequency=frequency
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        # srcLoc, obsLoc, component, orientation='Z', moment=1., mu=mu_0

        # btest = [MagneticDipoleFields(
        #     mdws.location, xyz, rx_orientation,
        #     orientation=orientation.upper()
        # ) for rx_orientation in ["x", "y", "z"]]

        bxtest, bytest, bztest = B_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
            orientation=orientation
        )

        b = mdws.magnetic_flux_density(xyz)
        print(
            "\n\nTesting Magnetic Dipole B: {} orientation\n".format(orientation)
        )

        passed = self.compare_fields(b, np.vstack([bxtest, bytest, bztest]).T)
        self.assertTrue(passed)

    def magnetic_dipole_e(self, orientation):
        sigma = 1e-2
        frequency = 1
        mdws = fdem.MagneticDipoleWholeSpace(
            orientation=orientation,
            sigma=sigma,
            frequency=frequency
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = discretize.utils.ndgrid([x, y, z])

        extest, eytest, eztest = E_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
            orientation=orientation
        )

        e = mdws.electric_field(xyz)
        print(
            "\n\nTesting Magnetic Dipole E: {} orientation\n".format(orientation)
        )

        passed = self.compare_fields(e, np.vstack([extest, eytest, eztest]).T)
        self.assertTrue(passed)

    def test_magnetic_dipole_x_b(self):
        self.magnetic_dipole_b("x")

    def test_magnetic_dipole_y_b(self):
        self.magnetic_dipole_b("y")

    def test_magnetic_dipole_z_b(self):
        self.magnetic_dipole_b("z")

    def test_magnetic_dipole_tilted_b(self):

        frequency = 1.0
        orientation = np.random.rand(3)
        orientation = orientation / np.linalg.norm(orientation)

        mdws = fdem.MagneticDipoleWholeSpace(
            frequency, orientation=orientation
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)

        xyz = discretize.utils.ndgrid([x, y, z])

        bxtest0, bytest0, bztest0 = B_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
            orientation='X'
        )
        bxtest1, bytest1, bztest1 = B_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
            orientation='Y'
        )
        bxtest2, bytest2, bztest2 = B_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
            orientation='Z'
        )

        bxtest = (
            orientation[0]*bxtest0 + orientation[1]*bxtest1 + orientation[2]*bxtest2
        )
        bytest = (
            orientation[0]*bytest0 + orientation[1]*bytest1 + orientation[2]*bytest2
        )
        bztest = (
            orientation[0]*bztest0 + orientation[1]*bztest1 + orientation[2]*bztest2
        )

        b = mdws.magnetic_flux_density(xyz)
        print(
            "\n\nTesting Magnetic Dipole B: {} orientation\n".format("45 degree")
        )

        self.compare_fields(b, np.vstack([bxtest, bytest, bztest]).T)

    def test_magnetic_dipole_x_e(self):
        self.magnetic_dipole_e("x")

    def test_magnetic_dipole_y_e(self):
        self.magnetic_dipole_e("y")

    def test_magnetic_dipole_z_e(self):
        self.magnetic_dipole_e("z")

    def test_magnetic_dipole_tilted_e(self):

        frequency = 1.0
        orientation = np.random.rand(3)
        orientation = orientation / np.linalg.norm(orientation)

        mdws = fdem.MagneticDipoleWholeSpace(
            frequency, orientation=orientation
        )
        x = np.linspace(-20., 20., 10)
        y = np.linspace(-30., 30., 10)
        z = np.linspace(-40., 40., 10)

        xyz = discretize.utils.ndgrid([x, y, z])

        extest0, eytest0, eztest0 = E_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
            orientation='X'
        )
        extest1, eytest1, eztest1 = E_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
            orientation='Y'
        )
        extest2, eytest2, eztest2 = E_from_MagneticDipoleWholeSpace(
            xyz, mdws.location, mdws.sigma, mdws.frequency,
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

        e = mdws.electric_field(xyz)
        print(
            "\n\nTesting Magnetic Dipole E: {} orientation\n".format("45 degree")
        )

        self.compare_fields(e, np.vstack([extest, eytest, eztest]).T)


# class TestFDEMdipole_SimPEG(unittest.TestCase):
#
#     tol = 1e-1 # error must be an order of magnitude smaller than results
#
#     def getProjections(self, mesh):
#         ignore_inside_radius = 10*mesh.hx.min()
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
#     def test_b_dipole_v_SimPEG(self):
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
#         print("\n\nComparing Magnetic dipole with SimPEG")
#
#         sigma_back = 1.
#         freqs = np.r_[10., 100.]
#
#         csx, ncx, npadx = 1, 50, 50
#         ncy = 1
#         csz, ncz, npadz = 1, 50, 50
#
#         hx = discretize.utils.meshTensor(
#             [(csx, ncx), (csx, npadx, 1.3)]
#         )
#         hy = 2*np.pi / ncy * np.ones(ncy)
#         hz = discretize.utils.meshTensor(
#             [(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)]
#         )
#
#         mesh = discretize.CylMesh([hx, hy, hz], x0='00C')
#
#         prob = FDEM.Problem3D_e(mesh, sigmaMap=Maps.IdentityMap(mesh))
#         srcList = [FDEM.Src.MagDipole([], loc=np.r_[0., 0., 0.], freq=f) for f in freqs]
#         survey = FDEM.Survey(srcList)
#
#         prob.pair(survey)
#
#         fields = prob.fields(sigma_back*np.ones(mesh.nC))
#
#         moment = 1.
#         mdws = fdem.MagneticDipoleWholeSpace(
#             sigma=sigma_back, moment=moment, orientation="z"
#         )
#
#         Pf, Pe = self.getProjections(mesh)
#
#         e_passed = []
#         b_passed = []
#         for i, f in enumerate(freqs):
#             mdws.frequency = f
#
#             b_xz = []
#             for b, component in zip([0, 2], ['x', 'z']):
#                 grid = getattr(mesh, "gridF{}".format(component))
#                 b_xz.append(
#                     mdws.magnetic_flux_density(grid)[:, b]
#                 )
#             b_geoana = np.hstack(b_xz)
#             e_geoana = mdws.electric_field(mesh.gridEy)[:, 1]
#
#             P_e_geoana = Pe*e_geoana
#             P_e_simpeg = Pe*discretize.utils.mkvc(fields[srcList[i], 'e'])
#
#             P_b_geoana = Pf*b_geoana
#             P_b_simpeg = Pf*discretize.utils.mkvc(fields[srcList[i], 'b'])
#
#             print("Testing {} Hz".format(f))
#
#             for comp in ['real', 'imag']:
#                 e_passed.append(compare_w_SimPEG(
#                     'E {}'.format(comp),
#                     getattr(P_e_geoana, comp),
#                     getattr(P_e_simpeg, comp)
#                 ))
#                 b_passed.append(compare_w_SimPEG(
#                     'B {}'.format(comp),
#                     getattr(P_b_geoana, comp),
#                     getattr(P_b_simpeg, comp)
#                 ))
#         assert(all(e_passed))
#         assert(all(b_passed))


if __name__ == '__main__':
    unittest.main()
