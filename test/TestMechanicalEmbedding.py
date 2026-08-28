import copy
import numpy as np
import openmm
import openmm.app
import os
import pytest

from openmmml import MLPotential
from openmmml.embeddings import utilities

ase = pytest.importorskip("ase", reason="ase is not installed")
mace = pytest.importorskip("mace", reason="mace is not installed")
platform_ints = range(openmm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

atol = 0.01

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMechanicalEmbedding:

    def getTopologyPositionsSubset(self, topology, positions, subset):
        modeller = openmm.app.Modeller(topology, positions)
        modeller.delete([atom for atom in topology.atoms() if atom.index not in subset])
        return modeller.getTopology(), modeller.getPositions()

    @pytest.mark.parametrize("periodic", (False, True))
    @pytest.mark.parametrize("interpolate", (False, True))
    def testEmbedding(self, platform_int, periodic, interpolate):
        """
        Mechanical embedding for a non-periodic system, or for a periodic
        long-range system (in both cases, all periodic images if any are present
        are included or excluded, so the verification calculation is the same).
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology
        positions_ml_mm = pdb.positions

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]
        topology_ml, positions_ml = self.getTopologyPositionsSubset(topology_ml_mm, positions_ml_mm, set(subset))

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("ase")

        from mace.calculators.foundations_models import mace_off

        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
        mm_system_ml = mm_force_field.createSystem(topology_ml, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
        ml_system_ml = ml_potential.createSystem(topology_ml, calculator=mace_off("small"))
        mixed_system = ml_potential.createMixedSystem(topology_ml_mm, mm_system_ml_mm, subset, embedding="mechanical", interpolate=interpolate, calculator=mace_off("small"), mlLongRange=periodic)

        # Disable the dispersion correction for this system for the test so that
        # the same dispersion correction contributions are present on both sides
        # of the energy comparison.
        for force in mm_system_ml.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(False)

        platform = openmm.Platform.getPlatform(platform_int)
        mm_context_ml_mm = openmm.Context(mm_system_ml_mm, openmm.VerletIntegrator(0.001), platform)
        mm_context_ml = openmm.Context(mm_system_ml, openmm.VerletIntegrator(0.001), platform)
        ml_context_ml = openmm.Context(ml_system_ml, openmm.VerletIntegrator(0.001), platform)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)

        mm_context_ml_mm.setPositions(positions_ml_mm)
        mm_context_ml.setPositions(positions_ml)
        ml_context_ml.setPositions(positions_ml)
        mixed_context.setPositions(positions_ml_mm)

        mm_energy_ml_mm = mm_context_ml_mm.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        mm_energy_ml = mm_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        ml_energy_ml = ml_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        # This is the standard expression for mechanical embedding.
        expected_energy = mm_energy_ml_mm - mm_energy_ml + ml_energy_ml

        if interpolate:
            for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
                mixed_context.setParameter("lambda_interpolate", lambda_value)
                mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                assert np.isclose(mixed_energy, expected_energy * lambda_value + mm_energy_ml_mm * (1 - lambda_value), rtol=0, atol=atol)

        else:
            mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(mixed_energy, expected_energy, rtol=0, atol=atol)

    @pytest.mark.parametrize("interpolate", (False, True))
    def testPeriodicShortRange(self, platform_int, interpolate):
        """
        Mechanical embedding for a periodic system where the ML potential is
        assumed to not include interactions with periodic images.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology
        positions_ml_mm = pdb.positions

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]
        topology_ml, positions_ml = self.getTopologyPositionsSubset(topology_ml_mm, positions_ml_mm, set(subset))

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("mace-off23-small")

        # When we compute the MM energy of the ML subset to subtract for the
        # energy comparison, compute it without contributions from any of the
        # periodic images.
        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, nonbondedMethod=openmm.app.PME)
        mm_system_ml = mm_force_field.createSystem(topology_ml, nonbondedMethod=openmm.app.NoCutoff)
        ml_system_ml = ml_potential.createSystem(topology_ml)
        mixed_system = ml_potential.createMixedSystem(topology_ml_mm, mm_system_ml_mm, subset, embedding="mechanical", interpolate=interpolate)

        # Disable the dispersion correction for this system for the test so that
        # the same dispersion correction contributions are present on both sides
        # of the energy comparison.
        for force in mm_system_ml.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(False)

        platform = openmm.Platform.getPlatform(platform_int)
        mm_context_ml_mm = openmm.Context(mm_system_ml_mm, openmm.VerletIntegrator(0.001), platform)
        mm_context_ml = openmm.Context(mm_system_ml, openmm.VerletIntegrator(0.001), platform)
        ml_context_ml = openmm.Context(ml_system_ml, openmm.VerletIntegrator(0.001), platform)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)

        mm_context_ml_mm.setPositions(positions_ml_mm)
        mm_context_ml.setPositions(positions_ml)
        ml_context_ml.setPositions(positions_ml)
        mixed_context.setPositions(positions_ml_mm)

        mm_energy_ml_mm = mm_context_ml_mm.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        mm_energy_ml = mm_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        ml_energy_ml = ml_context_ml.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        expected_energy = mm_energy_ml_mm - mm_energy_ml + ml_energy_ml

        if interpolate:
            for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
                mixed_context.setParameter("lambda_interpolate", lambda_value)
                mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                assert np.isclose(mixed_energy, expected_energy * lambda_value + mm_energy_ml_mm * (1 - lambda_value), rtol=0, atol=atol)

        else:
            mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(mixed_energy, expected_energy, rtol=0, atol=atol)

    @pytest.mark.parametrize("periodic", (False, True))
    @pytest.mark.parametrize("long_range", (False, True, None))
    def testMLLongRangeUnknown(self, platform_int, periodic, long_range):
        """
        An error should be raised if we need to know whether the ML potential is
        long-range or not, and this is not reported or specified.  Check all of
        the cases to ensure this.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("ase")

        from mace.calculators.foundations_models import mace_off

        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, nonbondedMethod=openmm.app.PME if periodic else openmm.app.NoCutoff)
        kwargs = dict(topology=topology_ml_mm, system=mm_system_ml_mm, atoms=subset, calculator=mace_off("small"), embedding="mechanical", mlLongRange=long_range)

        if periodic and long_range is None:
            with pytest.raises(ValueError, match="The system is periodic and it is unknown if the ML model uses long-range interactions"):
                ml_potential.createMixedSystem(**kwargs)
        else:
            ml_potential.createMixedSystem(**kwargs)

    @pytest.mark.parametrize("remove", (False, True))
    def testRemoveConstraints(self, platform_int, remove):
        """
        Constraints in the ML region should be removed if specified.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        topology_ml_mm = pdb.topology

        subset = [atom.index for atom in topology_ml_mm.atoms() if atom.residue.chain.index == 0]
        subset_set = set(subset)

        mm_force_field = openmm.app.ForceField("amber19-all.xml", "amber19/tip3pfb.xml")
        ml_potential = MLPotential("mace-off23-small")

        mm_system_ml_mm = mm_force_field.createSystem(topology_ml_mm, constraints=openmm.app.AllBonds)
        mixed_system = ml_potential.createMixedSystem(topology_ml_mm, mm_system_ml_mm, subset, removeConstraints=remove, embedding="mechanical")

        mm_constraints = set()
        for index in range(mm_system_ml_mm.getNumConstraints()):
            atom_1, atom_2, _ = mm_system_ml_mm.getConstraintParameters(index)
            mm_constraints.add((atom_1, atom_2))

        mixed_constraints = set()
        for index in range(mixed_system.getNumConstraints()):
            atom_1, atom_2, _ = mixed_system.getConstraintParameters(index)
            mixed_constraints.add((atom_1, atom_2))

        # Constraints should be removed only if removeConstraints is set, and
        # constraints should never be added.
        assert bool(mm_constraints - mixed_constraints) == remove
        assert not mixed_constraints - mm_constraints

        for bond in topology_ml_mm.bonds():
            atom_1 = bond.atom1.index
            atom_2 = bond.atom2.index

            assert (atom_1, atom_2) in mm_constraints or (atom_2, atom_1) in mm_constraints
            if atom_1 in subset_set and atom_2 in subset_set:
                assert ((atom_1, atom_2) in mixed_constraints or (atom_2, atom_1) in mixed_constraints) != remove

    def testLinkAtomForceProjection(self, platform_int):
        """
        The position of each link atom is a function of the positions of the
        two atoms of the bond spanning the ML and MM regions.  The force
        projection performed by `ml_forces_to_system()` must therefore equal
        the exact gradient of the energy of the link atoms with respect to the
        positions of all atoms, and `system_positions_to_ml()` must place each
        link atom at the position implied by the constraint.
        """

        def check(positions, indices, linkBondsData):
            # An arbitrary nonlinear energy that depends on the positions of
            # the link atoms only (each through the two atoms of its own bond).
            coeffs = [np.random.default_rng(i).normal(size=3) for i in range(len(linkBondsData))]

            def la_position(ps, la):
                vers = ps[la['mm']] - ps[la['ml']]
                vers /= np.linalg.norm(vers)
                return ps[la['ml']] + vers*la['d'].value_in_unit(openmm.unit.angstrom)

            def energy(ps):
                total = 0.0
                for i, la in enumerate(linkBondsData):
                    r = la_position(ps, la)
                    c = coeffs[i]
                    total += np.sum((r - c)**4) + np.sum(r**2) + 0.5*np.dot(r, [1, 2, 3])*np.sum(r)
                return total

            def la_forces(ps):
                forces = []
                for i, la in enumerate(linkBondsData):
                    r = la_position(ps, la)
                    c = coeffs[i]
                    forces.append(-(4*(r - c)**3 + 2*r + 0.5*np.array([1, 2, 3])*np.sum(r) + 0.5*np.dot(r, [1, 2, 3])))
                return forces

            nml = len(indices)
            ml_forces = np.vstack([np.zeros((nml, 3))] + [f.reshape(1, 3) for f in la_forces(positions)])
            forces = utilities.ml_forces_to_system(positions, ml_forces, indices, linkBondsData)

            # Check the projected forces against finite differences of the
            # energy for every atom.
            eps = 1e-7
            for atom in range(len(positions)):
                for i in range(3):
                    p = positions.copy()
                    p[atom, i] += eps
                    expected = -(energy(p) - energy(positions))/eps
                    assert np.isclose(forces[atom, i], expected, atol=1e-5), \
                        f"atom {atom}, component {i}: {forces[atom, i]} != {expected}"

            # Check the extracted positions: the ML subset positions followed
            # by the constrained link atom positions.
            expected_positions = np.vstack([positions[indices]] + [la_position(positions, la).reshape(1, 3) for la in linkBondsData])
            assert np.allclose(utilities.system_positions_to_ml(positions, indices, linkBondsData), expected_positions)

        rng = np.random.default_rng(0)

        # A single link bond between atom 0 (ML) and atom 1 (MM).
        r_ml = rng.normal(size=3)
        r_mm = r_ml + rng.normal(size=3) + 2.0
        check(np.stack([r_ml, r_mm]), [0], [{'ml': 0, 'mm': 1, 'laz': 1, 'd': 1.07*openmm.unit.angstrom}])

        # Multiple link bonds, with the ML and MM atoms interleaved.
        positions = rng.normal(size=(6, 3)) + 3.0
        check(positions, [0, 2, 4], [
            {'ml': 0, 'mm': 1, 'laz': 1, 'd': 1.09*openmm.unit.angstrom},
            {'ml': 2, 'mm': 3, 'laz': 1, 'd': 1.07*openmm.unit.angstrom},
        ])

    @pytest.mark.parametrize("model_name", ("mace-off23-small", "ase"))
    @pytest.mark.parametrize("override_distance", (False, True))
    def testLinkAtomTerms(self, platform_int, model_name, override_distance):
        """
        Test for presence of the appropriate terms in the link-atom method, and
        that the ML potential sees the ML region capped by link atoms.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "ethanol", "ethanol.pdb"))
        """
                  H4   H6
                  |    |
        H3 - O0 - C1 - C2 - H8
                  |    |
                  H5   H7
        """

        mm_force_field = openmm.app.ForceField(os.path.join(test_data_dir, "ethanol", "ethanol.xml"))
        if model_name == "ase":
            # Use the ASE potential with a MACE calculator to test a
            # non-MACE model.
            from mace.calculators.foundations_models import mace_off
            ml_potential = MLPotential("ase")
            potential_args = dict(calculator=mace_off("small", default_dtype="float32"))
        else:
            ml_potential = MLPotential(model_name)
            potential_args = {}
        subset = [0, 1, 3, 4, 5]

        mm_system = mm_force_field.createSystem(pdb.topology)
        args = {}
        if override_distance:
            args["linkAtomPrm"] = [(1, 2, 0.12*openmm.unit.nanometer)]
        mixed_system = ml_potential.createMixedSystem(pdb.topology, mm_system, subset, interpolate=False, **potential_args, **args)

        # The link atoms are not real particles: the particle count is
        # unchanged and no virtual sites are present.
        assert mixed_system.getNumParticles() == mm_system.getNumParticles() == pdb.topology.getNumAtoms()
        assert not any(mixed_system.isVirtualSite(i) for i in range(mixed_system.getNumParticles()))

        def get_terms(system):
            bonds = set()
            bond_force, = (force for force in system.getForces() if isinstance(force, openmm.HarmonicBondForce))
            for i in range(bond_force.getNumBonds()):
                bond = tuple(bond_force.getBondParameters(i)[:2])
                bonds.add(min(bond, bond[::-1]))

            angles = set()
            angle_force, = (force for force in system.getForces() if isinstance(force, openmm.HarmonicAngleForce))
            for i in range(angle_force.getNumAngles()):
                angle = tuple(angle_force.getAngleParameters(i)[:3])
                angles.add(min(angle, angle[::-1]))

            torsions = set()
            torsion_force, = (force for force in system.getForces() if isinstance(force, openmm.PeriodicTorsionForce))
            for i in range(torsion_force.getNumTorsions()):
                torsion = tuple(torsion_force.getTorsionParameters(i)[:4])
                torsions.add(min(torsion, torsion[::-1]))

            return bonds, angles, torsions

        # Get all of the bonded terms in both systems.
        mm_bonds, mm_angles, mm_torsions = get_terms(mm_system)
        mixed_bonds, mixed_angles, mixed_torsions = get_terms(mixed_system)

        # No bonded terms should be added to the mixed system.
        assert not mixed_bonds - mm_bonds
        assert not mixed_angles - mm_angles
        assert not mixed_torsions - mm_torsions

        # The appropriate terms should be removed from the mixed system.
        assert mm_bonds - mixed_bonds == {(0, 1), (0, 3), (1, 4), (1, 5)}
        assert mm_angles - mixed_angles == {(0, 1, 2), (0, 1, 4), (0, 1, 5), (1, 0, 3), (2, 1, 4), (2, 1, 5), (4, 1, 5)}
        assert mm_torsions - mixed_torsions == {(2, 1, 0, 3), (3, 0, 1, 4), (3, 0, 1, 5)}

        # The ML potential must see the ML region capped by a link atom on the
        # C1-C2 bond: its energy must equal that of an equivalent system in
        # which the capping hydrogen is a real atom.
        distance = 0.12*openmm.unit.nanometer if override_distance else utilities.get_linkatom_distance(pdb.topology, 1, 2, 1)

        pos_np = np.array([np.asarray(p) for p in pdb.positions.value_in_unit(openmm.unit.nanometer)])
        delta = pos_np[2] - pos_np[1]
        capping_position = pos_np[1] + delta/np.linalg.norm(delta)*distance.value_in_unit(openmm.unit.nanometer)

        # Reference system: the ML region capped by an explicit hydrogen,
        # modeled entirely by the ML potential.
        modeller = openmm.app.Modeller(pdb.topology, pdb.positions)
        modeller.delete([atom for atom in pdb.topology.atoms() if atom.index not in subset])
        ref_topology = modeller.getTopology()
        ref_topology.addAtom("LA", openmm.app.element.hydrogen, list(ref_topology.residues())[0])
        ref_atoms = list(ref_topology.atoms())
        ref_topology.addBond(ref_atoms[1], ref_atoms[-1])
        ref_system = ml_potential.createSystem(ref_topology, **potential_args)

        # The MM part of the mixed system: the conventional terms with the
        # ML-internal bonded terms removed and the ML-ML nonbonded terms
        # zeroed, as done by the mechanical embedding.
        linkBondsData = [{'ml': 1, 'mm': 2, 'laz': 1, 'd': distance}]
        mm_part = utilities.removeBonds(mm_system, subset, True, linkBondsData=linkBondsData)
        for force in mm_part.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for i1 in range(len(subset)):
                    for i2 in range(i1):
                        force.addException(subset[i1], subset[i2], 0, 1, 0, True)

        platform = openmm.Platform.getPlatform(platform_int)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)
        mixed_context.setPositions(pdb.positions)
        ref_context = openmm.Context(ref_system, openmm.VerletIntegrator(0.001), platform)
        ref_context.setPositions([openmm.Vec3(*p) for p in np.vstack([pos_np[subset], capping_position])])
        mm_context = openmm.Context(mm_part, openmm.VerletIntegrator(0.001), platform)
        mm_context.setPositions(pdb.positions)

        mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        ref_energy = ref_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        mm_part_energy = mm_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        assert np.isclose(mixed_energy, ref_energy + mm_part_energy, rtol=0, atol=atol)

        # Run some dynamics to make sure the link-atom terms behave during
        # integration.
        integrator = openmm.LangevinIntegrator(300, 1, 0.001)
        dynamics_context = openmm.Context(mixed_system, integrator, platform)
        dynamics_context.setPositions(pdb.positions)
        openmm.LocalEnergyMinimizer.minimize(dynamics_context)
        integrator.step(100)

    def testLinkAtomInterpolation(self, platform_int):
        """
        Ensure interpolation works as expected with the link-atom method.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "ethanol", "ethanol.pdb"))

        mm_force_field = openmm.app.ForceField(os.path.join(test_data_dir, "ethanol", "ethanol.xml"))
        ml_potential = MLPotential("mace-off23-small")

        mm_system = mm_force_field.createSystem(pdb.topology)
        mixed_system = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5], interpolate=False)
        interpolate_system = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5], interpolate=True)

        platform = openmm.Platform.getPlatform(platform_int)
        mm_context = openmm.Context(mm_system, openmm.VerletIntegrator(0.001), platform)
        mixed_context = openmm.Context(mixed_system, openmm.VerletIntegrator(0.001), platform)
        interpolate_context = openmm.Context(interpolate_system, openmm.VerletIntegrator(0.001), platform)

        mm_context.setPositions(pdb.positions)
        mixed_context.setPositions(pdb.positions)
        interpolate_context.setPositions(pdb.positions)

        mm_energy = mm_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        mixed_energy = mixed_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

        for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
            interpolate_context.setParameter("lambda_interpolate", lambda_value)
            interpolate_energy = interpolate_context.getState(energy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            assert np.isclose(interpolate_energy, mixed_energy * lambda_value + mm_energy * (1 - lambda_value), rtol=0, atol=atol)

    def testCreateMixedSystem(self, platform_int):
        """
        Ensure createMixedSystem() always returns a plain System and does not
        modify the input Topology or System.
        """

        pdb = openmm.app.PDBFile(os.path.join(test_data_dir, "ethanol", "ethanol.pdb"))
        mm_force_field = openmm.app.ForceField(os.path.join(test_data_dir, "ethanol", "ethanol.xml"))
        ml_potential = MLPotential("mace-off23-small")
        mm_system = mm_force_field.createSystem(pdb.topology)

        def bond_count(system):
            bond_force, = (force for force in system.getForces() if isinstance(force, openmm.HarmonicBondForce))
            return bond_force.getNumBonds()

        original_topology = copy.deepcopy(pdb.topology)
        original_system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        mixed_system = ml_potential.createMixedSystem(pdb.topology, mm_system, [0, 1, 3, 4, 5])

        # A plain System is returned, with the same number of particles as the
        # input (the link atoms are not real particles).
        assert isinstance(mixed_system, openmm.System)
        assert mixed_system.getNumParticles() == mm_system.getNumParticles() == pdb.topology.getNumAtoms()

        # Make sure the inputs were not modified.
        assert pdb.topology.getNumAtoms() == original_topology.getNumAtoms()
        assert mm_system.getNumParticles() == original_system.getNumParticles()
        assert mm_system.getNumConstraints() == original_system.getNumConstraints()
        assert bond_count(mm_system) == bond_count(original_system)
