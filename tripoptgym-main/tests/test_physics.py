"""
Unit tests for physics module.

Test cases adapted from TripOptGym.ipynb notebook.
"""

import pytest
import numpy as np
from tripoptgym.environment.physics import TrainPhysics, LocomotiveModel


class TestMonicQuadratic:
    """Tests for monic_quadratic solver."""
    
    def setup_method(self):
        self.physics = TrainPhysics()
    
    def test_case_1(self):
        """Test case from notebook: expected result 71.27"""
        result = self.physics.monic_quadratic(0.045753347, -5083.083704)
        assert abs(result - 71.27) < 0.01
    
    def test_case_2(self):
        """Test case from notebook: expected result 3"""
        result = self.physics.monic_quadratic(1e-19, -9)
        assert abs(result - 3.0) < 0.01
    
    def test_case_3(self):
        """Test case from notebook: expected result 10.0001"""
        result = self.physics.monic_quadratic(-12, 19.99919999)
        assert abs(result - 10.0001) < 0.0001
    
    def test_case_4(self):
        """Test case from notebook: expected result 5.9999"""
        result = self.physics.monic_quadratic(-2.9998, -18.00029999)
        assert abs(result - 5.9999) < 0.0001


class TestMonicCubic:
    """Tests for monic_cubic solver."""
    
    def setup_method(self):
        self.physics = TrainPhysics()
    
    def test_case_1(self):
        """Test case from notebook: expected result -0.682327803828019"""
        result = self.physics.monic_cubic(1e-19, 1, 1)
        assert abs(result - (-0.682327803828019)) < 0.000001


class TestSolveCubic:
    """Tests for solve_cubic function."""
    
    def setup_method(self):
        self.physics = TrainPhysics()
    
    def test_case_1(self):
        """Test case from notebook: expected result 1"""
        result = self.physics.solve_cubic([1, -1, 1, -1])
        assert abs(result - 1.0) < 0.01
    
    def test_case_2(self):
        """Test case from notebook: expected result 71.27"""
        result = self.physics.solve_cubic([0, 0.000127015, 5.81136E-06, -0.645627482])
        assert abs(result - 71.27) < 0.01


class TestTrapzIntegrateTrain:
    """Tests for trapz_integrate_train_one_step."""
    
    def setup_method(self):
        self.physics = TrainPhysics()
    
    def test_basic_integration(self):
        """Test basic train integration step."""
        vplus, pplus, fail_code = self.physics.trapz_integrate_train_one_step(
            h=0.1,           # step size (miles)
            v0=20,           # initial velocity (mph)
            g=0.1,           # current grade
            gplus=0.1,       # next grade
            M=10000,         # mass (tons)
            a=-0.2,          # Davis equation coefficients
            b=-0.2,
            c=-0.2,
            p=4000,          # current power (HP)
            pplus=4000,      # next power (HP)
            Fabe=0,          # air brake force
            FabePlus=0,      # next air brake force
            Fsat=50000,      # max tractive effort
            FsatPlus=50000,  # next max tractive effort
            v0fail=1,        # failure velocity threshold
            cbs=70,          # commutation breakpoint speed
            hpmaxspd=4000,   # max HP at speed
            maxbhp=6000      # max braking HP
        )
        
        # Check that we got valid outputs
        assert vplus > 0, "Velocity should be positive"
        assert pplus >= 0, "Power should be non-negative"
        assert fail_code in [0, 1, 2, 3], "Fail code should be valid"
        
        # Velocity should not change drastically in one step
        assert abs(vplus - 20) < 10, "Velocity change should be reasonable"


class TestLocomotiveModel:
    """Tests for LocomotiveModel class."""
    
    def setup_method(self):
        self.model = LocomotiveModel()
    
    def test_max_te_range(self):
        """Test MaxTEForNotch returns valid values across notch range."""
        # Dynamic brake notches (-8 to 0)
        for notch in range(-8, 1):
            te = self.model.MaxTEForNotch(notch)
            assert te < 0, f"Dynamic brake notch {notch} should have negative TE"
        
        # Motoring notches (1 to 8)
        for notch in range(1, 9):
            te = self.model.MaxTEForNotch(notch)
            assert te > 0, f"Motoring notch {notch} should have positive TE"
    
    def test_thp_range(self):
        """Test THPForNotch returns valid values across notch range."""
        # Dynamic brake notches (-8 to 0)
        for notch in range(-8, 1):
            hp = self.model.THPForNotch(notch)
            assert hp <= 0, f"Dynamic brake notch {notch} should have non-positive HP"
        
        # Motoring notches (1 to 8)
        for notch in range(1, 9):
            hp = self.model.THPForNotch(notch)
            assert hp > 0, f"Motoring notch {notch} should have positive HP"
    
    def test_thp_increases_with_notch(self):
        """Test that HP increases monotonically with throttle notch."""
        prev_hp = self.model.THPForNotch(1)
        for notch in range(2, 9):
            current_hp = self.model.THPForNotch(notch)
            assert current_hp > prev_hp, f"HP should increase from notch {notch-1} to {notch}"
            prev_hp = current_hp
    
    def test_max_te_increases_with_notch(self):
        """Test that TE increases monotonically with throttle notch."""
        prev_te = self.model.MaxTEForNotch(1)
        for notch in range(2, 9):
            current_te = self.model.MaxTEForNotch(notch)
            assert current_te > prev_te, f"TE should increase from notch {notch-1} to {notch}"
            prev_te = current_te
    
    def test_commutation_breakpoint(self):
        """Test commutation breakpoint speed."""
        cbs = self.model.CommutationBreakpointSpeed()
        assert cbs == 70, "Commutation breakpoint should be 70 mph"
    
    def test_max_commutation_braking_hp(self):
        """Test max commutation braking HP."""
        hp = self.model.MaxCommutationBrakingHP()
        assert hp < 0, "Max commutation braking HP should be negative"
        assert abs(hp - (-13416.75819)) < 0.01


class TestPhysicsEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def setup_method(self):
        self.physics = TrainPhysics()
    
    def test_quadratic_zero_coefficients(self):
        """Test quadratic with zero coefficients."""
        result = self.physics.monic_quadratic(0, 0)
        assert result == 0
    
    def test_quadratic_no_real_roots(self):
        """Test quadratic with no real roots."""
        result = self.physics.monic_quadratic(1, 1)
        assert result == -1, "Should return -1 for no real roots"
    
    def test_cubic_zero_constant(self):
        """Test cubic with zero constant term."""
        result = self.physics.monic_cubic(0, 0, 0)
        assert result >= 0, "Should find non-negative root"
