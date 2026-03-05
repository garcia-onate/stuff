"""Train physics simulation and locomotive models.

This module contains the physics engine for train dynamics and locomotive
characteristics used in the TripOpt Gym environment.
"""

import numpy as np


class TrainPhysics:
    """Physics simulation for freight train dynamics.
    
    Implements trapezoidal integration method for train motion simulation,
    including grade effects, resistance, and locomotive force calculations.
    """
    
    def __init__(self):
        pass

    def trapz_integrate_train_one_step(self, h, v0, g, gplus, M, a, b, c,
                                       p, pplus, F_abe, F_abe_plus, Fsat,
                                       Fsatplus, direction, cbs, hpmaxpsd,
                                       maxbhp):
        """Integrate train dynamics one step forward using trapezoidal method.
        
        Parameters
        ----------
        h : float
            Step size in miles
        v0 : float
            Current velocity in mph
        g : float
            Grade at current location (percent)
        gplus : float
            Grade at next location (percent)
        M : float
            Train mass (tons)
        a, b, c : float
            Davis equation coefficients for resistance
        p : float
            Current throttle horsepower
        pplus : float
            Next throttle horsepower
        F_abe : float
            Additional braking effort at current step
        F_abe_plus : float
            Additional braking effort at next step
        Fsat : float
            Maximum tractive effort at current step
        Fsatplus : float
            Maximum tractive effort at next step
        direction : int
            Direction of travel (0=forward, 1=reverse)
        cbs : float
            Commutation breakpoint speed
        hpmaxpsd : float
            Maximum commutation braking HP
        maxbhp : float
            Maximum dynamic braking HP
            
        Returns
        -------
        vplus : float
            Velocity at next step
        pplus : float
            Throttle HP at next step
        fail_code : int
            Failure code (0=success, 1=stalled, 2=velocity overflow)
        """
        if direction == 1:
            a = -a
            b = -b
            c = -c

        Fplus=0
        vplus=0

        if v0 > cbs:
            pmin = np.interp(v0, [cbs, 70], [maxbhp, hpmaxpsd])

            # direction is forward
            if (direction == 0):
                p = max(pmin, p)
                pplus = max(pmin, pplus)
            else:
                pmin = -1*pmin
                p = min(pmin, p)
                pplus = min(pmin, pplus)

        # Small velocity
        vmin = 0.01
        # large velocity
        vmax = 75

        U = 78972.6657272727

        # break speed
        if abs(Fsatplus) > 1.0:
            vb = pplus/Fsatplus*375
        else:
            vb = vmin

        # Mass in tons
        m = 2000*M

        # Trapezoidal Integration:
        #
        # (speed(k+1)^2 - speed(k)^2)/U/h(k) = (F(speed(k+1)) + F(speed(k))/m
        # - (g(k+1)+g(k))/100 - (a + b*speed(k) + c*speed(k)^2)/2000 - (a + b*speed(k+1) +
        # + c*speed(k+1)^2)/2000
        #
        # Define v := speed(k) and vplus := speed(k+1); Fplus =
        # F(speed(k+1)) and F = F(speed(k)), etc.
        # (vplus^2 - v^2)/U/h = (Fplus + F)/m - (gplus + g)/100 -
        # (a+b*v+c*v^2)/2000 - (a + b*vplus + c*vplus^2)/2000
        #
        # If Fplus is known, we get the following quadratic equation in
        # vplus:
        # vplus^2(1/U/h + c/2000) + (b/2000)vplus + [(gplus + g)/100
        #       + (a + b*v + c*v^2)/2000 + (a/2000) - v^2/U/h
        #       - (Fplus + F)/m] = 0
        #
        # If Fplus is assumed to be 375*pplus/vplus + F_abe_plus, we have the cubic
        #
        # vplus^3(1/U/h + c/2000) + (b/2000)vplus^2 + [(gplus + g)/100
        #       + (a + b*v + c*v^2)/2000 + (a/2000) - v^2/U/h
        #       - F_abe_plus/m - F/m] - 375*pplus/m = 0
        #
        # Thus, to calculate Fplus, use
        # Fplus = -F + mA, where
        # A = vplus^2(1/U/h + c/2000) + (b/2000)vplus + [(gplus + g)/100
        #       + (a + b*v + c*v^2)/2000 + (a/2000) - v^2/U/h ]
        #

        # Initialize
        if p > 0:
            F = min(375*p/v0, Fsat) + F_abe
        else:
            F = max(375*p/v0, Fsat) + F_abe

        quadratic_flag = 0
        fail_code = 0
        if (abs(pplus) < 1e-5):
            quadratic_flag = 1
            Fplus = F_abe_plus
        elif (pplus < 0):
            # The next time step is a braking event
            # Compute the force required to make the train slow down to the breakpoint speed
            vbsq = pow(vb,2)
            v0sq = pow(v0,2)
            A = vbsq*(1/U/h + c/2000) + (b/2000)*vb + (gplus + g)/100 + (a + b*v0 + c*v0sq)/2000 + (a/2000) - v0sq/U/h
            Fb = -F + m*A
            if (Fb < Fsatplus + F_abe_plus):
                # Force is smaller than break-force, so speed is larger than
                # break-speed. Therefore, we are in constant power region
                quadratic_flag = 0
            else:
                # Speed is less than break-speed, so Fplus is known to be
                # Fsatplus + F_abe_plus.
                Fplus = Fsatplus + F_abe_plus
                quadratic_flag = 1
        else:
            # Next time step is a motoring event. Assume that the force is
            # equal to the break-force
            Fb = Fsatplus + F_abe_plus
            # Solve the following quadratic for vplus
            # vplus^2(1/U/h + c/2000) + (b/2000)vplus + [(gplus + g)/100
            #       + (a + b*v + c*v^2)/2000 + (a/2000) - v^2/U/h
            #       - Fb/m - F/m] = 0
            q2 = (1/U/h + c/2000)
            q1 = (b/2000)
            v0sq = pow(v0,2)
            q0 = (gplus + g)/100 + (a + b*v0 + c*v0sq)/2000 + (a/2000) - v0sq/U/h -(Fb + F)/m
            vplus_vec = self.solve_cubic([0, q2, q1, q0])

            if vplus_vec <= vmin:
                # Train stalled
                fail_code = 1
                vplus = vmin
                Fplus = Fb
            elif vplus_vec <= vb:
                # Constant TE region
                Fplus = Fb
                quadratic_flag = 1
            else:
                # Constant power region
                quadratic_flag = 0

        if  not fail_code and quadratic_flag:
            # Solve the following quadratic for vplus
            # vplus^2(1/U/h + c/2000) + (b/2000)vplus + [(gplus + g)/100
            #       + (a + b*v + c*v^2)/2000 + (a/2000) - v^2/U/h
            #       - Fplus/m - F/m] = 0
            q2 = (1/U/h + c/2000)
            q1 = (b/2000)
            v0sq = pow(v0,2)
            q0 = (gplus + g)/100 + (a + b*v0 + c*v0sq)/2000 + (a/2000) - v0sq/U/h -(Fplus + F)/m
            if q2 < 1e-5:
                fail_code = 2
                vplus = vmax
            else:
                vplus_vec = self.solve_cubic([0, q2, q1, q0])
                if vplus_vec <= vmin:
                    fail_code = 1
                    vplus = vmin
                else:
                    vplus = vplus_vec

        elif not fail_code:
            # Solve the following cubic:
            # vplus^3(1/U/h + c/2000) + (b/2000)vplus^2 + [(gplus + g)/100
            #       + (a + b*v + c*v^2)/2000 + (a/2000) - v^2/U/h
            #       - F_abe_plus/m - F/m]*vplus - 375*pplus/m = 0
            c3 = (1/U/h + c/2000)
            c2 = (b/2000)
            v0sq = pow(v0,2)
            c1 = (gplus + g)/100 + (a + b*v0 + c*v0sq)/2000 + (a/2000) - v0sq/U/h - F_abe_plus/m - F/m
            c0 = - 375*pplus/m
            if (c3 < 1e-5):
                fail_code = 2
                vplus = vmax
                Fplus = 375*pplus/vplus + F_abe_plus
            else:
                vplus = self.solve_cubic([c3, c2, c1, c0])
                Fplus = 375*pplus/vplus + F_abe_plus

        return vplus, pplus, fail_code

    def solve_cubic(self, c):
        """Solve cubic equation with real coefficients.
        
        Parameters
        ----------
        c : list of float
            Coefficients [c3, c2, c1, c0] for c3*x^3 + c2*x^2 + c1*x + c0 = 0
            
        Returns
        -------
        x : float
            Real root of the cubic equation
        """
        # small number, choose this to be the approximately machine precision
        small = 1e-15

        # Crop out elements of c that are too small
        for i, element in enumerate(c):
            if abs(c[i]) <= small:
                c[i] = 0

        x = -1

        maxc3 = 0
        for i in c[:3]:
            maxc3 = max(maxc3, abs(i))

        maxc2 = 0
        for i in c[:2]:
            maxc2 = max(maxc2, abs(i))

        if maxc3 == 0:
            pass
        elif maxc2 == 0:
            x = -c[3]/c[2]
        elif c[0] == 0:
            c1 = c[1]
            for i, element in enumerate(c):
                c[i] = c[i]/c1
            x = self.monic_quadratic(c[2], c[3])
        else:
            c0 = c[0]
            for i, element in enumerate(c):
                c[i] = c[i]/c0
            x = self.monic_cubic(c[1], c[2], c[3])

        return x

    def monic_cubic(self, c2, c1, c0):
        """Solve monic cubic equation x^3 + c2*x^2 + c1*x + c0 = 0.
        
        Parameters
        ----------
        c2, c1, c0 : float
            Coefficients of the monic cubic
            
        Returns
        -------
        x : float
            Real root of the cubic equation
        """
        # Initialize outputs
        x = -1
        niter = 0

        # convergence tolerance, abs and relative
        abs_prec = 1e-12
        rel_prec = 1e-12

        # Tiny number (to prevent division by zero). Coefficients b and
        # c are restricted to be larger than these values, or zero
        tiny = 1e-18

        # Max iterations
        MAXITER = 10

        # Rule out degenerate cases
        if abs(c2) < tiny:
            c2 = 0

        if abs(c1) < tiny:
            c1 = 0

        if abs(c0) < tiny:
            c0 = 0

        if (c0 == 0):
            # One root is zero, other two found via quadratic
            discr = c2*c2 - 4*c1
            if discr < tiny:
                x = 0
            else:
                x= max(0, (-c2 + np.sqrt(discr))/2)
            return x

        # Convert to cubic in canoical form
        if (c2 == 0):
            p = 0
            c1hat = c1
            c0hat = c0
        else:
            p = -c2/3
            c1hat = c1+p*(3*p+2*c2)
            c0hat = c0+p*(c1 + p*(c2 + p))

        # new cubic is x^3 + c1hat*x + c0hat
        if abs(c0hat) < tiny:
            if (c1hat < 0):
                x = np.sqrt(-c1hat)+p
            else:
                x = p
            return x

        if abs(c1hat) < tiny:
            x = pow(-c0hat,(1/3))+p
            return x

        if c1hat < 0:
            left_max = -np.sqrt(-c1hat/3)
            right_min = np.sqrt(-c1hat/3)
            right_min_val = c0hat + right_min*(c1hat + (right_min*right_min))
            if right_min_val < -tiny:
                # look to the right of right_min
                look_right = 1
                x0 = right_min
                a = c1hat
                b = c0hat
            elif right_min_val > tiny:
                # look to the left of left_max
                look_right = 0
                left_max_val = c0hat + left_max*(c1hat + (left_max*left_max))
                x0 = -left_max
                a = c1hat
                b = -c0hat
            else:
                x = right_min + p
                return x
        else:
            if c0hat < 0:
                look_right = 1
                x0 = 0
                a = c1hat
                b = c0hat
            else:
                look_right = 0
                x0 = 0
                a = c1hat
                b = -c0hat

            niter = 1

        if (x0 == 0):
            x1 = -b/a
        else:
            c1 = a/(3*x0) - x0
            c0 = x0*x0/3 + b/(3*x0)
            discr = c1*c1 - 4*c0
            if discr < tiny*tiny:
                x1 = -c1/2
            else:
                x1 = (-c1 + np.sqrt(discr))/2

        # backward cubic interpolation
        # accelerate convergence
        sum01 = (x0+x1)
        prod01 = x0*x1
        ssq01 = x0*x0 + x1*x1 + prod01
        ztemp = -b*(1 + a*sum01*prod01/ssq01/b)/(1 + a/ssq01)
        if ztemp > 0:
            z = pow(ztemp,(1/3))
        else:
            z = 0

        z = min(x1, max(z, x0))

        yz = b + z*(z*z + a)
        if yz > tiny:
            x1 = z
        else:
            x0 = z

        while ((x1-x0) > abs_prec)and(niter < MAXITER):
            if ((x1-x0)/max(abs(x0), abs(x1)) > rel_prec):

                # Secant
                x0 = (-b + x1*x0*(x0 + x1))/(x1*x1 + x0*x0 + x0*x1 + a)
                y0 = b + x0*(x0*x0 + a)
                if y0 > -tiny:
                    break
                # Quadratic
                c1 = a/(3*x0) - x0
                c0 = x0*x0/3 + b/(3*x0)
                discr = c1*c1 - 4*c0
                if discr < tiny*tiny:
                    x1 = -c1/2
                else:
                    x1 = (-c1 + np.sqrt(discr))/2
                niter = niter + 1
            else:
                break

        if look_right:
            x = x0 + p
        else:
            x = -x0 + p

        return x

    def monic_quadratic(self, b, c):
        """Solve monic quadratic equation x^2 + b*x + c = 0.
        
        Parameters
        ----------
        b, c : float
            Coefficients of the monic quadratic
            
        Returns
        -------
        x : float
            Positive real root of the quadratic equation
        """
        small = 1e-15
        tiny = 1e-18

        x = -1

        if abs(b) < tiny:
            b = 0

        if abs(c) < tiny:
            c = 0

        if b == 0:
            if c == 0:
                x = 0
                return x
            else:
                if c < 0:
                    x = np.sqrt(-c)
                    return x

        # General quadratic
        realrootflag = 1 - 4*c/b/b
        if realrootflag < -small:
            # Discriminant is definitely negative, no real roots
            return x
        elif realrootflag < small:
            # Repeated roots, or two complex roots with small imaginary part.
            # Imaginary part at most as large as abs(b)/2*sqrt(small)
            x = -b/2
        else:
            x = (-b + np.sqrt(b*b-4*c))/2

        return x


class LocomotiveModel:
    """Locomotive characteristics and force calculations.
    
    Models the performance characteristics of a 3-locomotive consist,
    including tractive effort and horsepower curves across notch settings.
    """
    
    def __init__(self):
        # These values are for a 3 locomotive consist
        self.thp_vec = [-14909, -12100, -9560, -7301, -5334, -3658, -2255, -1199, 0, 515, 1326, 2931, 4278, 5902, 7894, 9784, 11776]
        self.max_te_vec = [-206478, -167331, -129363, -91005, -59799, -35574, -18261, -10284, 0, 45000, 154287, 256713, 334287, 405000, 462858, 507858, 540000]

    def MaxTEForNotch(self, value):
        """Get maximum tractive effort for a given notch setting.
        
        Parameters
        ----------
        value : int
            Notch setting (-8 for dynamic brake 8, to +8 for notch 8)
            
        Returns
        -------
        float
            Maximum tractive effort in pounds
        """
        # Ex. Dynamic Brake 8 is -8, return value at index 0.
        # Ex. Motoring Notch 4 is 4, return value at index 12.
        return self.max_te_vec[value + 8]

    def THPForNotch(self, value):
        """Get throttle horsepower for a given notch setting.
        
        Parameters
        ----------
        value : int
            Notch setting (-8 for dynamic brake 8, to +8 for notch 8)
            
        Returns
        -------
        float
            Throttle horsepower
        """
        # Ex. Dynamic Brake 8 is -8, return value at index 0.
        # Ex. Motoring Notch 4 is 4, return value at index 12.
        return self.thp_vec[value + 8]

    def CommutationBreakpointSpeed(self):
        """Get commutation breakpoint speed.
        
        Returns
        -------
        float
            Commutation breakpoint speed in mph
        """
        return 70

    def MaxCommutationBrakingHP(self):
        """Get maximum commutation braking horsepower.
        
        Returns
        -------
        float
            Maximum commutation braking HP
        """
        return -13416.75819

    def MaxDynamicBrakingHP(self):
        """Get maximum dynamic braking horsepower.
        
        Returns
        -------
        float
            Maximum dynamic braking HP
        """
        return self.THPForNotch(-8)
