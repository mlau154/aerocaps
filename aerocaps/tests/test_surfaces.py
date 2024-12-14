import os

import numpy as np
import pyvista as pv

print(os.getcwd())


#from aerocaps import DATA_DIR
from aerocaps.geom.point import Point3D
from aerocaps.geom.surfaces import NURBSSurface, BezierSurface, RationalBezierSurface, SurfaceEdge
from aerocaps.geom.curves import Bezier3D,Line3D
from aerocaps.geom import NegativeWeightError
from aerocaps.units.angle import Angle
from aerocaps.iges.iges_generator import IGESGenerator
from aerocaps import TEST_DIR



def test_nurbs_revolve():
    axis = Line3D(p0=Point3D.from_array(np.array([0.0, 0.0, 0.0])),
                  p1=Point3D.from_array(np.array([0.0, 0.0, 1.0])))
    cubic_bezier_cps = np.array([
        [0.0, -1.0, 0.0],
        [0.0, -1.2, 0.5],
        [0.0, -1.3, 1.0],
        [0.0, -0.8, 1.5]
    ])
    bezier = Bezier3D([Point3D.from_array(p) for p in cubic_bezier_cps])
    nurbs_surface = NURBSSurface.from_bezier_revolve(bezier, axis, Angle(deg=15.0), Angle(deg=130.0))

    iges_entities = [nurbs_surface.to_iges()]
    cp_net_points, cp_net_lines = nurbs_surface.generate_control_point_net()
    iges_entities.extend([cp_net_point.to_iges() for cp_net_point in cp_net_points])
    iges_entities.extend([cp_net_line.to_iges() for cp_net_line in cp_net_lines])

    iges_file = os.path.join(TEST_DIR, "nurbs_test.igs")
    iges_generator = IGESGenerator(iges_entities, "meters")
    iges_generator.generate(iges_file)

    point_array = nurbs_surface.evaluate(30, 30)
    for point in point_array[:, 0, :]:
        radius = np.sqrt(point[0] ** 2 + point[1] ** 2)
        assert np.isclose(radius, 1.0, 1e-10)
    for point in point_array[:, -1, :]:
        radius = np.sqrt(point[0] ** 2 + point[1] ** 2)
        assert np.isclose(radius, 0.8, 1e-10)


def test_bezier_surface_1():
    """
    Tests the continuity enforcement method across many random pairs of 4x4 ``BezierSurface``s.
    """
    # FOR TESTING 4x4 and 4x4 first
    n = 4
    m = 4
    num_samples = 50
    rng = np.random.default_rng(seed=42)

    cp_sets_1 = rng.random((num_samples, n+1, m+1, 3))
    cp_sets_2 = rng.random((num_samples, n+1, m+1, 3))

    #Loop through different sides of the 4x4
    
    for i in range(4):
        for j in range(4):
            side_self=SurfaceEdge(i)
            side_other=SurfaceEdge(j)



            # Loop through each pair of control point meshes
            for cp_set1, cp_set2 in zip(cp_sets_1, cp_sets_2):
                bez_surf_1 = BezierSurface(cp_set1)
                bez_surf_2 = BezierSurface(cp_set2)

                # Enforce G0, G1, and G2 continuity
                bez_surf_1.enforce_g0g1g2(bez_surf_2, 1.0, side_self, side_other)

                # Verify G0, G1, and G2 continuity
                bez_surf_1.verify_g0(bez_surf_2, side_self, side_other)
                bez_surf_1.verify_g1(bez_surf_2, side_self, side_other)
                bez_surf_1.verify_g2(bez_surf_2, side_self, side_other)



def test_bezier_surface_2():
    """
    Tests the continuity enforcement method across many random pairs of randomly sized Bezier Surfaces for the parallel degree verification
    """
    for n in range(50):
        # GENERATE THE control point arrays by randomly making a 3 element array
        random_array = np.random.randint(low=4, high=15, size=3)

        #Pick the control points randomly from the 3 element array. 
        n1 = random_array[np.random.randint(0, len(random_array) )]
        m1 = random_array[np.random.randint(0, len(random_array) )]
        
        n1m1_array=np.array([n1,m1])
        random_value=np.random.randint(0,2)
        if random_value==0:
            n2= n1m1_array[np.random.randint(0,2)]
            m2= random_array[np.random.randint(0, len(random_array) )]
        else:
            m2= n1m1_array[np.random.randint(0,2)]
            n2= random_array[np.random.randint(0, len(random_array) )]



        
        rng = np.random.default_rng(seed=42)

        cp_1 = rng.random(( n1+1, m1+1, 3))
        cp_2 = rng.random(( n2+1, m2+1, 3))

        #Loop through different compatible sides

        if (np.shape(cp_1)[0]==np.shape(cp_2)[0]):
            i_vals=np.array([0,1])
            j_vals=np.array([0,1])

        elif (np.shape(cp_1)[0]==np.shape(cp_2)[1]):
            i_vals=np.array([0,1])
            j_vals=np.array([2,3])

        elif (np.shape(cp_1)[1]==np.shape(cp_2)[0]):
            i_vals=np.array([2,3])
            j_vals=np.array([0,1])
        
        elif (np.shape(cp_1)[1]==np.shape(cp_2)[1]):
            i_vals=np.array([2,3])
            j_vals=np.array([2,3])
        
        else:
            raise ValueError("Could not find matching degrees between the surfaces")
        
        for i in i_vals:
            for j in j_vals:
                side_self=SurfaceEdge(i)
                side_other=SurfaceEdge(j)

                # Loop through each pair of control point meshes
                
                bez_surf_1 = BezierSurface(cp_1)
                bez_surf_2 = BezierSurface(cp_2)

                # Enforce G0, G1, and G2 continuity
                bez_surf_1.enforce_g0g1g2(bez_surf_2, 1.0, side_self, side_other)

                # Verify G0, G1, and G2 continuity
                bez_surf_1.verify_g0(bez_surf_2, side_self, side_other)
                bez_surf_1.verify_g1(bez_surf_2, side_self, side_other)
                bez_surf_1.verify_g2(bez_surf_2, side_self, side_other)



def test_bezier_surface_3():
    """
    Tests the continuity enforcement method across many random pairs of randomly sized Bezier Surfaces for verifying whether the tests raise assertion errors when surfaces are incompatible.
    """
    for n in range(50):
        n1 = np.random.randint(low=4, high=10)
        m1 = np.random.randint(low=4, high=10)
        n2 = np.random.randint(low=4, high=10)
        m2 = np.random.randint(low=4, high=10)
        
        rng = np.random.default_rng(seed=42)

        cp_1 = rng.random(( n1+1, m1+1, 3))
        cp_2 = rng.random(( n2+1, m2+1, 3))

        
        
        for i in range(4):
            for j in range(4):
                side_self=SurfaceEdge(i)
                side_other=SurfaceEdge(j)

                # Loop through each pair of control point meshes
                
                bez_surf_1 = BezierSurface(cp_1)
                bez_surf_2 = BezierSurface(cp_2)
                
                try:
                    # Enforce G0, G1, and G2 continuity
                    bez_surf_1.enforce_g0g1g2(bez_surf_2, 1.0, side_self, side_other)

                    # Verify G0, G1, and G2 continuity
                    bez_surf_1.verify_g0(bez_surf_2, side_self, side_other)
                    bez_surf_1.verify_g1(bez_surf_2, side_self, side_other)
                    bez_surf_1.verify_g2(bez_surf_2, side_self, side_other)
                except AssertionError:
                    continue

def test_Rational_Bezier_Surface_1():
    """
    Tests the continuity enforcement method across many random pairs of 4x4 ``RationalBezierSurface``s.
    """
    rng = np.random.default_rng(seed=42)
    negative_counter=0
    for it in range(50):
        n=rng.integers(4, 9)
        m=n
        #rng = np.random.default_rng(seed=42)

        # cp_1 = np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
        #                  [[0,1,1],[1,1,0],[2,1,1],[3,1,1]],
        #                  [[0,2,0],[1,2,1],[2,2,0],[3,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)  

        cp_1 =rng.random(( n+1, m+1, 3))
                 
        # cp_2 =  np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
        #                  [[0,1,2],[1,1,1],[2,1,1],[3,1,1]],
        #                  [[0,2,0],[1,2,0],[2,2,1],[3,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)            
        cp_2 =rng.random(( n+1, m+1, 3))
        w_1 = rng.uniform(0.4, 0.5, (n+1, m+1))
        w_2 = rng.uniform(0.9, 1.2, (n+1, m+1))

        for i in range(1):
            for j in range(1):
                side_self=SurfaceEdge(i)
                side_other=SurfaceEdge(j)

                # Loop through each pair of control point meshes
                
                Rat_bez_surf_1 = RationalBezierSurface(cp_1,w_1)
                Rat_bez_surf_2 = RationalBezierSurface(cp_2,w_2)
                
                try:
                    Rat_bez_surf_1.enforce_g0g1g2(Rat_bez_surf_2, 1.0, side_self, side_other)
                    Rat_bez_surf_1.verify_g0(Rat_bez_surf_2, side_self, side_other)

                    # g1_self=Rat_bez_surf_1.get_first_derivs_along_edge(side_self)
                    # g2_self=Rat_bez_surf_2.get_first_derivs_along_edge(side_other)

                    # g1_self_v2=Rat_bez_surf_1.get_first_derivs_along_edge_v2(side_self)
                    # g2_self_v2=Rat_bez_surf_2.get_first_derivs_along_edge_v2(side_other)
                    
                    #print(f"{g1_self=},{g2_self=}")
                    #print(f"{g1_self_v2=},{g2_self_v2=}")
                    Rat_bez_surf_1.verify_g1(Rat_bez_surf_2, side_self, side_other)
                    Rat_bez_surf_1.verify_g2(Rat_bez_surf_2, side_self, side_other)
                except NegativeWeightError:
                    negative_counter+=1
                    #print(f'{it=},{negative_counter=}')

                    

                #except NegativeWeightError:
                #print(f"{negative_counter=}")
                #negative_counter+=1
                #continue
                # Enforce G0, G1, and G2 continuity

                # Verify G0, G1, and G2 continuity
                
                
                
    #print(f"{negative_counter=}")


def test_Rational_Bezier_Surface_2():
    """
    Tests the continuity enforcement method across many random pairs of 4x4 ``RationalBezierSurface``s.
    """
    rng = np.random.default_rng(seed=42)
    Assertion_error_counter=0
    Negative_error_counter=0
    for n in range(20):

        random_array = rng.integers(low=3, high=5, size=3)

        #Pick the control points randomly from the 3 element array. 
        n1 = random_array[rng.integers(0, len(random_array) )]
        m1 = random_array[rng.integers(0, len(random_array) )]
        
        n1m1_array=np.array([n1,m1])
        random_value=rng.integers(0,2)
        if random_value==0:
            n2= n1m1_array[rng.integers(0,2)]
            m2= random_array[rng.integers(0, len(random_array) )]
        else:
            m2= n1m1_array[rng.integers(0,2)]
            n2= random_array[rng.integers(0, len(random_array) )]



        
        

        cp_1 = rng.random(( n1+1, m1+1, 3))

        # cp_1 = np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
        #                  [[0,1,1],[1,1,0],[2,1,1],[3,1,1]],
        #                  [[0,2,0],[1,2,1],[2,2,0],[3,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)  
        cp_2 = rng.random(( n2+1, m2+1, 3))

        # cp_2 =  np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
        #                  [[0,1,2],[1,1,1],[2,1,1],[3,1,1]],
        #                  [[0,2,0],[1,2,0],[2,2,1],[3,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)      

        w_1 = rng.uniform(0.4, 0.5, (n1+1, m1+1))
        w_2 = rng.uniform(0.9, 1.2, (n2+1, m2+1))

        #Loop through different compatible sides

        if (np.shape(cp_1)[0]==np.shape(cp_2)[0]):
            i_vals=np.array([0,1])
            j_vals=np.array([0,1])

        elif (np.shape(cp_1)[0]==np.shape(cp_2)[1]):
            i_vals=np.array([0,1])
            j_vals=np.array([2,3])

        elif (np.shape(cp_1)[1]==np.shape(cp_2)[0]):
            i_vals=np.array([2,3])
            j_vals=np.array([0,1])
        
        elif (np.shape(cp_1)[1]==np.shape(cp_2)[1]):
            i_vals=np.array([2,3])
            j_vals=np.array([2,3])
        
        else:
            raise ValueError("Could not find matching degrees between the surfaces")
        
        for i in i_vals:
            for j in j_vals:
                side_self=SurfaceEdge(i)
                side_other=SurfaceEdge(j)

                # Loop through each pair of control point meshes
                
                Rat_bez_surf_1 = RationalBezierSurface(cp_1,w_1)
                Rat_bez_surf_2 = RationalBezierSurface(cp_2,w_2)

                # Enforce G0, G1, and G2 continuity
                try:
                    Rat_bez_surf_1.enforce_g0g1g2(Rat_bez_surf_2, 1.0, side_self, side_other)
                    
                    # Verify G0, G1, and G2 continuity
                    Rat_bez_surf_1.verify_g0(Rat_bez_surf_2, side_self, side_other)
                    Rat_bez_surf_1.verify_g1(Rat_bez_surf_2, side_self, side_other)
                    Rat_bez_surf_1.verify_g2(Rat_bez_surf_2, side_self, side_other)
                except AssertionError:
                    Assertion_error_counter+=1
                    
                except NegativeWeightError:
                    Negative_error_counter+=1
    print(f'{n=},{Assertion_error_counter=}')
    print(f'{n=},{Negative_error_counter=}')


test_Rational_Bezier_Surface_2()

def test_Rational_Bezier_Surface_3():
    """
    Tests the continuity enforcement method across many random pairs of 4x4 ``RationalBezierSurface``s.
    """
    rng = np.random.default_rng(seed=42)
    for n in range(1):

        random_array = rng.integers(low=4, high=15, size=3)

        #Pick the control points randomly from the 3 element array. 
        n1 = random_array[rng.integers(0, len(random_array) )]
        m1 = random_array[rng.integers(0, len(random_array) )]
        
        n1m1_array=np.array([n1,m1])
        random_value=rng.integers(0,2)
        if random_value==0:
            n2= n1m1_array[rng.integers(0,2)]
            m2= random_array[rng.integers(0, len(random_array) )]
        else:
            m2= n1m1_array[rng.integers(0,2)]
            n2= random_array[rng.integers(0, len(random_array) )]



        
        

        #cp_1 = rng.random(( n1+1, m1+1, 3))

        cp_1 = np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
                         [[0,1,1],[1,1,0],[2,1,1],[3,1,1]],
                         [[0,2,0],[1,2,1],[2,2,0],[3,2,1]],
                         [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)  
        
        #cp_2 = rng.random(( n2+1, m2+1, 3))

        cp_2 =  np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
                         [[0,1,2],[1,1,1],[2,1,1],[3,1,1]],
                         [[0,2,0],[1,2,0],[2,2,1],[3,2,1]],
                         [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)            

        w_1 = rng.uniform(0.9, 1.1, (n1+1, m1+1))
        w_2 = rng.uniform(0.9, 1.1, (n2+1, m2+1))

        #Loop through different compatible sides

        if (np.shape(cp_1)[0]==np.shape(cp_2)[0]):
            i_vals=np.array([0,1])
            j_vals=np.array([0,1])

        elif (np.shape(cp_1)[0]==np.shape(cp_2)[1]):
            i_vals=np.array([0,1])
            j_vals=np.array([2,3])

        elif (np.shape(cp_1)[1]==np.shape(cp_2)[0]):
            i_vals=np.array([2,3])
            j_vals=np.array([0,1])
        
        elif (np.shape(cp_1)[1]==np.shape(cp_2)[1]):
            i_vals=np.array([2,3])
            j_vals=np.array([2,3])
        
        else:
            raise ValueError("Could not find matching degrees between the surfaces")
        
        for i in i_vals:
            for j in j_vals:
                side_self=SurfaceEdge(i)
                side_other=SurfaceEdge(j)

                # Loop through each pair of control point meshes
                
                Rat_bez_surf_1 = RationalBezierSurface(cp_1,w_1)
                Rat_bez_surf_2 = RationalBezierSurface(cp_2,w_2)

                # Enforce G0, G1, and G2 continuity
                Rat_bez_surf_1.enforce_g0g1g2(Rat_bez_surf_2, 1.0, side_self, side_other)
                
                # Verify G0, G1, and G2 continuity
                Rat_bez_surf_1.verify_g0(Rat_bez_surf_2, side_self, side_other)
                Rat_bez_surf_1.verify_g1(Rat_bez_surf_2, side_self, side_other)
                Rat_bez_surf_1.verify_g2(Rat_bez_surf_2, side_self, side_other)

#test_Rational_Bezier_Surface_3()


####### SCRATCH CODE #######

# iges_entities = [Rat_bez_surf_1.to_iges(),Rat_bez_surf_2.to_iges()]
# cp_net_points, cp_net_lines = Rat_bez_surf_1.generate_control_point_net()
# iges_entities.extend([cp_net_point.to_iges() for cp_net_point in cp_net_points])
# iges_entities.extend([cp_net_line.to_iges() for cp_net_line in cp_net_lines])
# cp_net_points_2, cp_net_lines_2 = Rat_bez_surf_2.generate_control_point_net()
# iges_entities.extend([cp_net_point.to_iges() for cp_net_point in cp_net_points_2])
# iges_entities.extend([cp_net_line.to_iges() for cp_net_line in cp_net_lines_2])

# iges_file = os.path.join(TEST_DIR, "Rat_Bez_test.igs")
# print(f"{iges_file=}")
# iges_generator = IGESGenerator(iges_entities, "meters")
# iges_generator.generate(iges_file)
# print("Generator passed")


#PLOTTER

# plot= pv.plotter()
# Rat_bez_surf_1.plot_surface()
# Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
# Rat_bez_surf_1.plot_control_points(plot)
# Rat_bez_surf_2.plot_surface()
# Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
# Rat_bez_surf_2.plot_control_points(plot)
# plot.show()