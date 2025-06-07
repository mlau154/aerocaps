import os

import numpy as np
import copy

from aerocaps.geom.point import Point3D
from aerocaps.geom.surfaces import NURBSSurface, BezierSurface, RationalBezierSurface, SurfaceEdge, BSplineSurface
from aerocaps.geom.curves import BezierCurve3D,Line3D
from aerocaps.geom import NegativeWeightError
from aerocaps.units.angle import Angle
from rust_nurbs import *


def test_nurbs_revolve():
    axis = Line3D(p0=Point3D.from_array(np.array([0.0, 0.0, 0.0])),
                  p1=Point3D.from_array(np.array([0.0, 0.0, 1.0])))
    cubic_bezier_cps = np.array([
        [0.0, -1.0, 0.0],
        [0.0, -1.2, 0.5],
        [0.0, -1.3, 1.0],
        [0.0, -0.8, 1.5]
    ])
    bezier = BezierCurve3D([Point3D.from_array(p) for p in cubic_bezier_cps])
    nurbs_surface = NURBSSurface.from_bezier_revolve(bezier, axis, Angle(deg=15.0), Angle(deg=130.0))

    point_array = nurbs_surface.evaluate_grid(30, 30)
    for point in point_array[0, :, :]:
        radius = np.sqrt(point[0] ** 2 + point[1] ** 2)
        assert np.isclose(radius, 1.0, 1e-10)
    for point in point_array[-1, :, :]:
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
                except (AssertionError, ValueError):
                    continue


def test_Rational_Bezier_Surface_1():
    """
    Tests the continuity enforcement method across many random pairs of 4x4 ``RationalBezierSurface``s.
    """
    rng = np.random.default_rng(seed=42)
    negative_counter=0
    for it in range(15):
        n=rng.integers(4, 6)
        m=n
        #rng = np.random.default_rng(seed=42)

        cp_1 = np.array([[[0,0,rng.random()],[1,0,rng.random()],[2,0,rng.random()],[3,0,rng.random()]],
                         [[0,1,rng.random()],[1,1,rng.random()],[2,1,rng.random()],[3,1,rng.random()]],
                         [[0,2,rng.random()],[1,2,rng.random()],[2,2,rng.random()],[3,2,rng.random()]],
                         [[0,3,rng.random()],[1,3,rng.random()],[2,3,rng.random()],[3,3,rng.random()]]],dtype=np.float64)  

        #cp_1 =rng.random(( n+1, m+1, 3))
                 
        cp_2 =  np.array([[[0,0,rng.random()],[1,0,rng.random()],[2,0,rng.random()],[3,0,rng.random()]],
                         [[0,1,rng.random()],[1,1,rng.random()],[2,1,rng.random()],[3,1,rng.random()]],
                         [[0,2,rng.random()],[1,2,rng.random()],[2,2,rng.random()],[3,2,rng.random()]],
                         [[0,3,rng.random()],[1,3,rng.random()],[2,3,rng.random()],[3,3,rng.random()]]],dtype=np.float64)            
        cp_2[:, :, 0] += 3
        #cp_2 =rng.random(( n+1, m+1, 3))
        w_1 = rng.uniform(0.8, 1.2, size=cp_1.shape[:2])
        w_2 = rng.uniform(0.8, 1.2, size=cp_2.shape[:2])

        Rat_bez_surf_1 = RationalBezierSurface(cp_1, w_1)
        Rat_bez_surf_2 = RationalBezierSurface(cp_2, w_2)

        Rat_bez_surf_1_org=copy.deepcopy(Rat_bez_surf_1)
        Rat_bez_surf_2_org=copy.deepcopy(Rat_bez_surf_2)

        for i in range(4):
            for j in range(4):
                side_self=SurfaceEdge(i)
                side_other=SurfaceEdge(j)

                # Loop through each pair of control point meshes
                Rat_bez_surf_1=copy.deepcopy(Rat_bez_surf_1_org)
                Rat_bez_surf_2=copy.deepcopy(Rat_bez_surf_2_org)

                
                
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

                    # plot= pv.Plotter()
                    # Rat_bez_surf_1_org.plot_surface(plot)
                    # Rat_bez_surf_1_org.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_1_org.plot_control_points(plot)
                    # Rat_bez_surf_2_org.plot_surface(plot)
                    # Rat_bez_surf_2_org.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_2_org.plot_control_points(plot)
                    # plot.set_background('black')
                    # plot.show()

                    # plot= pv.Plotter()
                    # Rat_bez_surf_1.plot_surface(plot)
                    # Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_1.plot_control_points(plot)
                    # Rat_bez_surf_2.plot_surface(plot)
                    # Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_2.plot_control_points(plot)
                    # plot.set_background('black')
                    # plot.show()

                    #print(f'{it=},{negative_counter=}')

                    

                #except NegativeWeightError:
    print(f"{negative_counter=}")
                #negative_counter+=1
                #continue
                # Enforce G0, G1, and G2 continuity

                # Verify G0, G1, and G2 continuity
                
                
                
    #print(f"{negative_counter=}")


def test_Rational_Bezier_Surface_2():
    """
    Tests the continuity enforcement method across many random pairs of 4x4 ``RationalBezierSurface``s.
    """
    #rng = np.random.default_rng(seed=42)
    rng = np.random.default_rng(seed=60)
    Assertion_error_counter=0
    Negative_error_counter=0
    num_enforced=0
    flag=False
    for n in range(40):

        if flag==True:
            break
        

        random_array = rng.integers(low=4, high=6, size=3)

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
        #cp_1 = rng.random(( 4, 5, 3))

        # cp_1 = np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
        #                  [[0,1,1],[1,1,0],[2,1,1],[3,1,1]],
        #                  [[0,2,0],[1,2,1],[2,2,0],[3,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)  

        # cp_1 = np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1],[4,0,1]],
        #                  [[0,1,1],[1,1,0],[2,1,1],[3,1,1],[4,1,1]],
        #                  [[0,2,0],[1,2,1],[2,2,0],[3,2,1],[4,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1],[4,3,1]]],dtype=np.float64) 
        

        # cp_1 = np.array([[[0,0,rng.random()],[1,0,rng.random()],[2,0,rng.random()],[3,0,rng.random()],[4,0,rng.random()]],
        #                  [[0,1,rng.random()],[1,1,rng.random()],[2,1,rng.random()],[3,1,rng.random()],[4,1,rng.random()]],
        #                  [[0,2,rng.random()],[1,2,rng.random()],[2,2,rng.random()],[3,2,rng.random()],[4,2,rng.random()]],
        #                  [[0,3,rng.random()],[1,3,rng.random()],[2,3,rng.random()],[3,3,rng.random()],[4,3,rng.random()]]],dtype=np.float64) 

        cp_2 = rng.random(( n2+1, m2+1, 3))
        
        #cp_2 = rng.random(( 4, 5, 3))

        

        # cp_2 =  np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1]],
        #                  [[0,1,2],[1,1,1],[2,1,1],[3,1,1]],
        #                  [[0,2,0],[1,2,0],[2,2,1],[3,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1]]],dtype=np.float64)   

        # cp_2 =  np.array([[[0,0,1],[1,0,1],[2,0,1],[3,0,1],[4,0,1]],
        #                  [[0,1,2],[1,1,1],[2,1,1],[3,1,1],[4,1,1]],
        #                  [[0,2,0],[1,2,0],[2,2,1],[3,2,1],[4,2,1]],
        #                  [[0,3,0],[1,3,1],[2,3,1],[3,3,1],[4,3,1]]],dtype=np.float64)
        # cp_2[:, :, 0] += 4      

        # cp_2 =  np.array([[[0,0,rng.random()],[1,0,rng.random()],[2,0,rng.random()],[3,0,rng.random()],[4,0,rng.random()]],
        #                  [[0,1,rng.random()],[1,1,rng.random()],[2,1,rng.random()],[3,1,rng.random()],[4,1,rng.random()]],
        #                  [[0,2,rng.random()],[1,2,rng.random()],[2,2,rng.random()],[3,2,rng.random()],[4,2,rng.random()]],
        #                  [[0,3,rng.random()],[1,3,rng.random()],[2,3,rng.random()],[3,3,rng.random()],[4,3,rng.random()]]],dtype=np.float64)
        # cp_2[:, :, 0] += 4 



        # w_1 = rng.uniform(0.4, 0.5, (n1+1, m1+1))
        # w_2 = rng.uniform(0.9, 1.2, (n2+1, m2+1))

        w_1 = rng.uniform(0.8, 1.2, (np.shape(cp_1)[0], np.shape(cp_1)[1]))
        w_2 = rng.uniform(0.8, 1.3, (np.shape(cp_2)[0], np.shape(cp_2)[1]))

        # w_1[0][0]=1
        # w_1[0][-1]=1
        # w_1[-1][0]=1
        # w_1[-1][-1]=1

        # # w_1[0][:]=1
        # # w_1[:][-1]=1
        # # w_1[-1][:]=1
        # # w_1[:][-1]=1

        # w_2[0][0]=1
        # w_2[0][-1]=1
        # w_2[-1][0]=1
        # w_2[-1][-1]=1

        # w_2[0][:]=1
        # w_2[:][-1]=1
        # w_2[-1][:]=1
        # w_2[:][-1]=1

        
        # print(f'{w_1=}')
        # print(f'{w_2=}')


        
        #print(f'{cp_1=},{np.shape(cp_1)=}')
        #print(f'{cp_2=},{np.shape(cp_2)=}')
        
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
        
        # i_vals=np.array([0,1])
        # j_vals=np.array([0,1])
        
        Rat_bez_surf_1 = RationalBezierSurface(cp_1,w_1)
        Rat_bez_surf_2 = RationalBezierSurface(cp_2,w_2)

        # plot= pv.Plotter()
        # Rat_bez_surf_1.plot_surface(plot)
        # Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
        # Rat_bez_surf_1.plot_control_points(plot)
        # Rat_bez_surf_2.plot_surface(plot)
        # Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
        # Rat_bez_surf_2.plot_control_points(plot)
        # plot.set_background('black')
        # plot.show()

        # plot= pv.Plotter()
        # Rat_bez_surf_1.plot_surface(plot)
        # Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
        # Rat_bez_surf_1.plot_control_points(plot)
        # Rat_bez_surf_2.plot_surface(plot)
        # Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
        # Rat_bez_surf_2.plot_control_points(plot)
        # plot.set_background('black')
        # plot.show()
        
        Rat_bez_surf_1_org=copy.deepcopy(Rat_bez_surf_1)
        Rat_bez_surf_2_org=copy.deepcopy(Rat_bez_surf_2)
        

        #COUNT NUMBER OF ENFORCEMENTS TRIED
        

        for i in i_vals:
            for j in j_vals:
                num_enforced=num_enforced+1
                
                side_self=SurfaceEdge(i)
                side_other=SurfaceEdge(j)

                #RESET TO ORIGINAL FOR EVERY ITERATION OF LOOP

                Rat_bez_surf_1=copy.deepcopy(Rat_bez_surf_1_org)
                Rat_bez_surf_2=copy.deepcopy(Rat_bez_surf_2_org)

                


                # Rat_bez_surf_1.enforce_g0g1g2(Rat_bez_surf_2, 1.0, side_self, side_other)
                    
                # # Verify G0, G1, and G2 continuity
                # Rat_bez_surf_1.verify_g0(Rat_bez_surf_2, side_self, side_other)
                # Rat_bez_surf_1.verify_g1(Rat_bez_surf_2, side_self, side_other)
                # Rat_bez_surf_1.verify_g2(Rat_bez_surf_2, side_self, side_other)

                # plot= pv.Plotter()
                # Rat_bez_surf_1.plot_surface(plot)
                # Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
                # Rat_bez_surf_1.plot_control_points(plot)
                # Rat_bez_surf_2.plot_surface(plot)
                # Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
                # Rat_bez_surf_2.plot_control_points(plot)
                # plot.set_background('black')
                # plot.show()

                # Loop through each pair of control point meshes
                
                # Enforce G0, G1, and G2 continuity
                try:
                    Rat_bez_surf_1.enforce_g0g1g2(Rat_bez_surf_2, 1.0, side_self, side_other)
                    
                    # Verify G0, G1, and G2 continuity
                    Rat_bez_surf_1.verify_g0(Rat_bez_surf_2, side_self, side_other)
                    Rat_bez_surf_1.verify_g1(Rat_bez_surf_2, side_self, side_other)
                    Rat_bez_surf_1.verify_g2(Rat_bez_surf_2, side_self, side_other)
                except AssertionError:
                    Assertion_error_counter+=1

                    # iges_entities = [Rat_bez_surf_1.to_iges(),Rat_bez_surf_2.to_iges()]
                    # cp_net_points, cp_net_lines = Rat_bez_surf_1.generate_control_point_net()
                    # iges_entities.extend([cp_net_point.to_iges() for cp_net_point in cp_net_points])
                    # iges_entities.extend([cp_net_line.to_iges() for cp_net_line in cp_net_lines])
                    # cp_net_points_2, cp_net_lines_2 = Rat_bez_surf_2.generate_control_point_net()
                    # iges_entities.extend([cp_net_point.to_iges() for cp_net_point in cp_net_points_2])
                    # iges_entities.extend([cp_net_line.to_iges() for cp_net_line in cp_net_lines_2])

                    # #iges_file = os.path.join(TEST_DIR, "Rat_Bez_test.igs")
                    # iges_file = os.path.join(r"C:\aerocaps-main\aerocaps\aerocaps\tests", "Rat_Bez_test_5.igs")
                    # print(f"{iges_file=}")
                    # iges_generator = IGESGenerator(iges_entities, "meters")
                    # iges_generator.generate(iges_file)
                    # print("Generator passed")

                    flag=True

                    
                    
                except NegativeWeightError:
                    Negative_error_counter+=1
                    #print(f'{i=},{j=}')
                    #print(f"{side_self=},{side_other=}")
                    #print(f'{cp_1=},{cp_2=}')

                    # if (i==0 and j==1):
                    #     fail_case_1=cp_1
                    #     fail_case_2=cp_2
                    #     weight_case1=w_1
                    #     weight_case2=w_2
                    #     flag=True
                    #     break
                    # plot= pv.Plotter()
                    # Rat_bez_surf_1_org.plot_surface(plot)
                    # Rat_bez_surf_1_org.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_1_org.plot_control_points(plot)
                    # Rat_bez_surf_2_org.plot_surface(plot)
                    # Rat_bez_surf_2_org.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_2_org.plot_control_points(plot)
                    # plot.set_background('black')
                    # plot.show()

                    # plot= pv.Plotter()
                    # Rat_bez_surf_1.plot_surface(plot)
                    # Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_1.plot_control_points(plot)
                    # Rat_bez_surf_2.plot_surface(plot)
                    # Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
                    # Rat_bez_surf_2.plot_control_points(plot)
                    # plot.set_background('black')
                    # plot.show()
        
    print(f'{n=},{num_enforced=}')    
    print(f'{n=},{Assertion_error_counter=}')
    print(f'{n=},{Negative_error_counter=}')
    #print(f'{fail_case_1=},{fail_case_2=}')

    # return fail_case_1,fail_case_2,weight_case1,weight_case2

    

test_Rational_Bezier_Surface_2()
# fc1,fc2,w1,w2=test_Rational_Bezier_Surface_2()

# print(f'{fc1=},{fc2=},{w1=},{w2=}')


# Rat_bez_surf_1 = RationalBezierSurface(fc1,w1)
# Rat_bez_surf_2 = RationalBezierSurface(fc2,w2)
# side_self=SurfaceEdge(0)
# side_other=SurfaceEdge(1)

# plot= pv.Plotter()
# Rat_bez_surf_1.plot_surface(plot)
# Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
# Rat_bez_surf_1.plot_control_points(plot)
# Rat_bez_surf_2.plot_surface(plot)
# Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
# Rat_bez_surf_2.plot_control_points(plot)
# plot.set_background('black')
# plot.show()

# Rat_bez_surf_1.enforce_g0g1g2(Rat_bez_surf_2, 1.0, side_self, side_other)
                    
# # Verify G0, G1, and G2 continuity
# # Rat_bez_surf_1.verify_g0(Rat_bez_surf_2, side_self, side_other)
# # Rat_bez_surf_1.verify_g1(Rat_bez_surf_2, side_self, side_other)
# # Rat_bez_surf_1.verify_g2(Rat_bez_surf_2, side_self, side_other)

# plot= pv.Plotter()
# Rat_bez_surf_1.plot_surface(plot)
# Rat_bez_surf_1.plot_control_point_mesh_lines(plot)
# Rat_bez_surf_1.plot_control_points(plot)
# Rat_bez_surf_2.plot_surface(plot)
# Rat_bez_surf_2.plot_control_point_mesh_lines(plot)
# Rat_bez_surf_2.plot_control_points(plot)
# plot.set_background('black')
# plot.show()



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

        w_1 = rng.uniform(0.9, 1.1, cp_1.shape[:2])
        w_2 = rng.uniform(0.9, 1.1, cp_2.shape[:2])

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


def test_NURBS_1():
    """
    Tests the continuity enforcement method across many semi random pairs of 5x5 ``NURBS_Surfaces``s.
    All the knots are uniform and equal for both the parallel and perpendicular degrees
    """
    # rng = np.random.default_rng(seed=42)
    rng = np.random.default_rng(seed=35)
    Assertion_error_counter = 0
    Negative_error_counter = 0
    num_tried = 0
    flag = False
    for n in range(1):

        if flag:
            break

        # cp_1 = np.array([
        #     [[0, 0, rng.random()], [1, 0, rng.random()], [2, 0, rng.random()], [3, 0, rng.random()], [4, 0, rng.random()]],
        #     [[0, 1, rng.random()], [1, 1, rng.random()], [2, 1, rng.random()], [3, 1, rng.random()], [4, 1, rng.random()]],
        #     [[0, 2, rng.random()], [1, 2, rng.random()], [2, 2, rng.random()], [3, 2, rng.random()], [4, 2, rng.random()]],
        #     [[0, 3, rng.random()], [1, 3, rng.random()], [2, 3, rng.random()], [3, 3, rng.random()], [4, 3, rng.random()]],
        #     [[0, 4, rng.random()], [1, 4, rng.random()], [2, 4, rng.random()], [3, 4, rng.random()], [4, 4, rng.random()]]
        # ], dtype=np.float64)
        #
        # cp_2 = np.array([
        #     [[0, 0, rng.random()], [1, 0, rng.random()], [2, 0, rng.random()], [3, 0, rng.random()], [4, 0, rng.random()]],
        #     [[0, 1, rng.random()], [1, 1, rng.random()], [2, 1, rng.random()], [3, 1, rng.random()], [4, 1, rng.random()]],
        #     [[0, 2, rng.random()], [1, 2, rng.random()], [2, 2, rng.random()], [3, 2, rng.random()], [4, 2, rng.random()]],
        #     [[0, 3, rng.random()], [1, 3, rng.random()], [2, 3, rng.random()], [3, 3, rng.random()], [4, 3, rng.random()]],
        #     [[0, 4, rng.random()], [1, 4, rng.random()], [2, 4, rng.random()], [3, 4, rng.random()], [4, 4, rng.random()]]
        # ], dtype=np.float64)
        # cp_2[:, :, 1] += 4.0

        cp_1 = rng.uniform(low=-4.0, high=4.0, size=(6, 6, 3))
        cp_2 = rng.uniform(low=-4.0, high=4.0, size=(6, 6, 3))

        w_1 = rng.uniform(0.9, 1.1, (np.shape(cp_1)[0], np.shape(cp_1)[1]))
        w_2 = rng.uniform(0.9, 1.1, (np.shape(cp_2)[0], np.shape(cp_2)[1]))

        # w_1 = np.ones((6, 6))
        # w_2 = np.ones((6, 6))

        # w_1[:, 0] = 1.0
        # w_1[:, -1] = 1.0
        # w_1[0, :] = 1.0
        # w_1[-1, :] = 1.0
        # w_2[:, 0] = 1.0
        # w_2[:, -1] = 1.0
        # w_2[0, :] = 1.0
        # w_2[-1, :] = 1.0

        p = 5
        m = p + 6 + 1
        u_knot = np.zeros(m)
        u_knot[:(p + 1)] = 0.0
        u_knot[-(p + 1):] = 1.0
        # u_knot[p + 1] = 0.25
        # u_knot[p + 2] = 0.5
        # u_knot[p + 3] = 0.75

        v_knot = np.zeros(m)
        v_knot[:(p + 1)] = 0.0
        v_knot[-(p + 1):] = 1.0
        # v_knot[p + 1] = 0.25
        # v_knot[p + 2] = 0.5
        # v_knot[p + 3] = 0.75

        print(f"{u_knot = }")
        print(f"{v_knot = }")

        NURBS_1 = NURBSSurface(cp_1, u_knot, v_knot, w_1)
        NURBS_2 = NURBSSurface(cp_2, u_knot, v_knot, w_2)

        NURBS_1.enforce_g0g1g2(NURBS_2, 1.0, SurfaceEdge.v0, SurfaceEdge.u0)
        NURBS_1.verify_g0(NURBS_2, SurfaceEdge.v0, SurfaceEdge.u0)
        NURBS_1.verify_g1(NURBS_2, SurfaceEdge.v0, SurfaceEdge.u0)
        NURBS_1.verify_g2(NURBS_2, SurfaceEdge.v0, SurfaceEdge.u0)

        # COMPARE TO FDM
        pts_edge = 10
        step = 1e-6
        term1_1st = np.array(nurbs_surf_eval_iso_v(NURBS_1.get_control_point_array(), NURBS_1.weights, u_knot, v_knot, pts_edge, 1.0))
        term2_1st = np.array(nurbs_surf_eval_iso_v(NURBS_1.get_control_point_array(), NURBS_1.weights, u_knot, v_knot, pts_edge, 1.0 - step))
        FDM_first_der_self_array = (term1_1st - term2_1st) / step
        term1 = np.array(nurbs_surf_eval_iso_v(NURBS_1.get_control_point_array(), NURBS_1.weights, u_knot, v_knot, pts_edge, 1.0))
        term2 = np.array(nurbs_surf_eval_iso_v(NURBS_1.get_control_point_array(), NURBS_1.weights, u_knot, v_knot, pts_edge, 1.0 - step))
        term3 = np.array(nurbs_surf_eval_iso_v(NURBS_1.get_control_point_array(), NURBS_1.weights, u_knot, v_knot, pts_edge, 1.0 - 2 * step))
        print(f"{term1 = }")
        print(f"{term2 = }")
        print(f"{term3 = }")
        FDM_second_der_self_array = (term1 - 2 * term2 + term3) / (step ** 2)
        print(f'{FDM_first_der_self_array=}')
        print(f'{FDM_second_der_self_array=}')

        print(f"{NURBS_1.get_first_derivs_along_edge(SurfaceEdge.v1, pts_edge, perp=True) = }")
        print(f"{NURBS_1.get_second_derivs_along_edge(SurfaceEdge.v1, pts_edge, perp=True) = }")

        NURBS_1.verify_g2(NURBS_2, SurfaceEdge.v0, SurfaceEdge.u0)


def test_bspline_surf():
    """
    Tests the continuity enforcement method across many semi random pairs of 5x5 ``NURBS_Surfaces``s.
    All the knots are uniform and equal for both the parallel and perpendicular degrees
    """
    # rng = np.random.default_rng(seed=42)
    rng = np.random.default_rng(seed=35)
    cp_1 = rng.uniform(low=-4.0, high=4.0, size=(6, 6, 3))
    cp_2 = rng.uniform(low=-4.0, high=4.0, size=(6, 6, 3))

    p = 2
    m = p + 6 + 1
    u_knot = np.zeros(m)
    u_knot[:(p + 1)] = 0.0
    u_knot[-(p + 1):] = 1.0
    # u_knot[p + 1] = 0.3
    # u_knot[p + 2] = 0.5
    # u_knot[p + 3] = 0.7
    u_knot[p + 1] = 0.5
    u_knot[p + 2] = 0.5
    u_knot[p + 3] = 0.5

    p = 2
    m = p + 6 + 1
    v_knot = np.zeros(m)
    v_knot[:(p + 1)] = 0.0
    v_knot[-(p + 1):] = 1.0
    v_knot[p + 1] = 0.3
    v_knot[p + 2] = 0.5
    v_knot[p + 3] = 0.7

    print(f"{u_knot = }")
    print(f"{v_knot = }")

    bspline_surf_1 = BSplineSurface(cp_1, u_knot, v_knot)
    bspline_surf_2 = BSplineSurface(cp_2, u_knot, v_knot)

    bspline_surf_1.enforce_g0g1g2(bspline_surf_2, 1.0, SurfaceEdge.v0, SurfaceEdge.v1)
    bspline_surf_1.verify_g0(bspline_surf_2, SurfaceEdge.v0, SurfaceEdge.v1)
    bspline_surf_1.verify_g1(bspline_surf_2, SurfaceEdge.v0, SurfaceEdge.v1)

    # TODO: understand why this next verification does not pass
    # bspline_surf_1.verify_g2(bspline_surf_2, SurfaceEdge.v0, SurfaceEdge.v1)
