"""Geometry checks that do not require licensed scenes or evaluation records."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from render_hugsim_trajectories import Projection, image_points, near_clip, pose


class ProjectionTest(unittest.TestCase):
    def test_flat_road_known_projection_and_pose_invariance(self):
        intrinsic = dict(W=800,H=450,cx=400,cy=225,fovx=np.pi/2,fovy=np.pi/2)
        parameters = {name:dict(v2c=np.eye(4),intrinsic=intrinsic)
                      for name in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']}
        with tempfile.TemporaryDirectory() as directory:
            ground = Path(directory)/'ground.pkl'
            with ground.open('wb') as stream:
                pickle.dump((np.eye(4)[None],2.0,[0]),stream)
            config=dict(camera=dict(cam_rect=dict(rot=[0,0,0],trans=[0,.3,0])))
            base=dict(cam_params=parameters,ego_pos=[0,0,0],ego_rot=[0,0,0])
            projector=Projection(config,base,ground)
            # x forward, y left -> negative image x; road is 1.7 m below camera.
            trajectory=np.array([[10.,1.],[20.,-2.]])
            camera,intrinsic=projector.project(trajectory,base,projector.cameras[1])
            np.testing.assert_allclose(camera,[[-1,1.7,10],[2,1.7,20]],atol=1e-10)
            np.testing.assert_allclose(image_points(camera,intrinsic),[[360,263.25],[440,244.125]],atol=1e-10)
            moved=dict(base,ego_pos=[12.,0.,30.],ego_rot=[0,.6,0])
            transformed,_=projector.project(trajectory,moved,projector.cameras[1])
            np.testing.assert_allclose(transformed,camera,atol=1e-10)

    def test_near_plane_clipping_keeps_original_visible_vertices(self):
        polygon=np.array([[-1.,0.,-.5],[1.,0.,-.5],[1.,0.,2.],[-1.,0.,2.]])
        result=near_clip(polygon)
        self.assertEqual(result.shape,(4,3))
        self.assertTrue(np.all(result[:,2]>=.1-1e-12))
        for point in polygon[2:]:
            self.assertTrue(np.any(np.all(np.isclose(result,point),axis=1)))
        self.assertEqual(len(near_clip(polygon-[0,0,4])),0)

    def test_pose_roundtrip(self):
        transform=pose([.1,.3,-.2],[2.,-1.,8.])
        point=np.array([4.,1.,15.,1.])
        np.testing.assert_allclose(np.linalg.inv(transform)@(transform@point),point,atol=1e-12)


if __name__=='__main__':
    unittest.main()
