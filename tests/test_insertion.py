"""Behavioral safety regression for the independent task and generic server."""
from types import SimpleNamespace
import multiprocessing as mp
import socket
import time
import unittest
from unittest import mock
import tempfile
import cv2
import mujoco
import numpy as np
from ur5e_sim.server.runtime import Controller, Link, dispatch, rpc
from ur5e_sim.config import load_settings, site_matrix, matrix_pose, pose_resources
from ur5e_sim.camera.calibration import load_spec
from ur5e_sim.vision.detect import detect_port
from ur5e_sim.tasks.insertion import Insertion, validated_world_pose, detect_in_worker
from ur5e_sim.control.kinematics import solve_ik


def slow_worker(*_): time.sleep(10)


class InsertionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings()
        cls.settings['tactile_align_enabled'] = False
        cls.model = mujoco.MjModel.from_xml_path(cls.settings['output_scene'])

    def setUp(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model,data,self.model.key('home').id)
        mujoco.mj_forward(self.model,data)
        self.c = Controller(self.model,data,SimpleNamespace(collision_samples=320,max_joint_speed=.75))

    def tearDown(self): self.c.close()
    def request(self, line): return dispatch(self.c, {'id':1,'command':line})

    def test_home_stable_and_grip(self):
        for _ in range(1000):
            self.c.apply_controls(); mujoco.mj_step(self.model,self.c.data)
            mujoco.mj_forward(self.model,self.c.data); self.c.after_step()
        from ur5e_sim.control.trajectory import unexpected_contact_count
        self.assertEqual(unexpected_contact_count(self.model, self.c.data), 0)
        self.assertTrue(np.all(np.isfinite(self.c.data.qpos)))
        plug_body = self.model.body('held_plug').id
        self.assertGreater(self.c.data.xpos[plug_body][2], 0.30,
                          "Plug should remain held in gripper above z=0.30")

    def test_release_plug_opens_gripper(self):
        self.assertTrue(self.c.plug_grasp_locked)
        rpc(self.c, {'id': 1, 'op': 'release_plug'})
        self.assertFalse(self.c.plug_grasp_locked)
        self.assertAlmostEqual(self.c.gripper_target, self.c.open_position, delta=1e-9)
        for _ in range(2000):
            self.c.apply_controls()
            mujoco.mj_step(self.model, self.c.data)
            self.c.after_step()
        self.assertEqual(self.c.gripper_state(), 'at_target')
        # After release, plug should have dropped (gravity on)
        plug_body = self.model.body('held_plug').id
        self.assertLess(self.c.data.xpos[plug_body][2], 0.30,
                       "Plug should fall after gripper opens")
        # Reset home should restore grip
        self.c.reset_to_home_keyframe()
        self.assertTrue(self.c.plug_grasp_locked)

    def test_slot_opening_and_back_wall(self):
        port = site_matrix(self.c.data,'socket_port'); alpha=self.model.geom_rgba[:,3].copy()
        try:
            self.model.geom_rgba[self.model.geom_group==3,3]=1
            for lateral, distance in [(0,.03),(.012,.01)]:
                hit=np.array([-1],np.int32)
                start=port[:3,3]-.01*port[:3,2]+lateral*port[:3,0]
                length=mujoco.mj_ray(self.model,self.c.data,start,port[:3,2].copy(),np.array([0,0,0,1,0,0],np.uint8),1,-1,hit)
                self.assertAlmostEqual(length,distance,delta=1e-7)
                self.assertTrue(self.model.geom(hit[0]).name.startswith('socket_wall_'))
        finally: self.model.geom_rgba[:,3]=alpha

    def test_task_stage_gates_and_server_has_no_task_commands(self):
        with tempfile.TemporaryDirectory() as output:
            task=Insertion(None,self.settings,'aruco',output)
            for method in [task.align,task.insert]:
                with self.assertRaises(ValueError): method()
        for cmd in ['align','insert','inspect aruco']: self.assertFalse(self.request(cmd)['ok'])
        self.assertIsNone(self.c.plan)

    def test_failed_ambiguous_planar_nonrigid_refused(self):
        snapshot={'qpos':self.c.data.qpos.tolist(),'T_world_camera':np.eye(4)}
        for status in ['ambiguous','failed','planar_only']:
            with self.assertRaises(ValueError): validated_world_pose({'status':status,'T_camera_port':np.eye(4)},snapshot,self.c.state())
        bad=np.eye(4);bad[0,0]=2
        with self.assertRaisesRegex(ValueError,'rigid'):
            validated_world_pose({'status':'ok','T_camera_port':bad},snapshot,self.c.state())

    def test_frame_composition_stale_and_horizontal_gate(self):
        camera=np.eye(4);camera[:3,3]=[.2,.3,.4]
        estimate=np.eye(4);estimate[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];estimate[:3,3]=[.01,.02,.28]
        snapshot={'qpos':self.c.data.qpos.tolist(),'T_world_camera':camera}
        result={'status':'ok','T_camera_port':estimate}
        np.testing.assert_allclose(validated_world_pose(result,snapshot,self.c.state()),camera@estimate)
        self.c.data.qpos[0]+=.01
        with self.assertRaisesRegex(ValueError,'stale'):validated_world_pose(result,snapshot,self.c.state())
        snapshot['qpos']=self.c.data.qpos.tolist()
        with self.assertRaisesRegex(ValueError,'horizontal'):validated_world_pose({'status':'ok','T_camera_port':np.eye(4)},snapshot,self.c.state())

    def test_bad_paths_are_atomic(self):
        before=self.c.data.ctrl.copy()
        for axis,amount,reason in [(0,100,'IK'),(2,-.2,'collision')]:
            target=site_matrix(self.c.data,'tcp');target[axis,3]+=amount
            with self.assertRaisesRegex(ValueError,reason):self.c.motion.execute([target])
            self.assertIsNone(self.c.plan);self.assertFalse(self.c.motion.busy)
            np.testing.assert_array_equal(self.c.data.ctrl,before)

    def test_stop_clears_queue_manual_motion_invalidates_capture(self):
        target=site_matrix(self.c.data,'tcp');target[2,3]+=.01
        self.c.motion.execute([target]);self.assertFalse(self.request('tcp-rel 0 0 .005')['ok'])
        self.c.stop();self.assertFalse(self.c.motion.queue);self.c.motion.tick();self.assertIsNone(self.c.plan)
        self.c.last_capture={'capture_id':'old'}
        self.assertTrue(self.request('tcp-rel 0 0 .005')['ok']);self.assertIsNone(self.c.last_capture)

    def test_contact_stops_motion(self):
        target=site_matrix(self.c.data,'tcp');target[2,3]-=.15
        with self.assertRaisesRegex(ValueError,'collision'):self.c.motion.execute([target])
        self.assertEqual(self.c.motion.phase,'idle');self.assertIsNone(self.c.plan)

    def test_image_failure_and_duplicate_ids(self):
        K=load_spec().intrinsic_matrix;white=np.full((1072,1280,3),255,np.uint8)
        for route in ['aruco','pnp']:self.assertEqual(detect_port(white,K,route,self.settings)['status'],'failed')
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        def marker(identifier,side=140):return np.repeat(cv2.aruco.generateImageMarker(dictionary,identifier,side)[:,:,None],3,axis=2)
        wrong=white.copy();wrong[100:240,100:240]=marker(1)
        self.assertEqual(detect_port(wrong,K,'aruco',self.settings)['status'],'failed')
        duplicate=white.copy();duplicate[100:240,100:240]=marker(0);duplicate[100:240,700:840]=marker(0)
        self.assertEqual(detect_port(duplicate,K,'aruco',self.settings)['status'],'ambiguous')
        small=white.copy();small[100:110,100:110]=marker(0,10)
        self.assertEqual(detect_port(small,K,'aruco',self.settings)['status'],'failed')
        blur=cv2.GaussianBlur(duplicate,(151,151),35)
        self.assertEqual(detect_port(blur,K,'aruco',self.settings)['status'],'failed')

    def test_disconnect_keeps_recovery_and_holds(self):
        target=site_matrix(self.c.data,'tcp')
        self.c.motion.execute([target],cartesian=True,recovery=target)
        link=Link(0,self.c);sock=socket.create_connection(link.listener.getsockname(),timeout=2)
        try:
            link.poll();sock.recv(8192);sock.close();link.poll()
            self.assertEqual(self.c.motion.phase,'stopped');self.assertTrue(self.c.motion.protected)
            self.assertFalse(self.request('tcp-rel 0 0 .005')['ok'])
            with self.assertRaises(ValueError):self.c.motion.execute([target])
            self.c.motion.recover();self.assertTrue(self.c.motion.busy)
        finally:sock.close();link.close()

    def test_timeout_and_disconnect_close_detection_worker(self):
        client=mock.Mock();settings=dict(self.settings,detection_timeout_s=.01)
        before={p.pid for p in mp.active_children()}
        with mock.patch('ur5e_sim.tasks.insertion.vision_worker',slow_worker):
            with self.assertRaises(TimeoutError):detect_in_worker(client,np.zeros((1,1,3),np.uint8),{'intrinsic_matrix':np.eye(3)},'aruco',settings)
            client.request.side_effect=ConnectionError('Disconnected')
            with self.assertRaises(ConnectionError):detect_in_worker(client,np.zeros((1,1,3),np.uint8),{'intrinsic_matrix':np.eye(3)},'aruco',self.settings)
        self.assertEqual({p.pid for p in mp.active_children()},before)

    def test_pnp_does_not_require_marker_settings(self):
        settings=dict(self.settings,aruco_config='/does/not/exist')
        config,_=pose_resources(settings,'pnp');self.assertIn('detection',config)

    def test_motion_snapshot_and_capture_generation_gate(self):
        q=self.c.data.qpos.tolist();target=site_matrix(self.c.data,'tcp').tolist()
        self.c.last_capture={'capture_id':'image','generation':0,'qpos':q}
        self.c.motion_generation=1
        with self.assertRaisesRegex(ValueError,'stale'):rpc(self.c,dict(op='motion',targets=[target],start_qpos=q,capture_id='image'))
        q[0]+=.01
        with self.assertRaisesRegex(ValueError,'stale'):rpc(self.c,dict(op='motion',targets=[target],start_qpos=q))

    def test_cartesian_spacing_speed_and_rotation_limits(self):
        start=site_matrix(self.c.data,'tcp');far=start.copy();far[2,3]+=.002
        with self.assertRaisesRegex(ValueError,'spacing'):self.c.motion.execute([far],cartesian=True)
        with self.assertRaisesRegex(ValueError,'speed'):self.c.motion.execute([start],cartesian=True,speed=.021)
        bad=start.copy();bad[:3,:3]=np.eye(3)
        with self.assertRaisesRegex(ValueError,'attitude'):self.c.motion.execute([bad],cartesian=True)

    def test_scene_parameter_mismatch_and_wrong_start_are_refused(self):
        with tempfile.TemporaryDirectory() as output:
            client=mock.Mock();client.state.return_value=self.c.state()
            changed=dict(self.settings,plug_mass_kg=.03)
            task=Insertion(client,changed,'aruco',output)
            with self.assertRaisesRegex(ValueError,'mismatch'):task.inspect()
            client.capture.assert_not_called();client.move.assert_not_called()
            state=self.c.state()
            state['tcp_position_m']=(np.asarray(state['tcp_position_m'])+np.array([.05,0.,0.])).tolist()
            client.state.return_value=state
            task=Insertion(client,self.settings,'aruco',output)
            with self.assertRaisesRegex(ValueError,'Home'):task.inspect()
            client.capture.assert_not_called();client.move.assert_not_called()

    def test_return_home_requires_completed_unchanged_retraction(self):
        with tempfile.TemporaryDirectory() as output:
            client=mock.Mock()
            task=Insertion(client,self.settings,'aruco',output)
            with self.assertRaisesRegex(ValueError,'finish retracting'):task.return_home()
            client.move.assert_not_called()
            state=self.c.state()
            state['motion'].update(phase='recovered',protected=False)
            task.phase='retracted'
            task.return_path=[site_matrix(self.c.data,'tcp')]
            task.report['phases']['retracted']=state
            for change in ('generation','qpos','protected','stopped'):
                import copy
                changed=copy.deepcopy(state)
                if change=='generation':changed['generation']+=1
                if change=='qpos':changed['qpos'][0]+=.01
                if change=='protected':changed['motion']['protected']=True
                if change=='stopped':changed['motion']['phase']='stopped'
                client.state.return_value=changed
                with self.assertRaisesRegex(ValueError,'state changed'):task.return_home()
                client.move.assert_not_called()

    def test_failed_retraction_never_runs_return_home(self):
        with tempfile.TemporaryDirectory() as output:
            client=mock.Mock()
            task=Insertion(client,self.settings,'aruco',output)
            for name in ('inspect','align','insert','release_plug'):setattr(task,name,mock.Mock())
            task.insert.side_effect=lambda: setattr(task,'phase','inserted')
            task.retract=mock.Mock(side_effect=RuntimeError('Retraction collision'))
            task.return_home=mock.Mock()
            with self.assertRaisesRegex(RuntimeError,'collision'):task.run()
            task.return_home.assert_not_called()
            client.stop.assert_called_once()
            self.assertEqual(task.report['status'],'failed')

    def test_speed_config_and_executor_agree_and_change_duration(self):
        import json
        from pathlib import Path
        start=site_matrix(self.c.data,'tcp');target=start.copy();target[2,3]+=.001
        durations={}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'task.json'
            for speed in (.005,.010,.015,.020):
                path.write_text(json.dumps(dict(self.settings,insert_speed_m_s=speed)))
                settings=load_settings(path)
                self.c.motion.execute([target],cartesian=True,speed=settings['insert_speed_m_s'])
                durations[speed]=self.c.motion.queue[0][2]
                self.c.stop()
            self.assertLess(durations[.015],durations[.005])
            for speed in (0,-.001,.020001,float('nan'),float('inf'),True,'0.015'):
                path.write_text(json.dumps(dict(self.settings,insert_speed_m_s=speed)))
                with self.assertRaisesRegex(ValueError,'insert_speed_m_s'):load_settings(path)
                with self.assertRaisesRegex(ValueError,'Cartesian speed'):
                    self.c.motion.execute([target],cartesian=True,speed=speed)
                self.assertFalse(self.c.motion.busy)

    def test_invalid_speed_cli_reports_field_without_traceback(self):
        import json,subprocess,sys
        from pathlib import Path
        from ur5e_sim.paths import ROOT
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'task.json'
            path.write_text(json.dumps(dict(self.settings,insert_speed_m_s=.021)))
            result=subprocess.run([sys.executable,str(ROOT/'apps/insert_socket.py'),'--config',str(path)],
                                  capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,1)
            self.assertIn('insert_speed_m_s',result.stdout)
            self.assertIn('0.02',result.stdout)
            self.assertNotIn('Traceback',result.stdout+result.stderr)
