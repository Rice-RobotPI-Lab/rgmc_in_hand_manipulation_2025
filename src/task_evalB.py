#!/usr/bin/env python
import rospy
import rospkg

import time
import yaml
import math
import numpy as np
from functools import partial
from pyquaternion import Quaternion
from threading import Lock

from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose, PoseStamped
from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Image

from std_srvs.srv import Trigger, TriggerResponse

from task_viz import TaskViz

TASKB_TIME_LIMIT = 30
TASKB_ROT_THREDSHOLD = 0.5
TASKB1_TIME_PENALTY = 30

            
class TaskEvalB():
    def __init__(self):
        # get parameters
        self.task = rospy.get_param("task_evalB/task")
        self.cube_file = rospy.get_param("task_evalB/cube_file")
        self.task_file = rospy.get_param("task_evalB/task_file")
        self.is_record = rospy.get_param("task_evalB/is_record")
        self.task_visualize = TaskViz()
        rospy.on_shutdown(partial(self.handle_stop_task, None))
        
        self.state_lock = Lock()

        self.goal_id = -1 # not started
        self.total_score = 0
        self.current_pom = None
        self.current_normal = []

        self.time_limit = TASKB_TIME_LIMIT
        
        with open(self.cube_file) as f:
            try:
                self.cube_tags = {}
                self.face_id = {}
                tags = yaml.safe_load(f)["faces"]
                for i, face in enumerate(tags):
                    for tag in tags[face]["tag"]:
                        self.cube_tags[tag] = i
                    self.face_id[face] = i
                    # self.cube_normal[face] = tags[face]["normal"]
                
            except yaml.YAMLError as exc:
                rospy.logerr('Invalid Cube File')
                print(exc)
                return
            
        with open(self.task_file) as f:
            try:
                self.task_spec = yaml.safe_load(f)["faces"]
                self.num_waypoints = len(self.task_spec)
            except yaml.YAMLError as exc:
                rospy.logerr('Invalid Cube File')
                print(exc)
                return
            
        # setup subscriber
        self.tag_subsriber = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.callback_tag, queue_size=1)
        # setup publisher
        self.goal_pub = rospy.Publisher('/rgmc_eval/task2/goal', String, queue_size=1)
        self.msg_pub = rospy.Publisher('/rgmc_eval/msg', String, queue_size=1)
        # setup services
        self.start = rospy.Service('/rgmc_eval/start', Trigger, self.handle_start_task)
        self.record = rospy.Service('/rgmc_eval/record', Trigger, self.handle_record_waypoint)
        self.stop = rospy.Service('/rgmc_eval/stop', Trigger, self.handle_stop_task)

        if self.is_record:
            rospy.wait_for_service('/rgmc_recorder/start')
            rospy.wait_for_service('/rgmc_recorder/stop')
            self.start_service = rospy.ServiceProxy('/rgmc_recorder/start', Trigger)
            self.stop_service = rospy.ServiceProxy('/rgmc_recorder/stop', Trigger)

    def callback_tag(self, msg):
        self.state_lock.acquire()
        detections = msg.detections
        self.current_normal = [[]] * 6
        self.current_pom = None
        for tag in detections:
            if tag.id[0] in self.cube_tags:
                pose = tag.pose.pose.pose
                p = [pose.position.x, pose.position.y, pose.position.z]
                q = Quaternion(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
                n = list(q.rotate([0, 0, 1]))
                i = self.cube_tags[tag.id[0]]
                self.current_normal[i]= self.current_normal[i] + [p + n]

        self.task_visualize.viz_cube(self.current_normal)
        self.state_lock.release()

    def handle_start_task(self, _):
        if self.current_pom is None and len(self.current_normal) == 0:
            return TriggerResponse(False, "Unable to detect the object")
        elif self.goal_id != -1:
            return TriggerResponse(False, "The task is started")
        
        self.state_lock.acquire()

        if self.is_record:
            self.start_service()
        # Task B Start
        self.goal_id = 0
        self.total_success = 0

        self.start_time = time.time()
        self.pub_timer = rospy.Timer(rospy.Duration(1.0/10), self.handle_goal_pub, self.goal_id)
        self.exec_timer = rospy.Timer(rospy.Duration(self.time_limit), self.handle_timeout, oneshot=True)  
        
        self.state_lock.release()
        return TriggerResponse(True, "Start Task%d"%(self.task))

    def handle_record_waypoint(self, _):
        if self.goal_id < 0:
            return (False, "The TaskB%d is not started yet."%(self.task))
        
        self.state_lock.acquire()
        t = time.time() - self.start_time
        # Task 2 Record
        self.exec_timer.shutdown()
        
        i = self.face_id[self.task_spec[self.goal_id]]
        result = False
        if len(self.current_normal[i]) > 0:
            for j in range(len(self.current_normal[i])):
                d = np.arccos(np.dot(self.current_normal[i][j][3:], [0, 0, -1]))
                if d <= TASKB_ROT_THREDSHOLD:
                    result = True

        if self.task == 1:
            self.total_success += int(result) 
            msg = "Current Tartget [%s]\nResult [%s]\nRemaining Targets [%d]\nAccumulated Execution Time [%f] s"%(
                    self.task_spec[self.goal_id],"Success" if result else "Failure", self.num_waypoints - self.goal_id - 1, t)
            print(msg)
            print("---------------------")
        else:
            self.total_success += int(result) 
            msg = "Current Tartget [%s]\nResult [%s]\nProgress [%d/%d]\nRemaining Targets [%d]\nAccumulated Execution Time [%f] s"%(
                    self.task_spec[self.goal_id],"Success" if result else "Failure",  self.total_success, self.goal_id + 1, self.num_waypoints - self.goal_id - 1, t)
            print(msg)
            print("---------------------")

        self.msg_pub.publish(msg)

        self.goal_id += 1
        if self.goal_id >= self.num_waypoints or (self.task == 1 and not result):
            if self.task == 1:
                rospy.sleep(0.1)
                penalty_time = (self.num_waypoints - self.total_success) * TASKB1_TIME_PENALTY
                msg = "Execution Time [%f]\nPenalty time [%f]\nTotal Score [%f]"%(t, penalty_time, penalty_time + t)
                print(msg)
                print("---------------------")
                self.msg_pub.publish(msg)

            self.handle_stop_task(_)
            self.pub_timer.shutdown()
            self.state_lock.release()
            return TriggerResponse(True, msg)
        else:
            self.exec_timer = rospy.Timer(rospy.Duration(self.time_limit), self.handle_timeout, oneshot=True)
            self.state_lock.release()
            return TriggerResponse(True, msg)
        
    def handle_stop_task(self, _):
        if self.goal_id == -1:
            return TriggerResponse(False, "No Task Running")
        self.goal_id = -1
        self.pub_timer.shutdown()
        self.exec_timer.shutdown()
        if self.is_record:
            self.stop_service()
        return TriggerResponse(True, "Stopped Task")

    def handle_timeout(self, _):
        print("Time out for Waypoint [%d]"%(self.goal_id))
        self.handle_record_waypoint(None)
    
    # Publish goal state every 1/10 seconds
    def handle_goal_pub(self, _):
        s = String()
        s.data = self.task_spec[self.goal_id]
        self.goal_pub.publish(s)

if __name__ == "__main__":
    rospy.init_node("taskB evaluation", anonymous=True)
    rospy.sleep(0.5)
    node = TaskEvalB()
    rospy.spin()