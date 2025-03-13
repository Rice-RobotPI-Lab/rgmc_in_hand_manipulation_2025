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

TASKA_TIME_LIMIT = 20
TASKA1_DISTANCE_TOLERANCE = 0.5
TASKA1_TIME_PENALTY = 20
TASKA2_DISTANCE_PENALTY = 10

class TaskEvalA():
    def __init__(self):
        # get parameters
        self.task = rospy.get_param("task_evalA/task")
        self.tag_id = rospy.get_param("task_evalA/tag")
        self.task_file = rospy.get_param("task_evalA/task_file")
        self.is_record = rospy.get_param("task_evalA/is_record")
        self.task_visualize = TaskViz()
        rospy.on_shutdown(partial(self.handle_stop_task, None))
        
        self.state_lock = Lock()

        self.goal_id = -1 # not started
        self.total_score = 0
        self.current_pom = None
        self.current_normal = []

        self.time_limit = TASKA_TIME_LIMIT
        with open(self.task_file) as f:
            try:
                self.task_spec = yaml.safe_load(f)["waypoints"]
                self.num_waypoints = len(self.task_spec)
            except yaml.YAMLError as exc:
                rospy.logerr('Invalid Task File')
                print(exc)
                return
            
        # setup subscriber
        self.tag_subsriber = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.callback_tag, queue_size=1)
        # setup publisher
        self.goal_pub = rospy.Publisher('/rgmc_eval/task1/goal', Point, queue_size=1)
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
            if self.tag_id in tag.id:
                self.current_pom = tag.pose.pose.pose
        self.state_lock.release()

    def viz_task1(self):
        self.task_visualize.viz_traj(self.all_traj)
        self.task_visualize.viz_waypoint(self.all_waypoint)
        self.task_visualize.viz_goal(self.goal)

    def handle_start_task(self, _):
        if self.current_pom is None:# and len(self.current_normal) == 0:
            return TriggerResponse(False, "Unable to detect the object")
        elif self.goal_id != -1:
            return TriggerResponse(False, "The task is started")
        
        self.state_lock.acquire()
        
        if self.is_record:
            self.start_service()

        # Task A Start
        self.goal_id = 0
        cur = self.current_pom
        self.start_pose = [cur.position.x, cur.position.y, cur.position.z]
        self.goal = [self.start_pose[0] + float(self.task_spec[self.goal_id]["x"]),
                            self.start_pose[1] + float(self.task_spec[self.goal_id]["y"]),
                            self.start_pose[2] + float(self.task_spec[self.goal_id]["z"])]
        self.all_traj = [self.start_pose]
        self.all_waypoint = [self.start_pose]
        self.total_score = 0
        self.total_success = 0
        self.viz_task1()

        self.start_time = time.time()
        self.pub_timer = rospy.Timer(rospy.Duration(1.0/10), self.handle_goal_pub, self.goal_id)
        self.exec_timer = rospy.Timer(rospy.Duration(self.time_limit), self.handle_timeout, oneshot=True)  
        
        self.state_lock.release()
        return TriggerResponse(True, "Start Task%d"%(self.task))
    
    def handle_record_waypoint(self, _):
        if self.goal_id < 0:
            return (False, "The TaskA%d is not started yet."%(self.task))
        
        self.state_lock.acquire()
        t = time.time() - self.start_time
        # Task A Record
        self.exec_timer.shutdown()
        if self.current_pom is None:
            dis = 10
        else:
            cur = self.current_pom
            cur_pose = [cur.position.x, cur.position.y, cur.position.z]
            dis = math.sqrt((cur_pose[0] - self.goal[0]) * (cur_pose[0] - self.goal[0]) + 
                            (cur_pose[1] - self.goal[1]) * (cur_pose[1] - self.goal[1]) +
                            (cur_pose[2] - self.goal[2]) * (cur_pose[2] - self.goal[2])) * 100
            
        # visualization
        self.all_traj += [cur_pose]
        self.all_waypoint += [self.goal]
        self.viz_task1()
        if self.task == 1:
            result = dis < TASKA1_DISTANCE_TOLERANCE
            if result:
                self.total_success += 1
            else:
                self.total_score +=  TASKA1_TIME_PENALTY
                
            msg = "Current Waypoint ID [%d]\nResult [%s]\nProgress [%d/%d]\nRemaining Waypoints [%d]\nAccumulated Execution Time [%f] s"%(
                    self.goal_id,"Success" if result==1 else "Failure",  self.total_success, self.goal_id + 1, self.num_waypoints - self.goal_id - 1, t)
            print(msg)
            print("---------------------")
        else:
            self.total_score += dis 
            msg = "Current Waypoint ID [%d]\nDistance Error [%f] cm\nAccumulated Error [%f] cm\nRemaining Waypoints [%d]\nAccumulated Execution Time [%f] s"%(
                    self.goal_id, dis, self.total_score, self.num_waypoints - self.goal_id - 1, t)
            print(msg)
            print("---------------------")

        self.msg_pub.publish(msg)

        self.goal_id += 1
        if self.goal_id < self.num_waypoints:
            self.goal = [self.start_pose[0] + float(self.task_spec[self.goal_id]["x"]),
                            self.start_pose[1] + float(self.task_spec[self.goal_id]["y"]),
                            self.start_pose[2] + float(self.task_spec[self.goal_id]["z"])]
            self.task_visualize.viz_goal(self.goal)
            self.exec_timer = rospy.Timer(rospy.Duration(self.time_limit), self.handle_timeout, oneshot=True)
            self.state_lock.release()
            return TriggerResponse(True, msg)
        else:
            if self.task == 1:
                penalty_time = (self.num_waypoints - self.total_success) * TASKA1_TIME_PENALTY
                msg = "Execution Time [%f]\nPenalty time [%f]\nTotal Score [%f]"%(t, penalty_time, penalty_time + t)
                print(msg)
                print("---------------------")
                self.msg_pub.publish(msg)
            self.handle_stop_task(_)
            self.pub_timer.shutdown()
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
        p = Point()
        p.x = self.goal[0]
        p.y = self.goal[1]
        p.z = self.goal[2]
        self.goal_pub.publish(p)

if __name__ == "__main__":
    rospy.init_node("taskA evaluation", anonymous=True)
    rospy.sleep(0.5)
    node = TaskEvalA()
    rospy.spin()