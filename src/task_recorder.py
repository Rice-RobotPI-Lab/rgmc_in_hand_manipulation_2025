#!/usr/bin/env python
import rospy
import rospkg

import os
import time
import yaml
import math
import numpy as np
import uuid
from functools import partial
from threading import Lock


from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose, PoseStamped
from apriltag_ros.msg import AprilTagDetectionArray
from sensor_msgs.msg import Image

from std_srvs.srv import Trigger, TriggerResponse

import cv2 as cv
import cv_bridge

class TaskRecorder(object):
    def __init__(self):
        self.recording = False
        self.writer = None

        rospack = rospkg.RosPack()
        pack_path = rospack.get_path("rgmc_in_hand_manipulation_2025")
        self.record_folder = os.path.join(pack_path, "records")

        self.camera_topic = rospy.get_param("task_recorder/camera_topic")
        self.video_width = rospy.get_param("task_recorder/video_width")
        self.video_height = rospy.get_param("task_recorder/video_height")
        self.frame_rate = rospy.get_param("task_recorder/frame_rate")
        self.prefix = rospy.get_param("task_recorder/prefix")

        self.cvb = cv_bridge.CvBridge()
        self.start = rospy.Service('/rgmc_recorder/start', Trigger, self.handle_start_task)
        self.stop = rospy.Service('/rgmc_recorder/stop', Trigger, self.handle_stop_task)
        self.camera_subsriber = rospy.Subscriber(self.camera_topic, Image, self.callback_record, queue_size=1)
        self.msg_subscriber = rospy.Subscriber('/rgmc_eval/msg', String, self.callback_message, queue_size=1)
        self.buffer = []

    def callback_record(self, msg):
        if self.recording:
            try:
                cv_image = self.cvb.imgmsg_to_cv2(msg, "bgr8")
                self.buffer.append(cv_image)
            except cv_bridge.CvBridgeError as e:
                print(e)
    
    def callback_message(self, msg):
        with open(os.path.join(self.record_folder, self.prefix + str(self.current_id) + ".txt"), "a+") as f:
            f.write(msg.data)
            f.write("\n---------------------\n")

    def handle_start_task(self, _):
        self.current_id = uuid.uuid4()
        self.buffer = []
        self.recording = True
        return TriggerResponse(True, "Start Record" + str(self.current_id))
    
     

    def handle_stop_task(self, _):
        self.recording = False
        self.writer = cv.VideoWriter(os.path.join(self.record_folder, self.prefix + str(self.current_id) + ".avi"),
                                             cv.VideoWriter_fourcc(*'MJPG'),
                                             self.frame_rate, 
                                             (self.video_width, self.video_height))
        for frame in self.buffer:
            self.writer.write(frame)
        self.writer.release()
        return TriggerResponse(True, "Stop Record" + str(self.current_id))


if __name__ == "__main__":
    rospy.init_node("Task Recorder", anonymous=True)
    rospy.sleep(0.5)
    node = TaskRecorder()
    rospy.spin()