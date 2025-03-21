import rospy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import numpy as np

color = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
def make_header(frame_id, stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header

def delete_marker(ns="task1", id=0, frame_id = "usb_cam"):
        m = Marker()
        m.header = make_header(frame_id)
        m.ns = ns
        m.id = id
        m.action = m.DELETE
        return m
    
class TaskViz():
    def __init__(self, ):
        self.task1_goal_pub = rospy.Publisher("/rgmc_eval/task1/goalpoint", Marker, queue_size=1)
        self.task1_traj_pub = rospy.Publisher("/rgmc_eval/task1/trag", Marker, queue_size=1)
        self.task1_waypoint_pub = rospy.Publisher("/rgmc_eval/task1/waypoint", Marker, queue_size=1)

        self.task2_normal_pub = rospy.Publisher("/rgmc_eval/task2/normal", Marker, queue_size=1)
        self.task2_cube_pub = rospy.Publisher("/rgmc_eval/task2/cube", MarkerArray, queue_size=1)
        rospy.sleep(0.2)

    def viz_goal(self, poses, color=[1, 1, 0, 1], ns="goal", id=0, scale=0.01, frame_id = "usb_cam"):
        m = Marker()
        m.header = make_header(frame_id)
        m.ns = ns
        m.id = id
        m.type = Marker.SPHERE
        m.action = m.ADD
        m.pose.position.x = poses[0]
        m.pose.position.y = poses[1]
        m.pose.position.z = poses[2]
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        self.task1_goal_pub.publish(m)
        rospy.sleep(0.05)

    def viz_traj(self, poses, color=[1, 0, 0, 1], ns="task1", id=0, scale=0.01, frame_id = "usb_cam"):
        m = Marker()
        m.header = make_header(frame_id)
        m.ns = ns
        m.id = id
        m.type = Marker.SPHERE_LIST
        m.action = m.ADD
        m.scale.x = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color

        for pose in poses:
            p = Point()
            p.x = pose[0]
            p.y = pose[1]
            p.z = pose[2]
            m.points.append(p) 
        self.task1_traj_pub.publish(m)
        rospy.sleep(0.05)

    def viz_waypoint(self, poses, color=[0, 1, 0, 1], ns="task1", id=0, scale=0.01, frame_id = "usb_cam"):
        m = Marker()
        m.header = make_header(frame_id)
        m.ns = ns
        m.id = id
        m.type = Marker.SPHERE_LIST
        m.action = m.ADD
        m.scale.x = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color

        for pose in poses:
            p = Point()
            p.x = pose[0]
            p.y = pose[1]
            p.z = pose[2]
            m.points.append(p) 
        self.task1_waypoint_pub.publish(m)
        rospy.sleep(0.05)

    def normal_marker(self, poses, color=[0, 1, 0, 1], ns="task1", id=0, frame_id = "usb_cam"):
        m = Marker()
        m.header = make_header(frame_id)
        m.ns = ns
        m.id = id
        m.type = Marker.ARROW
        m.action = m.ADD
        m.scale.x = 0.002
        m.scale.y = 0.003
        m.scale.z = 0
        m.pose.orientation.w = 1
        m.color.r, m.color.g, m.color.b, m.color.a = color

        for pose in poses:
            p = Point()
            p.x = pose[0]
            p.y = pose[1]
            p.z = pose[2]
            m.points.append(p) 
        return m

    def viz_cube(self, poses, ns="task2", frame_id = "usb_cam"):
        m = MarkerArray()
        for i, p in enumerate(poses):
            if len(p) > 0:
                p = np.mean(p, axis=0)
                m.markers.append(self.normal_marker([p[:3], p[:3] + 0.05 * p[3:]], color=color[i], ns=ns, id=i, frame_id=frame_id))
            else:
                m.markers.append(delete_marker(ns=ns, id=i, frame_id=frame_id))
        self.task2_cube_pub.publish(m)
        rospy.sleep(0.05)

