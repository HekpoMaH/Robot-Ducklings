#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
# Position.
from tf import TransformListener
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from tf.transformations import euler_from_quaternion

from obstacle_avoidance import SimpleLaser

from sim import vector_length, get_alpha
# that's just for prototyping
from obstacle_avoidance import rule_based

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.

X = 0
Y = 1
YAW = 2

LEADER = 'tb3_0'
FOLLOWERS = ['tb3_1', 'tb3_2']
FOLLOWER_1 = 'tb3_1'
FOLLOWER_2 = 'tb3_2'

def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # MISSING: Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.

  u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
  w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))

  return u, w



class SLAM(object):
  def __init__(self):
    rospy.Subscriber('/map', OccupancyGrid, self.callback)
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    
  def callback(self, msg):
    pass

  def update(self):
    # Get pose w.r.t. map.
    r0_pose = self.get_pose(LEADER)
    r1_pose = self.get_pose(FOLLOWER_1)
    r2_pose = self.get_pose(FOLLOWER_2)

    print('leader')
    print('\t X: ', r0_pose[X])
    print('\t Y: ', r0_pose[Y])
    print('\t YAW: ', r0_pose[YAW])

    print()

    print('follower 1')
    print('\t X: ', r1_pose[X])
    print('\t Y: ', r1_pose[Y])
    print('\t YAW: ', r1_pose[YAW])

    print()

    print('follower 2')
    print('\t X: ', r2_pose[X])
    print('\t Y: ', r2_pose[Y])
    print('\t YAW: ', r2_pose[YAW])

    print()
    print()
    print()

  def get_pose(self, robot):
    a = 'occupancy_grid'
    b = robot + '/base_link'
    if self._tf.frameExists(a) and self._tf.frameExists(b):
      try:
        t = rospy.Time(0)
        position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
        pose = np.zeros(3, dtype=np.float32)
        pose[X] = position[X]
        pose[Y] = position[Y]
        _, _, pose[YAW] = euler_from_quaternion(orientation)
        return pose

      except Exception as e:
        print(e)
    else:
      print('Unable to find:', self._tf.frameExists(a), self._tf.frameExists(b))


  @property
  def ready(self):
    return self._occupancy_grid is not None and not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def occupancy_grid(self):
    return self._occupancy_grid

#Not sure if we need this guy
class GoalPose(object):
  def __init__(self):
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback)
    self._position = np.array([np.nan, np.nan], dtype=np.float32)

  def callback(self, msg):
    # The pose from RViz is with respect to the "map".
    self._position[X] = msg.pose.position.x
    self._position[Y] = msg.pose.position.y
    print('Received new goal position:', self._position)

  @property
  def ready(self):
    return not np.isnan(self._position[0])

  @property
  def position(self):
    return self._position

zs_desired = {FOLLOWERS[0]: [0.04, np.math.pi],
              FOLLOWERS[1]: [0.06, np.math.pi]}
def set_distance_and_bearing(robot_name, dist, bearing):
    """ Bearing is always within [0; 2pi], not [-pi;pi] """
    global zs_desired
    zs_desired[robot_name] = [dist, bearing]

def run():
    global zs_desired
    rospy.init_node('robot_controller')

    l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
    f_publishers = [None] * len(FOLLOWERS)
    for i, follower in enumerate(FOLLOWERS):
        f_publishers[i] = rospy.Publisher('/' + follower + '/cmd_vel', Twist, queue_size=5)

    slam = SLAM()

    rate_limiter = rospy.Rate(100)

    stop_msg = Twist()
    stop_msg.linear.x = 0.
    stop_msg.angular.z = 0.

    leader_laser = SimpleLaser(LEADER)

    previous_time = rospy.Time.now().to_sec()

    # Make sure the robot is stopped.
    i = 0
    while i < 10 and not rospy.is_shutdown():
      l_publisher.publish(stop_msg)
      for f_publisher in f_publishers:
          f_publisher.publish(stop_msg)

      rate_limiter.sleep()
      i += 1
    


    d = 0.05
    k = np.array([1, 0.9])

    cnt = 0

    while not rospy.is_shutdown():
        if not leader_laser.ready:
            rate_limiter.sleep()
            continue
        
        print('measurments', leader_laser.measurements)
        u, w = rule_based(*leader_laser.measurements)
        print('vels', u, w)
        vel_msg_l = Twist()
        vel_msg_l.linear.x = u
        vel_msg_l.angular.z = w
        l_publisher.publish(vel_msg_l)
        leader_pose = slam.get_pose(LEADER)
        print('leader_pose', leader_pose)
        print('leader_speed', vel_msg_l.linear.x, vel_msg_l.angular.z)
        for i, follower in enumerate(FOLLOWERS):

            follower_pose = slam.get_pose(follower)
            print('\t i, follower_pose:', i, follower_pose)

            z = np.array([0., 0.])
            z[0] = vector_length(leader_pose[:-1] - follower_pose[:-1])
            z[1] = get_alpha(np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])]),
                             follower_pose[:-1] - leader_pose[:-1])

            beta = leader_pose[YAW] - follower_pose[YAW]
            gamma = beta + z[1]

            G=np.array([[np.cos(gamma), d*np.sin(gamma)],
                        [-np.sin(gamma)/z[0], d*np.cos(gamma)/z[0]]])
            F=np.array([[-np.cos(z[1]), 0],
                        [np.sin(z[1])/z[0], -1]])
        
            p = k * (zs_desired[follower]-z)

            speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
            speed_follower = np.matmul(np.linalg.inv(G), (p-np.matmul(F, speed_robot)))
            print('\t', speed_follower)
            speed_follower[0] = min(speed_follower[0], 0.5)
            
            vel_msg = Twist()
            if cnt < 500:
                vel_msg = stop_msg
                cnt += 500
            else:
                vel_msg.linear.x = speed_follower[0]
                vel_msg.angular.z = speed_follower[1]

            f_publishers[i].publish(vel_msg)
            


        rate_limiter.sleep()

if __name__ == '__main__':
    run()
