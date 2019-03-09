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
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# Position.
from tf import TransformListener
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from tf.transformations import euler_from_quaternion


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


def braitenberg(front, front_left, front_right, left, right):
  # I'm making a mini ANN as shown during lecture 3, slide 13, which will give me
  # wheels' speeds, from which we calculate u and omega using some forward kinematics
  I = np.array([left, front_left, front, front_right, right])
  I = np.clip(I, 0, 5) # I'm clipping it as sometimes (when there is nothing ahead)
                       # sensors are inf, and inf-inf is NaN, which causes me a
                       # numerical error;
                       # 5 will also indicate infinity, (instead of 3.5)
  I = I - np.array([1.1, 1.1, 1.1, 1.1, 1.1]) # this is what each sensor would consider as close
  print(I)
  # When 'I' values go below 0, the wheels coefficients ('w' below), +ve will be multiplied by -ve
  # which will contribute to that particular wheel starting to reverse
  w = np.array([[0.05, 0.3, 0.6, 1.0, 0.6], # something on the right -> left wheel reverse -> left turn made
                [0.6, 1.0, 0.6, 0.3, 0.05]])# vice versa
  phi = (np.matmul(w, I)) * 1/8 # phi <-> wheels' speeds
  print(phi)
  phi = np.tanh(phi) * 2.
  print(phi,'\n------\n')

  u = (phi[1] + phi[0]) / 2. # assume that the radius of the wheel is unit.
  omega = phi[1] - phi[0]  # and that the axle length is also unit
  # in this case if we had some non-unit wheel radius or axle length
  # we would just re-adjust the weights.

  return u, omega

def vector_length(v):
  return np.sqrt(v[0] ** 2 + v[1] ** 2)


def dot_prod(v1, v2):
  return sum((a * b) for a, b in zip(v1, v2))


def cross_prod(v1, v2):
  return v1[0] * v2[1] - v2[0] * v1[1]


def get_alpha(v1, v2):
  alpha = np.arccos(min(max(dot_prod(v1, v2) / (vector_length(v1) * vector_length(v2)), -1.), 1.))
  if cross_prod(v1, v2) < 0:
    alpha = 2 * np.math.pi - alpha

  return alpha

# def get_bearing(v1, v2):
#
#   dot = np.dot(v1, v2)
#   det = v1[X] * v2[Y] - v2[X] * v1[Y]
#
#   angle = np.arctan2(det, dot)
#   return angle

def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # MISSING: Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.

  u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
  w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))

  return u, w


class SimpleLaser(object):
  def __init__(self, robot_name):
    rospy.Subscriber('/' + robot_name + '/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None

  def callback(self, msg):
    # Helper for angles.
    def _within(x, a, b):
      pi2 = np.pi * 2.
      x %= pi2
      a %= pi2
      b %= pi2
      if a < b:
        return a <= x and x <= b
      return a <= x or x <= b;

    # Compute indices the first time.
    if self._indices is None:
      self._indices = [[] for _ in range(len(self._angles))]
      for i, d in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        for j, center_angle in enumerate(self._angles):
          if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
            self._indices[j].append(i)

    ranges = np.array(msg.ranges)
    for i, idx in enumerate(self._indices):
      # We do not take the minimum range of the cone but the 10-th percentile for robustness.
      self._measurements[i] = np.percentile(ranges[idx], 10)

  @property
  def ready(self):
    return not np.isnan(self._measurements[0])

  @property
  def measurements(self):
    return self._measurements


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
    r1_pose = self.get_pose(FOLLOWERS[0])
    r2_pose = self.get_pose(FOLLOWERS[1])

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


# Not sure if we need this guy
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


ZS_DESIRED = {FOLLOWERS[0]: [0.04, np.math.pi],
              FOLLOWERS[1]: [0.06, np.math.pi]}


def set_distance_and_bearing(robot_name, dist, bearing):
  """ Bearing is always within [0; 2pi], not [-pi;pi] """
  global ZS_DESIRED
  zs_desired[robot_name] = [dist, bearing]


def run():
  global ZS_DESIRED
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

  # separation distance
  d = 0.4
  k = np.array([1, 0.9])

  cnt = 0

  while not rospy.is_shutdown():
    if not leader_laser.ready:
      rate_limiter.sleep()
      continue

    print('measurments', leader_laser.measurements)
    # u, w = rule_based(*leader_laser.measurements)
    u, w = braitenberg(*leader_laser.measurements)
    print('vels', u, w)
    vel_msg_l = Twist()
    vel_msg_l.linear.x = max(min(u, 0.2), -0.2)
    vel_msg_l.angular.z = max(min(w, 0.1), -0.1)
    l_publisher.publish(vel_msg_l)
    leader_pose = slam.get_pose(LEADER)

    # chance of this happening if map-merge has not converged yet (or maybe some other reason)
    if leader_pose is None:
      rate_limiter.sleep()
      continue

    max_speed = np.abs(vel_msg_l.linear.x * 4.5)
    max_angular = np.abs(vel_msg_l.angular.z * 4.5)

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

      G = np.array([[np.cos(gamma), d * np.sin(gamma)],
                    [-np.sin(gamma) / z[0], d * np.cos(gamma) / z[0]]])
      F = np.array([[-np.cos(z[1]), 0],
                    [np.sin(z[1]) / z[0], -1]])

      p = k * (zs_desired[follower] - z)

      speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
      speed_follower = np.matmul(np.linalg.inv(G), (p - np.matmul(F, speed_robot)))
      # print('\t', speed_follower)
      speed_follower[0] = max(min(speed_follower[0], max_speed), -max_speed)
      speed_follower[1] = max(min(speed_follower[1], max_angular), -max_angular)

      vel_msg = Twist()
      if cnt < 500:
        vel_msg = stop_msg
        cnt += 500
      else:
        vel_msg.linear.x = speed_follower[0]
        vel_msg.angular.z = speed_follower[1]

      vel_msg.linear.x = speed_follower[0]
      vel_msg.angular.z = speed_follower[1]

      print('\t, follower ' + str(i) + ' speed:', vel_msg.linear.x, vel_msg.angular.z)
      print('\t distance to leader: ' + str(z[0]))
      print()
      print()

      f_publishers[i].publish(vel_msg)

    rate_limiter.sleep()


if __name__ == '__main__':
  run()
