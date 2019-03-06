#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys
from scipy.optimize import least_squares

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

from sim import vector_length, get_alpha
# that's just for prototyping
import obstacle_avoidance
import pprint

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
INTER_WHEEL_RADIUS = 0.6

ROSPY_RATE = 50

X = 0
Y = 1
YAW = 2
ANGLE = 2

ROBOT_COUNT = 3
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


class SimpleLaser(object):
  def __init__(self, robot_name, braitenberg = False):
    rospy.Subscriber('/' + robot_name + '/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None
    self._msg = None
    self._counter = 0
    self._increment = None
    self._ranges = None
    self._bb = braitenberg

  def callback(self, msg):
    # Helper for angles.

    if self._increment is None:
      self._increment = msg.angle_increment

    self._ranges = list(msg.ranges)

    if not self._bb:
      return

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

  def circle_dist(self, centre, radius, ang):

    vector = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)

    a = np.dot(vector, vector)
    b = 2 * np.dot(vector, -centre)
    c = np.dot(centre, centre) - radius*radius

    disc = b * b - 4 * a * c

    if disc < 0:
      return float('inf')

    sqrt_disc = np.sqrt(disc)
    k1 = (-b + sqrt_disc) / (2*a)
    k2 = (-b - sqrt_disc) / (2*a)

    k1_d = np.linalg.norm(k1 * vector)
    k2_d = np.linalg.norm(k2 * vector)

    return min(k1_d, k2_d)

  def boundary_circ_angle(self, center_distance, radius):
    return np.arcsin(radius / center_distance)

  def boundary_rect_angle(self, center_distance, width):
    return np.arctan((width/2) / center_distance)

  def find_robots_2(self):

    if self._ranges is None:
      self._counter += 1
      return

    increment = self._increment
    ranges = self._ranges

    def delete_outliers(s):

      outlier_thresh = 0.1
      new_s = []

      for i, (d, a) in enumerate(s):
        d_next, a_next = s[np.mod(i+1, len(s))]
        d_prev, a_prev = s[np.mod(i+1, len(s))]

        if  (np.abs(d - d_next) > outlier_thresh and np.abs(d - d_prev) > outlier_thresh):
          continue
        new_s.append((d, a))

      return new_s

    def fit_circle(points, guess):

      x0 = guess

      print("FIT CIRCLE GUESS", x0)

      def f(x):
        res = 0
        for i in range(0, len(points)):
          d_i = points[i][0]
          a_i = points[i][1]
          a = np.square(d_i * np.cos(a_i) - x[0])
          b = np.square(d_i * np.sin(a_i) - x[1])
          res += np.square(np.sqrt(a + b) - ROBOT_RADIUS)
        return res

      res = least_squares(f, x0)

      print("FIT CIRLCE RESULT", res)
      print()
      print()

      return res


    max_d = 1.5
    t_sec = 0.05
    t_max_sel = np.ceil(2 * self.boundary_circ_angle(3.5 + ROBOT_RADIUS, ROBOT_RADIUS) / increment)
    t_min_sel = np.ceil(2 * self.boundary_circ_angle(0.1 + ROBOT_RADIUS, ROBOT_RADIUS) / increment)

    ranges_dict = {index: r for (index, r) in enumerate(ranges) if r != float('inf')}
    s = [(dist, increment * index) for (index, dist) in enumerate(ranges)]
    s = delete_outliers(s)

    fs = []

    # distance truncation
    for i, (d, a) in enumerate(s):
      if d < max_d:
        fs.append((d, a))

    s = fs

    if len(s) < 1:
      return []

    n_cl = 0
    cl = [[s[n_cl]]]

    # point cloud clustering
    for k in range(2, len(s)):
      s_k = s[k]
      s_k_m = s[k-1]
      s_k_mm = s[k-2]

      if np.abs(s_k_m[0] - s_k[0]) > t_sec:
        n_cl += 1
        cl.append([s_k])
      else:
        cl[n_cl].append(s_k)

    f_cl = []

    # cluster selection
    for k in range(0, n_cl + 1):
      cl_k = cl[k]

      center_d = min(cl_k, key=lambda x: x[0])[0]
      min_a = min(cl_k, key=lambda x: x[1])[1]
      max_a = max(cl_k, key=lambda x: x[1])[1]

      a_span = max_a - min_a + increment
      e_span = 2 * self.boundary_circ_angle(center_d + ROBOT_RADIUS, ROBOT_RADIUS)

      print("CENTER D", center_d)
      print("A SPAN", a_span)
      print("E SPAN", e_span)
      print()
      print()

      if np.abs(a_span - e_span) > 3 * increment:
        n_cl -= 1
        continue

      f_cl.append(cl_k)

    print()
    print()
    print("NUMBER OF CLUSTERS", len(f_cl))

    robots = []

    for i, cl in enumerate(f_cl):
      center_t = min(cl, key=lambda x: x[0])
      len_to_center = center_t[0] + ROBOT_RADIUS
      print("CLUSTER", i)
      for dist, ang in cl:
        print("ANG", ang, "DIST", dist)
      # print("CLUSTER", i, "CIRCLE CENTER", coord)
      res = fit_circle(cl, np.array([len_to_center * np.cos(center_t[1]), len_to_center * np.sin(center_t[1])]))
      robots.append(res.x)
      print()

    print()

    return robots

  def find_robots_3(self):

    if self._ranges is None:
      self._counter += 1
      return

    increment = self._increment
    ranges = self._ranges

    def delete_outliers(s):

      outlier_thresh = 0.1
      new_s = []

      for i, (d, a) in enumerate(s):
        d_next, a_next = s[np.mod(i+1, len(s))]
        d_prev, a_prev = s[np.mod(i+1, len(s))]

        if  (np.abs(d - d_next) > outlier_thresh and np.abs(d - d_prev) > outlier_thresh):
          continue
        new_s.append((d, a))

      return new_s

    def fit_circle(points, guess):

      x0 = guess

      def f(x):
        res = 0
        for i in range(0, len(points)):
          d_i = points[i][0]
          a_i = points[i][1]
          a = np.square(d_i * np.cos(a_i) - x[0])
          b = np.square(d_i * np.sin(a_i) - x[1])
          res += np.square(np.sqrt(a + b) - ROBOT_RADIUS)
        return res

      res = least_squares(f, x0)

      return res

    max_d = 2.5
    min_points = 3
    t_sec = 0.05
    s = [(dist, increment * index) for (index, dist) in enumerate(ranges)]
    s = delete_outliers(s)

    fs = []

    # distance truncation
    for i, (d, a) in enumerate(s):
      if d < max_d:
        fs.append((d, a))

    s = fs

    if len(s) < 1:
      return []

    n_cl = 0
    cl = [[s[n_cl]]]

    # point cloud clustering
    for k in range(2, len(s)):
      s_k = s[k]
      s_k_m = s[k-1]
      s_k_mm = s[k-2]

      if np.abs(s_k_m[0] - s_k[0]) > t_sec:
        n_cl += 1
        cl.append([s_k])
      else:
        cl[n_cl].append(s_k)

    f_cl = []

    # cluster selection
    for k in range(0, n_cl + 1):
      cl_k = cl[k]

      if len(cl_k) < min_points:
        n_cl -= 1
        continue

      center_d = min(cl_k, key=lambda x: x[0])[0]
      min_a = min(cl_k, key=lambda x: x[1])[1]
      max_a = max(cl_k, key=lambda x: x[1])[1]

      a_span = max_a - min_a + increment
      e_span = 2 * self.boundary_circ_angle(center_d + ROBOT_RADIUS, ROBOT_RADIUS)

      # print("CENTER D", center_d)
      # print("A SPAN", a_span)
      # print("E SPAN", e_span)

      if np.abs(a_span - e_span) > 3 * increment:
        n_cl -= 1
        continue

      # size of cluster is close to robot. Now check that points are circular using Internal Angle Variance (IAV)
      cart_cl_k = [np.array([dist * np.cos(ang), dist * np.sin(ang)]) for (dist, ang) in cl_k]
      angles = np.zeros(len(cart_cl_k) - 2)
      extrem_1 = cart_cl_k[0]
      extrem_2 = cart_cl_k[len(cart_cl_k) - 2]

      for i in range(1, len(cart_cl_k) - 1):
        point = cart_cl_k[i]
        pe1 = extrem_1 - point
        pe2 = extrem_2 - point
        angle = np.arccos(np.dot(pe1, pe2) / np.linalg.norm(pe1) * np.linalg.norm(pe2))
        angles[i-1] = angle

      mean = angles.mean()
      std = angles.std()

      # print("ANGLES", angles)
      # print("ANGLES MEAN", mean)
      # print("ANGLES STD", std)

      if mean < 1.4 or mean > 2.3 or std > 0.15:
        continue

      f_cl.append(cl_k)

    # print()
    # print()
    # print("************** NUMBER OF CLUSTERS", len(f_cl))

    robots = []

    for i, cl in enumerate(f_cl):
      center_t = min(cl, key=lambda x: x[0])
      len_to_center = center_t[0] + ROBOT_RADIUS
      # print("CLUSTER", i)
      # for dist, ang in cl:
      #   print("ANG", ang, "DIST", dist)
      # print("CLUSTER", i, "CIRCLE CENTER", coord)
      res = fit_circle(cl, np.array([len_to_center * np.cos(center_t[1]), len_to_center * np.sin(center_t[1])]))
      coords = res.x

      # convert to polar
      r = np.sqrt(np.square(coords[0]) + np.square(coords[1]))
      theta = np.arctan2(coords[1], coords[0])

      robots.append((r, theta))
      # print()

    # print()

    return robots

  def find_robots(self):

    if self._ranges is None:
      self._counter += 1
      return

    increment = self._increment
    ranges = self._ranges
    ranges_dict = {index: r for (index, r) in enumerate(ranges) if r != float('inf')}

    robots = []
    min_elements = 3

    loop = 1
    while(len(robots) < ROBOT_COUNT - 1 and len(ranges_dict) > min_elements):

      # print("LOOP", loop)
      # print("DICT LENGTH", len(ranges_dict))

      index_min = min(ranges_dict, key=ranges_dict.get)
      min_val = ranges[index_min]
      angle = increment * index_min

      # assume this is the center of the circle
      center_circ_dist = min_val + ROBOT_RADIUS
      max_circ_angle = self.boundary_circ_angle(center_circ_dist, ROBOT_RADIUS)
      mca_rel_index = int(np.floor(np.abs(max_circ_angle) / increment))

      explore_right_angs = True
      explore_left_angs = True
      r_index = 0
      l_index = 0

      # for i in range(index_min - 20, index_min + 20, 1):
      #   print("Index", i, "Val", ranges[np.mod(i, len(ranges))])
      # print()

      def ranges_mod(x):
        return ranges[np.mod(x, len(ranges))]

      # the two while statements below look to the left and right on index_min to found the boundary of the object

      while explore_right_angs:
        if r_index < 90 and np.abs(ranges_mod(index_min + r_index + 1) - ranges_mod(index_min + r_index)) < 0.05:
          r_index += 1
        else:
          explore_right_angs = False

      while explore_left_angs:
        if l_index < 90 and np.abs(ranges_mod(index_min - l_index - 1) - ranges_mod(index_min - l_index)) < 0.05:
          l_index += 1
        else:
          explore_left_angs = False

      block_index_span = r_index + l_index

      if 2 * mca_rel_index > min_elements and np.abs(block_index_span - 2 * mca_rel_index) < 4:

        robot_center_index = np.mod(int(index_min - l_index + block_index_span / 2), len(ranges))
        robot_inline_dist = ranges[robot_center_index]
        robot_center_dist = robot_inline_dist + ROBOT_RADIUS
        robot_center = np.array([robot_center_dist * np.cos(robot_center_index * self._increment),
                                 robot_center_dist * np.sin(robot_center_index * self._increment),
                                 angle])
        robots.append(robot_center)
        print("PROBABLY A ROBOT WITH RELATIVE COORDS", robot_center)

      # print("INDEX_MIN: ", index_min)

      for i in range(index_min - l_index, index_min + r_index + 1):
        if i < 0:
          i = len(ranges) + i
        # print("DELETING KEY", i)
        i = np.mod(i, len(ranges))
        if i in ranges_dict:
          del ranges_dict[i]

      loop += 1

    return robots


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


# zs_desired = {FOLLOWERS[0]: [0.5, np.math.pi],
#               FOLLOWERS[1]: [1.0, np.math.pi]}

zs_desired = {FOLLOWERS[0]: np.array([0.35, 3.*np.math.pi/4.]),
              FOLLOWERS[1]: np.array([0.7, 5.*np.math.pi/4.])}
# right triangle, two sides 0.4
#                  l12,  psi12          , l13,   l23
zs_both_desired = [0.4, 5.*np.math.pi/4., 0.4, np.sqrt(0.32)]
#              psi13,            psi23
extra_psis = [3.*np.math.pi/4., np.math.pi/2.]


def set_distance_and_bearing(robot_name, dist, bearing):
  """ Bearing is always within [0; 2pi], not [-pi;pi] """
  global zs_desired
  zs_desired[robot_name] = [dist, bearing]

def get_relative_position(absolute_pose, absolute_position):
  relative_position = absolute_position.copy()
  diff_vec = absolute_position - absolute_pose[:2]

  rx_x = (diff_vec[X]) * np.cos(absolute_pose[YAW])
  rx_y = (diff_vec[Y]) * np.sin(absolute_pose[YAW])
  ry_x = (diff_vec[X]) * np.sin(absolute_pose[YAW])
  ry_y = (diff_vec[Y]) * np.cos(absolute_pose[YAW])

  relative_position[X] = rx_x + rx_y
  relative_position[Y] = -ry_x + ry_y

  return relative_position

speed_coeff = 1.


def run():
  global zs_desired
  global speed_coeff
  rospy.init_node('robot_controller')
  rate_limiter = rospy.Rate(ROSPY_RATE)

  l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
  f_publishers = [None] * len(FOLLOWERS)
  for i, follower in enumerate(FOLLOWERS):
    f_publishers[i] = rospy.Publisher('/' + follower + '/cmd_vel', Twist, queue_size=5)

  leader_laser = SimpleLaser(LEADER, True)
  follower_lasers = [SimpleLaser(FOLLOWER_1),
                     SimpleLaser(FOLLOWER_2)]

  stop_msg = Twist()
  stop_msg.linear.x = 0.
  stop_msg.angular.z = 0.

  previous_time = rospy.Time.now().to_sec()

  # Make sure the robot is stopped.
  i = 0
  while i < 10 and not rospy.is_shutdown():
    l_publisher.publish(stop_msg)
    for f_publisher in f_publishers:
      f_publisher.publish(stop_msg)

    rate_limiter.sleep()
    i += 1

  max_speed = 0.1
  max_angular = 0.1
  k = np.array([0.45, 0.24])
  d = 0.05
  cnt = 0

  while not rospy.is_shutdown():
    if not leader_laser.ready:
      rate_limiter.sleep()
      continue

    # print('measurments', leader_laser.measurements)
    # u, w = rule_based(*leader_laser.measurements)
    u, w = obstacle_avoidance.braitenberg(*leader_laser.measurements)
    u *= speed_coeff * 0.25
    w *= speed_coeff * 0.25
    print('vels', u, w)
    vel_msg_l = Twist()
    vel_msg_l.linear.x = max(min(u, max_speed), -max_speed)
    vel_msg_l.angular.z = max(min(w, max_speed), -max_speed)
    l_publisher.publish(vel_msg_l)

    lrs = leader_laser.find_robots_3()
    f1rs = follower_lasers[0].find_robots_3()
    f2rs = follower_lasers[1].find_robots_3()

    print()
    print("ROBOTS FROM LEADER PERSPECTIVE:", lrs)
    print("ROBOTS FROM FOLLOWER1 PERSPECTIVE:", f1rs)
    print("ROBOTS FROM FOLLOWER2 PERSPECTIVE:", f2rs)

    print()

    # if the followers are not picked up on radar
    if len(lrs) < 2:
      speed_coeff = np.abs(speed_coeff) * 0.95
      stop_msg = Twist()
      stop_msg.linear.x = 0.
      stop_msg.angular.z = 0.
      f_publishers[0].publish(stop_msg)
      f_publishers[1].publish(stop_msg)
      rate_limiter.sleep()
      continue

    speed_coeff = 1.

    f1_set = []
    f2_set = []

    f1_pred = (lrs[0], lrs[0])
    f2_pred = (lrs[1], lrs[1])

    if len(f1rs) >= 2 and len(f2rs) >= 2:

      for j,lr in enumerate(lrs):

        lr_dist = lr[0]

        for i, fr in enumerate(f1rs):
          fr_dist = fr[0]
          diff = np.abs(lr_dist - fr_dist)
          f1_set.append((j, diff, fr))

        for i, fr in enumerate(f2rs):
          fr_dist = fr[0]
          diff = np.abs(lr_dist - fr_dist)
          f2_set.append((j, diff, fr))

      f1_set = sorted(f1_set, key=lambda x: x[1])
      f2_set = sorted(f2_set, key=lambda x: x[1])

      f1 = f1_set[0]
      f2 = f2_set[0]

      if f1[0] == f2[0]:
        if f1[1] > f2[1]:
          f1 = f1_set[1]
        else:
          f2 = f2_set[1]

      print("LRS", lrs)
      print("f1", f1)
      print("f2", f2)
      print()

      f1_pred = (lrs[f1[0]], f1[2])
      f2_pred = (lrs[f2[0]], f2[2])

    # now have f1_pred and f2_pred predictions

    z = np.array([0., 0.])
    # z[0] = vector_length(corrected_leader_pose[:-1] - f1_pose[:-1])
    # z[1] = get_alpha(np.array([np.cos(corrected_leader_pose[YAW]), np.sin(corrected_leader_pose[YAW])]),
    #                  f1_pose[:-1] - corrected_leader_pose[:-1])
    # z[2] = vector_length(corrected_leader_pose[:-1] - f2_pose[:-1])
    # z[3] = vector_length(f1_pose[:-1] - f2_pose[:-1])
    # z[0] = f1_pred[0]
    # z[1] = f1_pred[1]
    # z[2] = f2_pred[0]

    fps = [f1_pred, f2_pred]

    for i, follower in enumerate(FOLLOWERS):

      f_pred = fps[i]

      print("FOLLOWER ", i)
      print()

      z = np.array([0., 0.])
      z[0] = f_pred[0][0]
      z[1] = f_pred[0][1]

      if z[1] < 0.:
        z[1] += 2 * np.math.pi

      print("Z[0]", z[0])


      beta = np.pi + f_pred[1][1] - f_pred[0][1]
      gamma = beta + z[1]

      G = np.array([[np.cos(gamma), d * np.sin(gamma)],
                    [-np.sin(gamma) / z[0], d * np.cos(gamma) / z[0]]])
      F = np.array([[-np.cos(z[1]), 0],
                    [np.sin(z[1]) / z[0], -1]])

      # print("ZS", zs_desired[follower])
      # print("ZS SHAPE", zs_desired[follower].shape)
      # print("Z", z)
      # print("Z shape", z.shape)

      print('\t z<->zs', z, ' <-> ', zs_desired[follower])
      p = k * (zs_desired[follower] - z)
      print('\t p k * (zs - z)', p)

      speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
      speed_follower = np.matmul(np.linalg.inv(G), (p - np.matmul(F, speed_robot)))
      print('\t speed follower', speed_follower)
      speed_follower[0] = max(min(speed_follower[0], max_speed * 2), -max_speed * 2)
      speed_follower[1] = max(min(speed_follower[1], max_angular * 5), -max_angular * 5)

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

      f_publishers[i].publish(vel_msg)

    rate_limiter.sleep()

    # #         g12
    # gammas = [corrected_leader_pose[YAW] - f1_pose[YAW] + z[1]]
    # #             g13
    # gammas.append(corrected_leader_pose[YAW] - f2_pose[YAW] + extra_psis[0])
    # #             g23
    # gammas.append(f1_pose[YAW] - f2_pose[YAW] + extra_psis[1])
    # # beta =
    # # gamma = beta + z[1]
    #
    # G = np.array([[np.cos(gammas[0]), d * np.sin(gammas[0]), 0, 0],
    #               [-np.sin(gammas[0]) / z[0], d * np.cos(gammas[0]) / z[0], 0, 0],
    #               [0, 0, np.cos(gammas[1]), d * np.sin(gammas[1])],
    #               [0, 0, np.cos(gammas[2]), d * np.sin(gammas[2])]])
    # F = np.array([[-np.cos(z[1]), 0],
    #               [np.sin(z[1]) / z[0], -1],
    #               [-np.cos(extra_psis[0]), 0],
    #               [0, 0]])
    #
    # print('\t zs', z, ' <-> ', zs_both_desired)
    # p = k * (zs_both_desired - z)
    #
    # speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
    # speed_follower = np.matmul(np.linalg.inv(G), (p - np.matmul(F, speed_robot)))
    # print('\t', speed_follower)
    # speed_follower[0] = max(0.5 * min(speed_follower[0], max_speed), -max_speed)
    # speed_follower[1] = max(0.4 * min(speed_follower[1], max_angular), -max_angular)
    # speed_follower[2] = max(0.5 * min(speed_follower[2], max_speed), -max_speed)
    # speed_follower[3] = max(0.4 * min(speed_follower[3], max_angular), -max_angular)
    #
    # vel_msg = Twist()
    #
    # vel_msg.linear.x = speed_follower[0]
    # vel_msg.angular.z = speed_follower[1]
    #
    # print('\t, follower ' + str(0) + ' speed:', vel_msg.linear.x, vel_msg.angular.z)
    # print()
    # print()
    #
    # f_publishers[0].publish(vel_msg)
    #
    # vel_msg.linear.x = speed_follower[2]
    # vel_msg.angular.z = speed_follower[3]
    # print('\t, follower ' + str(1) + ' speed:', vel_msg.linear.x, vel_msg.angular.z)
    # print()
    # print()
    #
    # f_publishers[1].publish(vel_msg)
    #
    # rate_limiter.sleep()


if __name__ == '__main__':
  run()
