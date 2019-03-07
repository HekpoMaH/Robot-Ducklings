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

ROBOT_RADIUS = 0.105
INTER_WHEEL_RADIUS = 0.8
ROBOT_WIDTH = 0.16
LIDAR_RADIUS = 0.035

ROSPY_RATE = 50

LIDAR_ROBOTS = 0
LIDAR_OBSTACLES = 1
LIDAR_ALL = 2

X = 0
Y = 1
YAW = 2
ANGLE = 2

ROBOT_COUNT = 3
LEADER = 'tb3_0'
FOLLOWERS = ['tb3_1', 'tb3_2']
FOLLOWER_1 = 'tb3_1'
FOLLOWER_2 = 'tb3_2'


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


  def cluster_environment(self):

    result = [[]]*2

    if self._ranges is None:
      self._counter += 1
      return result

    increment = self._increment
    ranges = self._ranges
    robots = []
    obstacles = []
    all = []


    def delete_outliers(s):

      outlier_thresh = 0.25
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

    max_dist = 2.5
    min_points = 3
    t_sec = 0.05
    cluster_angle_mult = 1.1
    lidar_radius_fuzz = 0.05
    shape_mean_min = 1.4
    shape_mean_max = 1.6
    shape_std = 0.15

    s = [(dist, increment * index) for (index, dist) in enumerate(ranges)]
    s = delete_outliers(s)
    fs = []

    # distance truncation
    for i, (d, a) in enumerate(s):
      if d < max_dist:
        fs.append((d, a))

    s = fs

    if len(s) < 1:
      return result

    cl = []
    start = 0

    # start clustering on boundary of cluster
    for k in range(0, len(s)):
      s_k = s[k]
      s_k_m = s[k-1]
      ang_diff = s_k[1] - s_k_m[1] + 2*np.pi if s_k[1] - s_k_m[1] < 0 else s_k[1] - s_k_m[1]
      if np.abs(s_k_m[0] - s_k[0]) > t_sec or ang_diff > cluster_angle_mult * increment:
        start = k
        break

    # point cloud clustering
    for k in range(start, start + len(s)):
      s_k = s[np.mod(k, len(s))]
      s_k_m = s[np.mod(k-1, len(s))]
      ang_diff = s_k[1] - s_k_m[1] + 2 * np.pi if s_k[1] - s_k_m[1] < 0 else s_k[1] - s_k_m[1]
      if np.abs(s_k_m[0] - s_k[0]) > t_sec or ang_diff > cluster_angle_mult * increment:
        cl.append([s_k])
      else:
        cl[len(cl) - 1].append(s_k)

    f_cl = []

    # cluster selection
    for k in range(0, len(cl)):
      cl_k = cl[k]

      all.append(cl_k)

      if len(cl_k) < min_points:
        continue

      center_d = min(cl_k, key=lambda x: x[0])[0]
      min_a = cl_k[0][1]
      max_a = cl_k[len(cl_k)-1][1]
      diff_ang = max_a - min_a
      if diff_ang < 0:
        diff_ang = diff_ang + np.pi * 2

      a_span = diff_ang + increment * 2
      e_span_lidar = 2 * self.boundary_circ_angle(center_d + LIDAR_RADIUS, LIDAR_RADIUS)
      e_span_lidar_p = 2 * self.boundary_circ_angle(center_d + LIDAR_RADIUS + lidar_radius_fuzz,
                                                    LIDAR_RADIUS + lidar_radius_fuzz)

      # e_span = 2 * self.boundary_circ_angle(center_d + ROBOT_RADIUS, ROBOT_RADIUS)
      # e_rect_span = 2 * self.boundary_rect_angle(center_d, ROBOT_WIDTH)
      # e_rect_lidar = 2 * self.boundary_rect_angle(center_d, 2 * LIDAR_RADIUS)
      # e_rect_lidar_p = 2 * self.boundary_rect_angle(center_d, 2 * (LIDAR_RADIUS+lidar_radius_fuzz))

      # print("CENTER D", center_d)
      # print("A SPAN", a_span)
      # print("E SPAN", e_span)
      # print("E SPAN LIDAR", e_span_lidar)
      # print("E RECT SPAN", e_rect_span)
      # print("E SPAN LIDAR P", e_span_lidar_p)
      # print("E RECT LIDAR", e_rect_lidar)
      # print("E RECT LIDAR", e_rect_lidar_p)
      # print()

      # if np.abs(a_span - e_span_lidar) > 3 * increment:
      if a_span < e_span_lidar or a_span > e_span_lidar_p:
        # print("PURGING CLUSTER (1) AS OBSTACLE")
        # for c in cl_k:
        #   print("\t ", c)
        obstacles.append(cl_k)
        continue
      # else:
        # print("KEEPING CLUSTER (1)")
        # for c in cl_k:
        #   print("\t ", c)

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

      if mean < shape_mean_min or mean > shape_mean_max or std > shape_std:
        # print("PURGING CLUSTER (2) AS OBSTACLE")
        # for c in cl_k:
        #   print("\t ", c)
        obstacles.append(cl_k)
        continue

      f_cl.append(cl_k)

    for i, cl in enumerate(f_cl):
      center_t = min(cl, key=lambda x: x[0])
      len_to_center = center_t[0] + LIDAR_RADIUS
      res = fit_circle(cl, np.array([len_to_center * np.cos(center_t[1]), len_to_center * np.sin(center_t[1])]))
      coords = res.x

      # convert to polar
      r = np.sqrt(np.square(coords[0]) + np.square(coords[1]))
      theta = np.arctan2(coords[1], coords[0])

      robots.append((r, theta))

    result[LIDAR_ROBOTS] = robots
    result[LIDAR_OBSTACLES] = obstacles
    result[LIDAR_ALL] = all

    return result


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


class ThreeRobotMatcher(object):

  def __init__(self, leader_set, f1_set, f2_set):
    self._lrs = leader_set
    self._frs = [f1_set, f2_set]
    self._mfrs = []
    self._followers = []

    for i, set in enumerate(self._frs):
      for r in set:
        self._mfrs.append((i, r))

    self._match()

  @staticmethod
  def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi])

  @staticmethod
  def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])

  def _match(self):

    followers = [None, None]

    # if the leader can only see one follower, and one of the followers can see both the leader and other follower
    if len(self._lrs) == 1:
      # matches = self._find_matches(self._lrs[0], self._mfrs)
      # match is (f_index, diff, fr, lrs[0])

      middle_f, lower_f = ((0, self._frs[0]), (1, self._frs[1])) if len(self._frs[0]) > 1 else ((1, self._frs[1]), (0, self._frs[0]))
      middle_f = [(middle_f[0], r) for r in middle_f[1]]
      lower_f = [(lower_f[0], r) for r in lower_f[1]]

      middle_matches = self._find_matches(self._lrs[0], middle_f)
      middle_match = middle_matches[0]
      m_i = middle_match[0]
      followers[m_i] = (middle_match[3], middle_match[2])

      # match = matches[0]
      # m_i = match[0]
      # followers[m_i] = (self._lrs[0], match[3])

      # match_f_cart is cartesian vector from middle follower to leader (in mf frame)
      # match_l_cart is cartesian vector from leader to middle follower (in leader frame)
      match_f_cart = self.pol2cart(*middle_match[2])
      match_l_cart = self.pol2cart(*middle_match[3])

      # index of the other follower
      o_i = lower_f[0][0]

      # the other robot found is presumably from the other follower.
      # other_f is from middle follower to end follower (in mf frame)
      # other_ff is from end follower to middle follower (in ef frame)
      other_f = [m for (f_i, m) in middle_f if m != middle_match[2]][0]
      other_ff = lower_f[0][1]

      of_cart = self.pol2cart(*other_f)
      off_cart = self.pol2cart(*other_ff)

      om_l_cart = match_l_cart + of_cart
      om_f_cart = match_f_cart + off_cart

      om_l = self.cart2pol(*om_l_cart)
      om_f = self.cart2pol(*om_f_cart)

      print("**LINE")

      followers[o_i] = (om_l, om_f)

    else:  # leader can see both followers

      f1_set = []
      f2_set = []

      # for i, lr in enumerate(self._lrs):
      #   # matches = self._find_matches(lr, self._mfrs)
      #   frs1 = [(0, m) for m in self._frs[0]]
      #   frs2 = [(1, m) for m in self._frs[0]]
      #   f1_matches = self._find_matches(lr, frs1)
      #   f2_matches = self._find_matches(lr, frs2)
      #
      #   followers[0] = (f1_matches[0][3], f1_matches[0][2])
      #   followers[1] = (f2_matches[0][3], f2_matches[0][2])

      for j, lr in enumerate(self._lrs):
        lr_dist = lr[0]

        for i, fr in enumerate(self._frs[0]):
          fr_dist = fr[0]
          diff = np.abs(lr_dist - fr_dist)
          f1_set.append((j, diff, fr, lr))

        for i, fr in enumerate(self._frs[1]):
          fr_dist = fr[0]
          diff = np.abs(lr_dist - fr_dist)
          f2_set.append((j, diff, fr, lr))

      f1_set = sorted(f1_set, key=lambda x: x[1])
      f2_set = sorted(f2_set, key=lambda x: x[1])

      f1 = f1_set[0]
      f2 = f2_set[0]

      if f1[0] == f2[0]:
        if len(f1_set) == 1:
          f2 = f2_set[1]
        elif len(f2_set) == 1:
          f1 = f1_set[1]
        else:
          if f1[1] > f2[1]:
            f1 = f1_set[1]
          else:
            f2 = f2_set[1]

      followers[0] = (f1[3], f1[2])
      followers[1] = (f2[3], f2[2])


      # f1_set = []
      # f2_set = []
      #
      # f1_pred = (lrs[0], lrs[0])
      # f2_pred = (lrs[1], lrs[1])
      #
      # if len(f1rs) >= 2 and len(f2rs) >= 2:
      #
      #   for j,lr in enumerate(lrs):
      #
      #     lr_dist = lr[0]
      #
      #     for i, fr in enumerate(f1rs):
      #       fr_dist = fr[0]
      #       diff = np.abs(lr_dist - fr_dist)
      #       f1_set.append((j, diff, fr))
      #
      #     for i, fr in enumerate(f2rs):
      #       fr_dist = fr[0]
      #       diff = np.abs(lr_dist - fr_dist)
      #       f2_set.append((j, diff, fr))
      #
      #   f1_set = sorted(f1_set, key=lambda x: x[1])
      #   f2_set = sorted(f2_set, key=lambda x: x[1])
      #
      #   f1 = f1_set[0]
      #   f2 = f2_set[0]
      #
      #   if f1[0] == f2[0]:
      #     if f1[1] > f2[1]:
      #       f1 = f1_set[1]
      #     else:
      #       f2 = f2_set[1]

      print("FORWARD")

        # if len(f1_matches) == 1 or len(f2_matches) == 1:
        #   followers[0] = (f1_matches[0][3], f1_matches[0][2])
        #   followers[1] = (f2_matches[0][3], f2_matches[0][2])
        # else:
        #
        #
        # m_ind = 0
        #
        # while True:
        #   match = matches[m_ind]
        #   m_i = match[0]
        #   if followers[m_i] is None:
        #     followers[m_i] = (match[3], match[2])
        #     break
        #   m_ind += 1

    self._followers = followers
    print("FOLLOWERS", followers)
    print()

  def _find_matches(self, needle, haystack):

    n_dist = needle[0]
    n_ang = needle[1]

    res = []
    for (i, r) in haystack:
      dist = r[0]
      ang = r[1]
      diff = np.abs(n_dist - dist)
      res.append((i, diff, r, needle))

    return sorted(res, key= lambda x: x[1])

  @property
  def followers(self):
    return self._followers


class RobotControl(object):

  def __init__(self, followers, leader_vel, desired_pose):
    self._followers = followers
    self._leader_vel = leader_vel
    self._desired_pose = desired_pose

  def basic(self, max_speed, max_angular):

    k = np.array([0.45, 0.24])
    d = 0.05
    angular_coeff = 5
    speed_coeff = 5

    velocities = [0]*2

    for i, follower in enumerate(self._followers):

      # lf is (r, theta) from leader to follower
      # fl is (r, theta) from follower to leader
      lf = follower[0]
      fl = follower[1]

      z = np.array([0., 0.])
      z[0] = lf[0]
      z[1] = lf[1]

      if z[1] < 0.:
        z[1] += 2 * np.math.pi

      # this gets the angle between the bearing of the leader and follower (from frame of follower)
      beta = np.pi + fl[1] - lf[1]
      gamma = beta + z[1]

      G = np.array([[np.cos(gamma), d * np.sin(gamma)],
                    [-np.sin(gamma) / z[0], d * np.cos(gamma) / z[0]]])
      F = np.array([[-np.cos(z[1]), 0],
                    [np.sin(z[1]) / z[0], -1]])

      print('\t z<->zs', z, ' <-> ', self._desired_pose[FOLLOWERS[i]])
      p = k * (self._desired_pose[FOLLOWERS[i]] - z)
      print('\t p k * (zs - z)', p)

      speed_robot = np.array([self._leader_vel.linear.x, self._leader_vel.angular.z])
      vel_follower = np.matmul(np.linalg.inv(G), (p - np.matmul(F, speed_robot)))

      vel_msg = Twist()

      vel_msg.linear.x = np.clip(vel_follower[0], -max_speed*speed_coeff, max_speed*speed_coeff)
      vel_msg.angular.z = np.clip(vel_follower[1], -max_angular * angular_coeff, max_angular * angular_coeff)
      velocities[i] = vel_msg

    return velocities

  def three_robot(self):
    pass

  def three_robot_with_potential_field(self):
    pass


zs_desired = {FOLLOWERS[0]: np.array([0.4, 3.*np.math.pi/4.]),
              FOLLOWERS[1]: np.array([0.75, 4.3*np.math.pi/4.])}
# right triangle, two sides 0.4
#                  l12,  psi12          , l13,   l23
zs_both_desired = [0.4, 5.*np.math.pi/4., 0.4, np.sqrt(0.32)]
#              psi13,            psi23
extra_psis = [3.*np.math.pi/4., np.math.pi/2.]

speed_coefficient = 1.


def run():
  global zs_desired
  global speed_coefficient

  rospy.init_node('robot_controller')
  rate_limiter = rospy.Rate(ROSPY_RATE)

  l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
  f_publishers = [None] * len(FOLLOWERS)
  for i, follower in enumerate(FOLLOWERS):
    f_publishers[i] = rospy.Publisher('/' + follower + '/cmd_vel', Twist, queue_size=5)

  leader_laser = SimpleLaser(LEADER, True)
  follower_lasers = [SimpleLaser(FOLLOWER_1), SimpleLaser(FOLLOWER_2)]

  stop_msg = Twist()
  stop_msg.linear.x = 0.
  stop_msg.angular.z = 0.

  # Make sure the robot is stopped.
  i = 0
  while i < 10 and not rospy.is_shutdown():
    l_publisher.publish(stop_msg)
    for f_publisher in f_publishers:
      f_publisher.publish(stop_msg)

    rate_limiter.sleep()
    i += 1

  max_speed = 0.06
  max_angular = 0.06

  while not rospy.is_shutdown():
    if not leader_laser.ready:
      rate_limiter.sleep()
      continue

    # print('measurments', leader_laser.measurements)
    # u, w = rule_based(*leader_laser.measurements)
    u, w = obstacle_avoidance.braitenberg(*leader_laser.measurements)
    u *= speed_coefficient * 0.25
    w *= speed_coefficient * 0.25
    print('vels', u, w)
    vel_msg_l = Twist()
    vel_msg_l.linear.x = np.clip(u, -max_speed, max_speed)
    vel_msg_l.angular.z = np.clip(w, -max_speed, max_speed)
    l_publisher.publish(vel_msg_l)

    print()
    print("LEADER: FINDING ROBOTS")
    l_res = leader_laser.cluster_environment()
    lrs = l_res[LIDAR_ROBOTS]
    lobs = l_res[LIDAR_OBSTACLES]
    lall = l_res[LIDAR_ALL]
    print()
    print("FOLLOWER1: FINDING ROBOTS")
    f1_res = follower_lasers[0].cluster_environment()
    f1rs = f1_res[LIDAR_ROBOTS]
    f1obs = f1_res[LIDAR_OBSTACLES]
    f1all = f1_res[LIDAR_ALL]
    print()
    print("FOLLOWER2: FINDING ROBOTS")
    f2_res = follower_lasers[1].cluster_environment()
    f2rs = f2_res[LIDAR_ROBOTS]
    f2obs = f2_res[LIDAR_OBSTACLES]
    f2all = f2_res[LIDAR_ALL]
    print()
    print("ROBOTS FROM LEADER PERSPECTIVE:", lrs)
    print("ROBOTS FROM FOLLOWER1 PERSPECTIVE:", f1rs)
    print("ROBOTS FROM FOLLOWER2 PERSPECTIVE:", f2rs)

    print()

    # if the robots can't see eachother (with the leader seeing at least one follower)
    if not (len(lrs) > 0 and ((len(f1rs) > 0 and len(f2rs) > 1) or (len(f2rs) > 0 and len(f1rs) > 1))):
      speed_coefficient = np.abs(speed_coefficient) * 0.95
      f_publishers[0].publish(stop_msg)
      f_publishers[1].publish(stop_msg)
      rate_limiter.sleep()
      continue
    else:
      speed_coefficient = 1.

    # match the observed robots from the lidar to {leader, follower1, follower2}
    matcher = ThreeRobotMatcher(lrs, f1rs, f2rs)
    fps = matcher.followers

    # initiate the control class
    control = RobotControl(fps, vel_msg_l, zs_desired)

    # get the follower velocities calling the desired control algo
    velocities = control.basic(max_speed, max_angular)

    for i, f_publisher in enumerate(f_publishers):
      f_publisher.publish(velocities[i])

    rate_limiter.sleep()


if __name__ == '__main__':
  run()
