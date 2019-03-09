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
# for the leg tracker
from people_msgs.msg import PositionMeasurementArray
# that's just for prototyping
import obstacle_avoidance
import pprint
import itertools

import rrt_improved as rrt

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105
INTER_WHEEL_RADIUS = 0.8
ROBOT_WIDTH = 0.16
LIDAR_RADIUS = 0.035
EPSILON = .1

ROSPY_RATE = 50

LIDAR_ROBOTS = 0
LIDAR_OBSTACLES = 1
LIDAR_ALL = 2
LIDAR_RAW = 3

X = 0
Y = 1
YAW = 2
ANGLE = 2

ROBOT_COUNT = 3
LEADER = 'tb3_0'
FOLLOWERS = ['tb3_1', 'tb3_2']
FOLLOWER_1 = 'tb3_1'
FOLLOWER_2 = 'tb3_2'

F1_INDEX = 0
F2_INDEX = 1

HUMAN_MIN = 1.2
HUMAN_MAX = 3.5
HUMAN_CONE = np.pi
MIN_SEPARATION_DIST = 0.2

STOP = False

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
    self._robot_name = robot_name

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

    result = [[]]*4

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

    def get_lidar_fuzz(d):

      lf = 0.01
      uf = 0.01

      # if d < 0.2:
      #   uf = 0.02
      # elif d < 0.3:
      #   uf = 0.015
      # elif d < 0.7:
      #   uf = 0.01
      # elif d < 0.9:
      #   uf = 0.015
      # elif d < 1.2:
      #   uf = 0.0125
      # elif d < 1.4:
      #   uf = 0.014
      # else:
      #   uf = 0.01

      if d < 0.2:
        uf = 0.025
      elif d < 0.3:
        uf = 0.02
      elif d < 0.7:
        uf = 0.015
      elif d < 0.9:
        uf = 0.02
      elif d < 1.2:
        uf = 0.0175
      elif d < 1.4:
        uf = 0.0175
      else:
        uf = 0.015


      return lf, uf


    max_dist = 2.5
    min_points = 3
    t_sec = 0.05
    cluster_angle_mult = 1.1
    lidar_radius_fuzz = 0.01
    lidar_radius_gap = 0.05
    shape_mean_min = 1.4
    shape_mean_max = 1.6
    shape_std = 0.15

    s = [(dist, increment * index) for (index, dist) in enumerate(ranges)]
    s = delete_outliers(s)
    fs = []

    result[LIDAR_RAW] = s

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

      lower_fuzz, upper_fuzz = get_lidar_fuzz(center_d)

      a_span = diff_ang + increment * 2
      e_span_lidar = 2 * self.boundary_circ_angle(center_d + LIDAR_RADIUS + lower_fuzz,
                                                  LIDAR_RADIUS + lower_fuzz)
      e_span_lidar_p = 2 * self.boundary_circ_angle(center_d + LIDAR_RADIUS + lidar_radius_gap - upper_fuzz,
                                                    LIDAR_RADIUS + lidar_radius_gap - upper_fuzz)

      # e_span = 2 * self.boundary_circ_angle(center_d + ROBOT_RADIUS, ROBOT_RADIUS)
      # e_rect_span = 2 * self.boundary_rect_angle(center_d, ROBOT_WIDTH)
      # e_rect_lidar = 2 * self.boundary_rect_angle(center_d, 2 * LIDAR_RADIUS)
      # e_rect_lidar_p = 2 * self.boundary_rect_angle(center_d, 2 * (LIDAR_RADIUS+lidar_radius_fuzz))

      # if self._robot_name == LEADER:
      #   print("CENTER D", center_d)
      #   print("A SPAN", a_span)
        # print("E SPAN", e_span)
        # print("E SPAN LIDAR", e_span_lidar)
        # print("E RECT SPAN", e_rect_span)
        # print("E SPAN LIDAR P", e_span_lidar_p)
        # print("E RECssT LIDAR", e_rect_lidar)
        # print("E RECT LIDAR", e_rect_lidar_p)
        # print()

      # if np.abs(a_span - e_span_lidar) > 3 * increment:
      if a_span < e_span_lidar or a_span > e_span_lidar_p:
        # if self._robot_name == LEADER:
          # print("PURGING CLUSTER (1) AS OBSTACLE")
          # print("\t ", cl_k[0])
          # print("\t ...")
          # print("\t ", cl_k[len(cl_k)-1])
        obstacles.append(cl_k)
        continue
      # else:
      #   if self._robot_name == LEADER:
      #     print("KEEPING CLUSTER (1)")
      #     print("\t ", cl_k[0])
      #     print("\t ...")
      #     print("\t ", cl_k[len(cl_k) - 1])

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

    # if self._robot_name == LEADER:
    #   print("LEADER_RESULT LENGTH", len(result))

    return result


class LegDetector(object):
  def __init__(self, fps, ffs):
    rospy.Subscriber('/leg_tracker_measurements', PositionMeasurementArray, self.callback)
    self._position = np.array([np.nan, np.nan], dtype=np.float32)
    self._fps = fps
    self._ffs = ffs

  def callback(self, msg):



    highest_reliability = -1e9
    highest_reliable_person = None
    # print("messages are", msg.people)
    # This one below just discards the follower to leader (r,phi) TODO confirm that [0]th element of the followers is always coordinate of follower relative to leader
    relative_coords = [relative_coord[0] for relative_coord in self.other_robots]
    print("other robots are", relative_coords)

    robots_in_slam_coords = [
      # TODO If I didn't get the geometry wrong (50% chance), these should give coordinate of followers in global
      np.array([leader_pose[X] + relative_coord[0] * np.cos(leader_pose[YAW] + relative_coord[1]),
                leader_pose[Y] + relative_coord[0] * np.sin(leader_pose[YAW] + relative_coord[1])])
      for relative_coord in relative_coords]

    print("leader in SLAM", leader_pose)
    print("robots in SLAM", robots_in_slam_coords)

    print("distances to each detected leg")
    for j, robots in enumerate(robots_in_slam_coords):
      print("\tROBO", j)
      for i, person in enumerate(msg.people):
        print("\t\t ", i, "->", vector_length(robots - np.array([person.pos.x, person.pos.y])))

    for i, person in enumerate(msg.people):

      # if tags[i] == True:
      #     continue

      if person.reliability > highest_reliability:
        highest_reliable_person = person
        highest_reliability = person.reliability

    if highest_reliable_person is None:
      return

    x = highest_reliable_person.pos.x
    y = highest_reliable_person.pos.y

    # Sometimes this is a bit off
    # print("\t This is assumed to be the person")
    # print("\t X", x)
    # print("\t Y", y)
    # print()

    self._position[X] = x
    self._position[Y] = y

  @property
  def ready(self):
    return not np.isnan(self._position[0])

  @property
  def position(self):
    return self._position

  # @other_robots.setter
  def set_other_robots(self, other_robots):
    self.other_robots = other_robots


class LegDetector2(object):
  def __init__(self):
    rospy.Subscriber('/leg_tracker_measurements', PositionMeasurementArray, self.callback)
    self._position = np.array([np.nan, np.nan], dtype=np.float32)
    self._predictions = []
    self.other_robots = []
    self._ready = False

  def find_leg(self, fps, ffs):

    #copy predictions in case it gets rewritten during evalution of this func
    unfiltered_preds = list(self._predictions)
    preds = []

    # print("PREDS BEFORE FILTER", len(unfiltered_preds))

    for pred in unfiltered_preds:
      pos = np.array([pred.pos.x, pred.pos.y])
      pos_pol = ThreeRobotMatcher.cart2pol(*pos)

      # print("PRED")

      r = pos_pol[0]
      phi = pos_pol[1]

      # print("R", r, "PHI", phi)

      if phi > np.pi:
        phi -= 2*np.pi
      if phi < -np.pi:
        phi += 2*np.pi

      if r < HUMAN_MIN or r > HUMAN_MAX or np.abs(phi) > HUMAN_CONE/2:
        continue

      # print("PRED ADDED")

      preds.append(pred)

    leg = (None ,None)

    if len(preds) == 0:
      return leg

    # print("NUMBER OF PREDS AFTER FILTER", len(preds))

    if ffs is not None and len(ffs) != 0:
      ffs = [ffs]
      # print("FFS", ffs)

      fpreds = list(itertools.product(fps, ffs, preds))
      error = 2 * MIN_SEPARATION_DIST

      # print("LEG PERMS")

      for perm in fpreds:

        follower = perm[0]
        lf_pol = follower[0]

        ff_pol = perm[1]

        # print("FF_POL", ff_pol)

        person_pred = perm[2]
        pos = np.array([person_pred.pos.x, person_pred.pos.y])

        lf_cart = ThreeRobotMatcher.pol2cart(*lf_pol)
        ff_cart = ThreeRobotMatcher.pol2cart(*ff_pol)

        # print("\t follower", lf_pol)
        # print('\t possible leg', ThreeRobotMatcher.cart2pol(*pos))
        # print()

        diff = np.linalg.norm(pos - lf_cart) + np.linalg.norm(pos - ff_cart)

        if diff > error:
          error = diff
          leg = (pos, ThreeRobotMatcher.cart2pol(*pos))

    else:

      fpreds = list(itertools.product(fps, preds))
      error = MIN_SEPARATION_DIST

      # print("LEG PERMS")

      for perm in fpreds:

        follower = perm[0]
        lf_pol = follower[0]

        person_pred = perm[1]
        pos = np.array([person_pred.pos.x, person_pred.pos.y])

        lf_cart = ThreeRobotMatcher.pol2cart(*lf_pol)

        # print("\t follower", lf_pol)
        # print('\t possible leg', ThreeRobotMatcher.cart2pol(*pos))
        # print()

        diff = np.linalg.norm(pos - lf_cart)

        if diff > error:
          error = diff
          leg = (pos, ThreeRobotMatcher.cart2pol(*pos))

    return leg

  def callback(self, msg):

    self._ready = True

    self._predictions = []

    for i, person in enumerate(msg.people):
      self._predictions.append(person)

  @property
  def ready(self):
    return self._ready

  @property
  def position(self):
    return self._position

  # @other_robots.setter
  def set_other_robots(self, other_robots):
    self.other_robots = other_robots


class SLAM(object):
  def __init__(self):
    rospy.Subscriber('/tb3_0/map', OccupancyGrid, self.callback)
    self._tf = TransformListener()
    self._occupancy_grid = None
    self._pose = np.array([0., 0., 0.], dtype=np.float32)

  def callback(self, msg):
    print("SLAM CALLBACK CALLED")
    values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
    processed = np.empty_like(values)
    processed[:] = rrt.FREE
    processed[values < 0] = rrt.UNKNOWN
    processed[values > 50] = rrt.OCCUPIED
    processed = processed.T
    origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
    resolution = msg.info.resolution
    self._occupancy_grid = rrt.OccupancyGrid(processed, origin, resolution)

  def print_poses(self):
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

  def update(self, robot):
    # Get pose w.r.t. map.
    a = 'occupancy_grid'
    b = robot + '/base_link'
    if self._tf.frameExists(a) and self._tf.frameExists(b):
      try:
        t = rospy.Time(0)
        position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
        self._pose[X] = position[X]
        self._pose[Y] = position[Y]
        _, _, self._pose[YAW] = euler_from_quaternion(orientation)
      except Exception as e:
        print(e)
    else:
      print('Unable to find:', self._tf.frameExists(a), self._tf.frameExists(b))

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
    # print('ogrid', self._occupancy_grid, 'nan', np.isnan(self.pose[0]))
    # return self._occupancy_grid is not None and not np.isnan(self._pose[0])
    return self._occupancy_grid is not None

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
    self._ff = None

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

      middle_f, lower_f = ((0, self._frs[0]), (1, self._frs[1])) if len(self._frs[0]) > 1 else (
      (1, self._frs[1]), (0, self._frs[0]))
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

      # TODO: fix this as coordinate axis are different for each robot

      of_cart = self.pol2cart(*other_f)
      off_cart = self.pol2cart(*other_ff)

      om_l_cart = match_l_cart + of_cart
      om_f_cart = match_f_cart + off_cart

      om_l = self.cart2pol(*om_l_cart)
      om_f = self.cart2pol(*om_f_cart)

      print("**LINE")

      followers[o_i] = (om_l, om_f)

    else:  # leader can see both followers

      # if each follower sees the leader and the other follower...
      if len(self._frs[0]) > 1 and len(self._frs[1]) > 1:

        permutations = list(itertools.product(self._mfrs, self._lrs))
        f1_perms = [m for m in permutations if m[0][0] == 0]
        f2_perms = [m for m in permutations if m[0][0] == 1]
        nperms = list(itertools.product(f1_perms, f2_perms))
        fperms = list(itertools.product(self._frs[0], self._frs[1]))
        cperms = list(itertools.product(nperms, fperms))

        min_error = float('inf')

        res_set = []
        filtered_res = []

        for perm in cperms:
          p1 = perm[0][0]
          p2 = perm[0][1]
          ff12 = perm[1]

          pf1 = p1[0][1]
          pf2 = p2[0][1]
          p1l = p1[1]
          p2l = p2[1]
          ff1 = ff12[0]
          ff2 = ff12[1]

          if p1l == p2l or pf1 == ff1 or pf2 == ff2:
            continue

          # print()
          # print("\t f1<-->l", p1)
          # print("\t f2<-->l", p2)
          # print("f1<-->f2", ff12)

          error = np.abs(pf1[0] - p1l[0]) + np.abs(pf2[0] - p2l[0]) + np.abs(ff1[0] - ff2[0])
          res_set.append((error, (p1l, pf1), (p2l, pf2), (ff1, ff2)))

          if error < 0.4:
            filtered_res.append((error, (p1l, pf1), (p2l, pf2), (ff1, ff2)))

          # if error < min_error:
          #   min_error = error
          #   followers[0] = (p1l, pf1)
          #   followers[1] = (p2l, pf2)
          #   self._ff = (ff1, ff2)

        if len(filtered_res) != 0:
          # find the closest to the leader
          best = min(filtered_res, key=lambda x: x[1][1] + x[2][1])
          followers[0] = best[1]
          followers[1] = best[2]
          self._ff = best[3]
        else:
          sorted_res = sorted(res_set, key=lambda x: x[0])
          best = sorted_res[0]
          followers[0] = best[1]
          followers[1] = best[2]
          self._ff = best[3]



      else:
        permutations = list(itertools.product(self._mfrs, self._lrs))
        f1_perms = [m for m in permutations if m[0][0] == 0]
        f2_perms = [m for m in permutations if m[0][0] == 1]
        nperms = list(itertools.product(f1_perms, f2_perms))

        min_error = float('inf')

        for perm in nperms:
          p1 = perm[0]
          p2 = perm[1]

          pf1 = p1[0][1]
          pf2 = p2[0][1]
          p1l = p1[1]
          p2l = p2[1]

          if p1l == p2l:
            continue

          # print()
          # print("\t f1<-->l", p1)
          # print("\t f2<-->l", p2)
          # print("f1<-->f2", ff12)

          error = np.square(pf1[0] - p1l[0])

          if error < min_error:
            min_error = error
            followers[0] = (p1l, pf1)
            followers[1] = (p2l, pf2)

      # print()
      # print("PERM RESULT")
      # for f in followers:
      #   print("\t", f)

      # print("FORWARD")

    self._followers = followers
    # print("FOLLOWERS")
    # for i, f in enumerate(followers):
      # print("\t", i, f)

    # if self._ff is not None:
    #   print("\t FF", self._ff)

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

  @property
  def ff(self):
    return self._ff


class RobotControl(object):

  def __init__(self, followers, leader_vel, desired_pose):
    self._followers = followers
    self._leader_vel = leader_vel
    self._desired_pose = desired_pose

  def basic(self, max_speed, max_angular):

    k = np.array([0.45, 0.24])
    d = 0.05
    angular_coeff = 5
    speed_coeff = 3

    velocities = [0] * 2

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
      b1 = -np.pi + lf[1] - fl[1]
      b2 = lf[1] - fl[1]
      b3 = fl[1] - lf[1]
      b4 = np.pi - lf[1] - fl[1]
      b5 = 2*np.pi - beta
      print("BETA", beta)
      print("B1", b1)
      print("B2", b2)
      print("B3", b3)
      print("B4",b4)
      print("B5", 2*np.pi - beta)
      beta = b5

      gamma = beta + z[1]
      gamma = beta - z[1]

      # print("z[0] l_12", z[0])
      # print("z[1] psi_ij", z[1])
      # print("b_12", beta)
      # print("g_12", gamma)

      G = np.array([[np.cos(gamma), d * np.sin(gamma)],
                    [-np.sin(gamma) / z[0], d * np.cos(gamma) / z[0]]])
      F = np.array([[-np.cos(z[1]), 0],
                    [np.sin(z[1]) / z[0], -1]])

      # print('\t z<->zs', z, ' <-> ', self._desired_pose[FOLLOWERS[i]])
      p = k * (self._desired_pose[FOLLOWERS[i]] - z)
      # print('\t p k * (zs - z)', p)

      speed_robot = np.array([self._leader_vel.linear.x, self._leader_vel.angular.z])
      vel_follower = np.matmul(np.linalg.inv(G), (p - np.matmul(F, speed_robot)))

      vel_msg = Twist()

      vel_msg.linear.x = np.clip(vel_follower[0], -max_speed * speed_coeff, max_speed * speed_coeff)
      vel_msg.angular.z = np.clip(vel_follower[1], -max_angular * angular_coeff, max_angular * angular_coeff)
      velocities[i] = vel_msg

    return velocities

  def three_robot(self, max_speed, max_angular, ff12):
    z = np.array([0., 0., 0., 0.])

    f1 = self._followers[0]
    f2 = self._followers[1]

    lf1 = f1[0]
    lf2 = f2[0]
    f1l = f1[1]
    f2l = f2[1]

    # r f1 to leader
    z[0] = f1l[0]

    # theta leader to f1
    z[1] = lf1[1]

    if z[1] < 0.:
        z[1] += 2*np.math.pi

    # r f2 to leader
    z[2] = f2l[0]

    print("f1 <--> leader", z[0])
    print("theta leader \/ f1", z[1])
    print("f2 <--> leader", z[2])
    print("theta leader \/ f2", lf2[1])
    print("f1 <--true--> f2", ff12[0][0], ff12[1][0])

    # r f1 to leader
    # inner_ang = np.abs(lf1[1] - lf2[1])
    # if inner_ang > np.pi:
    #   inner_ang = 2 * np.pi - inner_ang
    #
    # print("f1 - l - f2 ang", inner_ang)

    # cosine rule
    # z[3] = np.sqrt(np.square(z[0]) + np.square(z[2]) - 2 * z[0] * z[2] * np.cos(inner_ang))
    # z[3] = (ff12[0][0] + ff12[1][0]) / 2
    z[3] = ff12[0][0]

    print("f1 <--> f2", z[3])

    print("f1l[1]", f1l[1])
    print("lf1[1]", lf1[1])
    print("f2l[1]", f2l[1])
    print("lf2[1]", lf2[1])

    psi_12 = z[1]
    psi_13 = lf2[1]
    psi_23 = ff12[0][1]

    b_12 = np.pi + (f1l[1] - lf1[1])
    b_13 = np.pi + (f2l[1] - lf2[1])
    b_23 = np.pi + ff12[0][1] - ff12[1][1]

    g_12 = b_12 + z[1]
    g_13 = b_13 + lf2[1]
    g_23 = b_23 + ff12[1][1]

    print()
    print("z[0] l_12", z[0])
    print("z[1] psi_12", z[1])
    print("z[2] l_13", z[2])
    print("z[3] l_23", z[3])
    print("b_12", b_12)
    print("b_13", b_13)
    print("b_23", b_23)
    print("psi_23", ff12[0][1])
    print("psi_32", ff12[1][1])


    d = 0.05
    G = np.array([[np.cos(g_12), d * np.sin(g_12), 0, 0],
                  [-np.sin(g_12) / z[0], d * np.cos(g_12) / z[0], 0, 0],
                  [0, 0, np.cos(g_13), d * np.sin(g_13)],
                  [-np.cos(psi_23), 0, np.cos(g_23), d * np.sin(g_23)]])

    F = np.array([[-np.cos(z[1]), 0],
                  [np.sin(z[1]) / z[0], -1],
                  [-np.cos(psi_13), 0],
                  [0, 0]])

    z1 = self._desired_pose[FOLLOWERS[0]]
    z2 = self._desired_pose[FOLLOWERS[1]]
    z1_coord = ThreeRobotMatcher.pol2cart(*z1)
    z2_coord = ThreeRobotMatcher.pol2cart(*z2)
    z12_coord = -z1_coord + z2_coord
    z12 = ThreeRobotMatcher.cart2pol(*z12_coord)

    l_arr = np.array([z2[0], z12[0]])

    zd = np.concatenate((self._desired_pose[FOLLOWERS[0]], l_arr), axis=0)

    k = np.array([0.45, 0.23, 0.45, 0.45])
    print('\t z_desired', zd)
    print("\t z_current", z)
    print("\t z_diff", zd - z)
    p = k * (zd - z)

    speed_robot = np.array([self._leader_vel.linear.x, self._leader_vel.angular.z])
    speed_followers = np.matmul(np.linalg.inv(G), (p - np.matmul(F, speed_robot)))

    print("\t p", p)
    print("\t speed_followers")
    for f in speed_followers:
      print("\t\t",f)

    vel_msgs = []
    vel_msg = Twist()

    speed_coeff = 5
    angular_coeff = 3
    vel_msg.linear.x = np.clip(speed_followers[0], -max_speed * speed_coeff, max_speed * speed_coeff)
    vel_msg.angular.z = np.clip(speed_followers[1], -max_angular * angular_coeff, max_angular * angular_coeff)
    vel_msgs.append(vel_msg)
    vel_msg.linear.x = np.clip(speed_followers[2], -max_speed * speed_coeff, max_speed * speed_coeff)
    vel_msg.angular.z = np.clip(speed_followers[3], -max_angular * angular_coeff, max_angular * angular_coeff)
    vel_msgs.append(vel_msg)
    return vel_msgs
    pass

  def three_robot_with_potential_field(self, max_speed, max_angular, obstacles_for_each_robot):

    def cap(v, max_speed):
      n = np.linalg.norm(v)
      if n > max_speed:
        return v / n * max_speed
      return v

    def dist_to_obstacle(position, obstacle_position, obstacle_radius):
      # gets the distance to the obstacle's wall
      dist = vector_length(position-obstacle_position)
      dist -= obstacle_radius
      return dist

    def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii, q_star=0.35):
      v = np.zeros(2, dtype=np.float32)

      # If an obstacle is further away, (more than Q*) it should not
      # have any repulsive potential

      for obstacle, radius in zip(obstacle_positions, obstacle_radii):

        if obstacle is None:
            continue

        # distance to cylinder's WALL, not CENTER
        d = dist_to_obstacle(position, obstacle, radius)
        if d > q_star:
          continue # skip that one
        
        # print(obstacle, )

        if d < 0.:
          v = 0
          break

        # take the vector pointing outwards of the obstacle
        vec = position - obstacle
        vec /= vector_length(vec) # normalise it
        # On the edge of the obstacle's field
        # Q* will be equal to the distance, so the gradient
        # will have magnitude 0.
        # The closer to the obstacle, the larger
        # the vector should be. As we approach the
        # obstacle and the distance approaches 0,
        # the gradient's magnitude goes
        # towards infinity.

        vec *= 0.005*(q_star-d)/d

        v += vec

      v = cap(v, max_speed)
      return v

    def get_potential_speed(pose, measurements, angles):
      tot = np.array([0., 0.])
      # could probably be done in 1 call instead of iterating, but for now, it's acceptable
      for alpha, lm in zip(angles, measurements):

          if np.isnan(lm) or np.isinf(lm):
              continue

          obstacle_position = pose[:2] + lm*np.array([np.cos(alpha), np.sin(alpha)])
          vec = get_velocity_to_avoid_obstacles(pose[:2], [obstacle_position], [0.])
          tot += vec

      up, wp = GoalFollower.feedback_linearized(pose, tot, .2)
      up *= .28
      wp *= .26
      print('\t \t total', up, wp)
      return up, wp

    for i,_ in enumerate(self._followers):
      obstacles_for_each_robot[i] = [obstacle for obstacles_sublist in obstacles_for_each_robot[i] for obstacle in obstacles_sublist]


    vel_msgs = self.basic(max_speed, max_angular)
    print('basic vels', vel_msgs)
    for i, follower in enumerate(self._followers):
        # Position of robot relative to it's own frame is always [0., 0., 0.]
        # print('\t lidar all', obstacles_for_each_robot[i])
        print('\t', i, 'potential from obstacles')
        ut, wt = get_potential_speed([0., 0., 0.], *zip(*obstacles_for_each_robot[i]))
        print('\t', i, 'potential from other robots')
        # ut2, wt2 = get_potential_speed([0., 0., 0.], *zip(*obstacles_for_each_robot[i][LIDAR_ROBOTS]))
        vel_msgs[i].linear.x += ut #+ 0.1 * ut2
        vel_msgs[i].angular.z += wt# + 0.1 * wt2

    return vel_msgs
    pass

  def total(self):
    pass


class GoalFollower(object):

  last_path_calc = 0
  path = []
  frame_id = 0
  path_publisher = rospy.Publisher('/path', Path, queue_size=1)

  def __init__(self, goal_pos, leader_pose):
    self._goal_pos = goal_pos
    self._leader_pose = leader_pose

  def get_velocity(self, occupancy_grid):

    goal_min = 0.05
    vel_msg = Twist()

    if np.linalg.norm(self._goal_pos - self._leader_pose[:-1]) < goal_min:
      vel_msg.linear.x = 0
      vel_msg.angular.z = 0
      return vel_msg

    position = np.array([
      self._leader_pose[X] + EPSILON * np.cos(self._leader_pose[YAW]),
      self._leader_pose[Y] + EPSILON * np.sin(self._leader_pose[YAW])], dtype=np.float32)

    current_time = rospy.Time.now().to_sec()
    if current_time - self.last_path_calc > 2.:

      print("LEADER POSE", self._leader_pose)
      print("GOAL POS", self._goal_pos)

      start_node, final_node = rrt.rrt_star(self._leader_pose, self._goal_pos, occupancy_grid)
      new_path = self.get_path(final_node)
      if new_path is not None:
        self.path = new_path

      self.last_path_cal = current_time

      # # Publish path to RViz.
      path_msg = Path()
      path_msg.header.seq = self.frame_id
      path_msg.header.stamp = rospy.Time.now()
      path_msg.header.frame_id = '/tb3_0/map'
      for u in self.path:
        pose_msg = PoseStamped()
        pose_msg.header.seq = self.frame_id
        pose_msg.header.stamp = path_msg.header.stamp
        pose_msg.header.frame_id = '/tb3_0/map'
        pose_msg.pose.position.x = u[X]
        pose_msg.pose.position.y = u[Y]
        path_msg.poses.append(pose_msg)
      self.path_publisher.publish(path_msg)

      self.frame_id += 1

    u, w = 0, 0

    if len(self.path) != 0:
      print("**PATH FOUND")

    if len(self.path) > 0:
      v = self.calc_velocity(position, np.array(self.path, dtype=np.float32))
      u, w = self.feedback_linearized(self._leader_pose, v, epsilon=EPSILON)

    print("U", u)
    print("W", w)
    print()

    vel_msg.linear.x = u
    vel_msg.angular.z = w

    return vel_msg

  # def find_free_close(self):
  #   rand = np.random.rand(2)
  #   pos = leg_detector.position + rand
  #   while not slam.occupancy_grid.is_free(pos):
  #     rand = np.random.rand(2)
  #     pos = leg_detector.position + rand
  #
  #   return pos

  def calc_velocity(self, position, path_points):
    v = np.zeros_like(position)
    if len(path_points) == 0:
      return v
    # Stop moving if the goal is reached.
    if np.linalg.norm(position - path_points[-1]) < .2:
      return v

    """
      This uses the path finding algorithm developed by Craig Reynolds in his paper
      https://www.red3d.com/cwr/papers/1999/gdc99steer.pdf

      The algorithm has been slightly modified as the current velocity is not given and in the original algorithm it is 
      needed to predict the future location of the robot. I use the current position in all places where Reynolds uses
      the future position. The algorithm still performs well. When the robot deviates from the path, it corrects with a 
      velocity proportional to its deviation. 

      An outline of the algorithm is given below:  

      step 1: Find the closest point, cp, on the path to the robots position, p

      step 2: Determine a target point, tp, a fixed distance ahead of cp on the path

      step 3: If distance(cp - p) is greater than the radius of the path, move the robot towards the target point. If not,
      move the robot parallel to the segment which contains tp. 

    """

    SPEED = .1
    segment_size = 0.05  # a segment spans 5cm
    radius = 0.02  # the max distance either side of the path before the robot corrects
    velocity_scale = SPEED  # scales the forward velocity (when not correcting)
    target_offset = segment_size * 0.2  # distance of target point ahead of the closest point to robot on path
    correction_scale = 10  # scales the correcting velocity
    correction_min = 1 * velocity_scale  # minimum correcting velocity
    correction_max = 2 * velocity_scale  # maximum correcting velocity

    def magnitude(v):
      return np.linalg.norm(v)

    def get_normal(point, a, b):
      ab_norm = (b - a) / magnitude(b - a)
      return a + np.dot(ab_norm, point - a) * ab_norm

    def point_in_segment(point, a, b):
      eps = 0.0001
      d = magnitude(point - a) + magnitude(point - b) - magnitude(a - b)
      return -eps < d or eps > d

    def truncate_vector(v, min, max):
      mag = magnitude(v)
      if mag < min:
        return min * v / mag
      if mag > max:
        return max * v / mag
      return v

    closest_point = None
    target_point = None
    target_segment = None
    min_dist = np.inf

    for i in range(len(path_points) - 2):
      a = path_points[i]
      b = path_points[i + 1]
      c = path_points[i + 2]

      normal_point = get_normal(position, a, b)
      norm_in_segment = point_in_segment(normal_point, a, b)

      if not norm_in_segment:
        normal_point = b

      # distance between the normal point and the position of the robot
      dist = magnitude(position - normal_point)

      # evaluate if smallest distance so far
      if dist < min_dist:

        min_dist = dist
        closest_point = normal_point

        # calculate distance to end of segment
        segment_distance_left = magnitude(b - closest_point)

        # if distance to end of segment is less than distance from closest point to target point, target in next segment
        if segment_distance_left < target_offset:
          target_point = (target_offset - segment_distance_left) * (c - b) / magnitude(c - b)
          target_segment = c - b
        else:
          target_point = target_offset * (b - a) / magnitude(b - a)
          target_segment = b - a

    # get the vector and distance between the robots position and closest point on the path
    cp_dir = closest_point - position
    cp_dist = magnitude(cp_dir)

    # if robot is outside of radius of path, move towards target point
    if cp_dist > radius:
      v = truncate_vector(correction_scale * target_point, correction_min, correction_max)
    else:  # otherwise move parallel to the target_segment
      v = velocity_scale * target_segment / magnitude(target_segment)

    return v

  def get_path(self, final_node):
    # Construct path from RRT solution.
    if final_node is None:
      return []
    path_reversed = []
    path_reversed.append(final_node)
    while path_reversed[-1].parent is not None:
      path_reversed.append(path_reversed[-1].parent)
    path = list(reversed(path_reversed))
    # Put a point every 5 cm.
    distance = 0.05
    offset = 0.
    points_x = []
    points_y = []
    for u, v in zip(path, path[1:]):
      center, radius = rrt.find_circle(u, v)
      du = u.position - center
      theta1 = np.arctan2(du[1], du[0])
      dv = v.position - center
      theta2 = np.arctan2(dv[1], dv[0])
      # Check if the arc goes clockwise.
      clockwise = np.cross(u.direction, du).item() > 0.
      # Generate a point every 5cm apart.
      da = distance / radius
      offset_a = offset / radius
      if clockwise:
        da = -da
        offset_a = -offset_a
        if theta2 > theta1:
          theta2 -= 2. * np.pi
      else:
        if theta2 < theta1:
          theta2 += 2. * np.pi
      angles = np.arange(theta1 + offset_a, theta2, da)
      offset = distance - (theta2 - angles[-1]) * radius
      points_x.extend(center[X] + np.cos(angles) * radius)
      points_y.extend(center[Y] + np.sin(angles) * radius)
      return zip(points_x, points_y)

  @staticmethod
  def feedback_linearized(pose, velocity, epsilon):

    u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
    w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))

    return u, w


zs_desired = {FOLLOWERS[0]: np.array([0.3, 5.*np.math.pi/8.]),
              FOLLOWERS[1]: np.array([0.8, 10.*np.math.pi/8.])}
# right triangle, two sides 0.4
#                  l12,  psi12          , l13,   l23
# zs_both_desired = [zs_desired[FOLLOWERS[0]], zs_desired[FOLLOWERS[1]]]
#              psi13,            psi23
extra_psis = [3.*np.math.pi/4., 5*np.math.pi/4.]

speed_coefficient = 1.



def run():
  global zs_desired
  global speed_coefficient

  rospy.init_node('robot_controller')
  rate_limiter = rospy.Rate(ROSPY_RATE)

  path_publisher = rospy.Publisher('/path', Path, queue_size=1)
  l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
  f_publishers = [None] * len(FOLLOWERS)
  for i, follower in enumerate(FOLLOWERS):
    f_publishers[i] = rospy.Publisher('/' + follower + '/cmd_vel', Twist, queue_size=5)

  slam = SLAM()
  leg_detector = LegDetector()
  frame_id = 0
  current_path = []

  leader_laser = SimpleLaser(LEADER, True)
  follower_lasers = [SimpleLaser(FOLLOWER_1), SimpleLaser(FOLLOWER_2)]

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

  max_speed = 0.06
  max_angular = 0.06

  while not rospy.is_shutdown():
    if not leader_laser.ready or not leg_detector.ready:
      print('laser', leader_laser.ready, 'leg', leg_detector.ready)
      rate_limiter.sleep()
      continue

    if not slam.ready:
      print('slam', slam.ready)
      rate_limiter.sleep()
      continue

    # chance of this happening if map-merge has not converged yet (or maybe some other reason)
    current_time = rospy.Time.now().to_sec()
    leader_pose = slam.get_pose(LEADER)
    if leader_pose is None:
      rate_limiter.sleep()
      continue

    goal_reached = np.linalg.norm(leader_pose[:2] - leg_detector.position) < .002
    if goal_reached:
      print("GOAL REACHED")
      l_publisher.publish(stop_msg)
      f_publishers[0].publish(stop_msg)
      f_publishers[1].publish(stop_msg)
      rate_limiter.sleep()
      continue
    # print('measurments', leader_laser.measurements)
    # u, w = rule_based(*leader_laser.measurements)

    # Follow path using feedback linearization.
    position = np.array([
      leader_pose[X] + EPSILON * np.cos(leader_pose[YAW]),
      leader_pose[Y] + EPSILON * np.sin(leader_pose[YAW])], dtype=np.float32)
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = feedback_linearized(leader_pose, v, epsilon=EPSILON)
    vel_msg_l = Twist()
    # TODO FIX These below
    vel_msg_l.linear.x = 3*u
    vel_msg_l.angular.z = w
    print("LEADER VeL MSG", vel_msg_l)
    l_publisher.publish(vel_msg_l) if not STOP else l_publisher.publish(stop_msg)



    # Update plan every 1s.
    time_since = current_time - previous_time
    if current_path and time_since < 2.:
      rate_limiter.sleep()
      continue
    previous_time = current_time

    def find_free_close(pos):
      rand = np.random.rand(2)
      pos = leg_detector.position + rand
      while not slam.occupancy_grid.is_free(pos):
        rand = np.random.rand(2)
        pos = leg_detector.position + rand

      return pos


    g_pos = find_free_close(leg_detector.position)

    # Run RRT.
    # positionnn = np.random.rand(2)*4-2
    # print(leg_detector.position)
    print("Goal POS", g_pos, 'Leader POS', leader_pose)
    start_node, final_node = rrt.rrt(leader_pose, g_pos, slam.occupancy_grid)


    current_path = get_path(final_node)

    # Publish path to RViz.
    path_msg = Path()
    path_msg.header.seq = frame_id
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = 'map'
    for u in current_path:
      pose_msg = PoseStamped()
      pose_msg.header.seq = frame_id
      pose_msg.header.stamp = path_msg.header.stamp
      pose_msg.header.frame_id = 'map'
      pose_msg.pose.position.x = u[X]
      pose_msg.pose.position.y = u[Y]
      path_msg.poses.append(pose_msg)
    path_publisher.publish(path_msg)

    # rate_limiter.sleep()
    frame_id += 1
    # TODO remove commented out codes
    # u, w = obstacle_avoidance.braitenberg(*leader_laser.measurements)
    # u *= speed_coefficient * 0.25
    # w *= speed_coefficient * 0.25
    # print('vels', u, w)
    # vel_msg_l = Twist()
    # vel_msg_l.linear.x = np.clip(u, -max_speed, max_speed)
    # vel_msg_l.angular.z = np.clip(w, -max_speed, max_speed)
    # l_publisher.publish(vel_msg_l)

    # print()
    # print("LEADER: FINDING ROBOTS")
    l_res = leader_laser.cluster_environment()
    lrs = l_res[LIDAR_ROBOTS]
    lobs = l_res[LIDAR_OBSTACLES]
    lall = l_res[LIDAR_ALL]
    # print()
    # print("FOLLOWER1: FINDING ROBOTS")
    f1_res = follower_lasers[0].cluster_environment()
    f1rs = f1_res[LIDAR_ROBOTS]
    f1obs = f1_res[LIDAR_OBSTACLES]
    f1all = f1_res[LIDAR_ALL]
    print()
    # print("FOLLOWER2: FINDING ROBOTS")
    f2_res = follower_lasers[1].cluster_environment()
    f2rs = f2_res[LIDAR_ROBOTS]
    f2obs = f2_res[LIDAR_OBSTACLES]
    f2all = f2_res[LIDAR_ALL]
    # print("RAW", f2_res[LIDAR_RAW])
    # print()
    # print("ROBOTS FROM LEADER PERSPECTIVE:", lrs)
    # print("ROBOTS FROM FOLLOWER1 PERSPECTIVE:", f1rs)
    # print("ROBOTS FROM FOLLOWER2 PERSPECTIVE:", f2rs)

    # print()

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
    ffs = matcher.ff
    print("Matched followers", fps)
    leg_detector.set_other_robots(fps)

    # initiate the control class
    control = RobotControl(fps, vel_msg_l, zs_desired)

    # get the follower velocities calling the desired control algo
    # velocities = control.basic(max_speed, max_angular)
    # velocities = control.three_robot(max_speed, max_angular)
    # print("F1ALL", f1all)
    velocities = control.three_robot_with_potential_field(max_speed, max_angular, [f1all, f2all])

    for i, f_publisher in enumerate(f_publishers):
      print('follower', i, velocities[i].linear.x, velocities[i].angular.z)
      f_publisher.publish(velocities[i]) if not STOP else f_publisher.publish(stop_msg)

    rate_limiter.sleep()

def run1():
  global zs_desired
  global speed_coefficient

  rospy.init_node('robot_controller')
  rate_limiter = rospy.Rate(ROSPY_RATE)

  l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
  f_publishers = [None] * len(FOLLOWERS)
  for i, follower in enumerate(FOLLOWERS):
    f_publishers[i] = rospy.Publisher('/' + follower + '/cmd_vel', Twist, queue_size=5)

  slam = SLAM()
  leader_laser = SimpleLaser(LEADER, True)
  follower_lasers = [SimpleLaser(FOLLOWER_1), SimpleLaser(FOLLOWER_2)]
  leg_detector = LegDetector2()

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

  max_speed = 0.04
  max_angular = 0.04

  while not rospy.is_shutdown():
    if not leader_laser.ready:
      print("***1")
      rate_limiter.sleep()
      continue

    if not leg_detector.ready:
      print("***2")
      rate_limiter.sleep()
      continue

    if not slam.ready:
      print("***3")
      rate_limiter.sleep()
      continue

    leader_pose = slam.get_pose(LEADER)
    if leader_pose is None:
      rate_limiter.sleep()
      continue

    l_res = leader_laser.cluster_environment()
    lrs = l_res[LIDAR_ROBOTS]

    f1_res = follower_lasers[0].cluster_environment()
    f1rs = f1_res[LIDAR_ROBOTS]

    f2_res = follower_lasers[1].cluster_environment()
    f2rs = f2_res[LIDAR_ROBOTS]

    # print()
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
    ffs = matcher.ff

    # find the leg wrt the leader
    leg_cart, leg_pol = leg_detector.find_leg(fps, ffs)

    print("LEG POLAR WRT LEADER", leg_pol)

    if leg_pol is None:
      vel_msg_l = stop_msg
    else:
      # convert to slam frame
      leg_slam_phi = leg_pol[1] + leader_pose[2]
      leg_slam = leader_pose[:-1] + ThreeRobotMatcher.pol2cart(leg_pol[0] - 0.2, leg_slam_phi)

      # get the velocity to the leg
      gf = GoalFollower(leg_slam, leader_pose)
      vel_msg_l = gf.get_velocity(slam.occupancy_grid)

    l_publisher.publish(vel_msg_l) if not STOP else l_publisher.publish(stop_msg)

    # initiate the control class
    control = RobotControl(fps, vel_msg_l, zs_desired)

    # get the follower velocities calling the desired control algo
    # ffs indicate that the two followers can see each other
    if ffs is not None:
      # velocities = control.three_robot(max_speed, max_angular, ffs)
      velocities = control.basic(max_speed, max_angular)
    else:
      velocities = control.basic(max_speed, max_angular)

    for i, f_publisher in enumerate(f_publishers):
      f_publisher.publish(velocities[i]) if not STOP else f_publisher.publish(stop_msg)

    rate_limiter.sleep()

def run2():
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
    l_publisher.publish(vel_msg_l) if not STOP else l_publisher.publish(stop_msg)

    # print()
    # print("LEADER: FINDING ROBOTS")
    l_res = leader_laser.cluster_environment()
    lrs = l_res[LIDAR_ROBOTS]
    lobs = l_res[LIDAR_OBSTACLES]
    lall = l_res[LIDAR_ALL]
    # print()
    # print("FOLLOWER1: FINDING ROBOTS")
    f1_res = follower_lasers[0].cluster_environment()
    f1rs = f1_res[LIDAR_ROBOTS]
    f1obs = f1_res[LIDAR_OBSTACLES]
    f1all = f1_res[LIDAR_ALL]
    # print()
    # print("FOLLOWER2: FINDING ROBOTS")
    f2_res = follower_lasers[1].cluster_environment()
    f2rs = f2_res[LIDAR_ROBOTS]
    f2obs = f2_res[LIDAR_OBSTACLES]
    f2all = f2_res[LIDAR_ALL]

    # print()
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
    ffs = matcher.ff

    # initiate the control class
    control = RobotControl(fps, vel_msg_l, zs_desired)

    # get the follower velocities calling the desired control algo

    # ffs indicate that the two followers can see each other
    if ffs is not None:
      velocities = control.basic(max_speed, max_angular)
      # velocities = control.basic(max_speed, max_angular)
    else:
      velocities = control.basic(max_speed, max_angular)

    for i, f_publisher in enumerate(f_publishers):
      f_publisher.publish(velocities[i]) if not STOP else f_publisher.publish(stop_msg)

    rate_limiter.sleep()

if __name__ == '__main__':
  run2()
