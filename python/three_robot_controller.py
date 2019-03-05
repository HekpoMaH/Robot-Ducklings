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

from sim import vector_length, get_alpha
# that's just for prototyping
import obstacle_avoidance
# for the leg tracker
from people_msgs.msg import PositionMeasurementArray

# Import the rrt_improved.py code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
  import rrt_improved as rrt
except ImportError:
  raise ImportError('Unable to import rrt_impoved.py. Make sure this file is in "{}"'.format(directory))

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

SPEED = .2
EPSILON = .1

ROBOT_RADIUS = 0.105 / 2.

ROSPY_RATE = 50

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


def get_velocity(position, path_points):
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
    pass

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


class LegDetector(object):
  def __init__(self):
    rospy.Subscriber('/leg_tracker_measurements', PositionMeasurementArray, self.callback)
    self._position = np.array([np.nan, np.nan], dtype=np.float32)

  def callback(self, msg):
    # The pose from RViz is with respect to the "map".
    x = msg.people[0].pos.x
    y = msg.people[0].pos.y
    print("PERSON X", x)
    print("PERSON Y", y)
    print("NUMBER FOUND:", len(msg.people))
    print()

    self._position[X] = x
    self._position[Y] = y

  @property
  def ready(self):
    return not np.isnan(self._position[0])

  @property
  def position(self):
    return self._position


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


def get_path(final_node):
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


zs_desired = {FOLLOWERS[0]: [0.5, np.math.pi],
              FOLLOWERS[1]: [1.0, np.math.pi]}

# right triangle, two sides 0.4
#                  l12,  psi12          , l13,   l23
zs_both_desired = [0.4, 5. * np.math.pi / 4., 0.4, np.sqrt(0.32)]
#              psi13,            psi23
extra_psis = [3. * np.math.pi / 4., np.math.pi / 2.]


def set_distance_and_bearing(robot_name, dist, bearing):
  """ Bearing is always within [0; 2pi], not [-pi;pi] """
  global zs_desired
  zs_desired[robot_name] = [dist, bearing]


def run():
  global zs_desired
  rospy.init_node('three_robot_controller')

  path_publisher = rospy.Publisher('/path', Path, queue_size=1)
  l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
  f_publishers = [None] * len(FOLLOWERS)
  for i, follower in enumerate(FOLLOWERS):
    f_publishers[i] = rospy.Publisher('/' + follower + '/cmd_vel', Twist, queue_size=5)

  slam = SLAM()
  leg_detector = LegDetector()
  rate_limiter = rospy.Rate(ROSPY_RATE)
  frame_id = 0
  current_path = []
  previous_time = rospy.Time.now().to_sec()

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
  k = np.array([0.45, 0.24, 0.45, 0.45])

  cnt = 0

  while not rospy.is_shutdown():
    if not leader_laser.ready or not leg_detector.ready or not slam.ready:
      rate_limiter.sleep()
      continue

    slam.update(LEADER)
    current_time = rospy.Time.now().to_sec()
    leader_pose = slam.get_pose(LEADER)

    # chance of this happening if map-merge has not converged yet (or maybe some other reason)
    if leader_pose is None:
      rate_limiter.sleep()
      continue

    goal_reached = np.linalg.norm(leader_pose[:2] - leg_detector.position) < .2
    if goal_reached:
      l_publisher.publish(stop_msg)
      f_publishers[0].publish(stop_msg)
      f_publishers[1].publish(stop_msg)
      rate_limiter.sleep()
      continue

    # Follow path using feedback linearization.
    position = np.array([
      leader_pose[X] + EPSILON * np.cos(leader_pose[YAW]),
      leader_pose[Y] + EPSILON * np.sin(leader_pose[YAW])], dtype=np.float32)
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = feedback_linearized(leader_pose, v, epsilon=EPSILON)
    vel_msg_l = Twist()
    vel_msg_l.linear.x = u
    vel_msg_l.angular.z = w
    l_publisher.publish(vel_msg_l)

    # Update plan every 1s.
    time_since = current_time - previous_time
    if current_path and time_since < 2.:
      rate_limiter.sleep()
      continue
    previous_time = current_time

    # Run RRT.
    start_node, final_node = rrt.rrt_star(leader_pose, leg_detector.position, slam.occupancy_grid)
    current_path = get_path(final_node)
    if not current_path:
      print('Unable to reach goal position:', leg_detector.position)

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

    max_speed = 1.0
    max_angular = 0.7

    if leader_pose[YAW] < 0.:
      leader_pose[YAW] += 2 * np.math.pi
    print('leader_pose', leader_pose)
    print('leader_speed', vel_msg_l.linear.x, vel_msg_l.angular.z)

    # unfortunately the eqns and theorem for 3 robots (1 followr)
    # n robots' rules have to be handcrafted

    # there are also some other fancy complex methods in the paper,
    # but haven't investigated yet
    f1_pose = slam.get_pose(FOLLOWERS[0])
    f2_pose = slam.get_pose(FOLLOWERS[1])

    if f1_pose[YAW] < 0.:
      f1_pose[YAW] += 2 * np.math.pi

    if f2_pose[YAW] < 0.:
      f2_pose[YAW] += 2 * np.math.pi

    print('\t follower1_pose:', f1_pose)
    print('\t follower2_pose:', f2_pose)

    z = np.array([0., 0., 0., 0.])
    z[0] = vector_length(leader_pose[:-1] - f1_pose[:-1])
    z[1] = get_alpha(np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])]),
                     f1_pose[:-1] - leader_pose[:-1])
    z[2] = vector_length(leader_pose[:-1] - f2_pose[:-1])
    z[3] = vector_length(f1_pose[:-1] - f2_pose[:-1])

    #         g12
    gammas = [leader_pose[YAW] - f1_pose[YAW] + z[1]]
    #             g13
    gammas.append(leader_pose[YAW] - f2_pose[YAW] + extra_psis[0])
    #             g23
    gammas.append(f1_pose[YAW] - f2_pose[YAW] + extra_psis[1])
    # beta =
    # gamma = beta + z[1]

    G = np.array([[np.cos(gammas[0]), d * np.sin(gammas[0]), 0, 0],
                  [-np.sin(gammas[0]) / z[0], d * np.cos(gammas[0]) / z[0], 0, 0],
                  [0, 0, np.cos(gammas[1]), d * np.sin(gammas[1])],
                  [0, 0, np.cos(gammas[2]), d * np.sin(gammas[2])]])
    F = np.array([[-np.cos(z[1]), 0],
                  [np.sin(z[1]) / z[0], -1],
                  [-np.cos(extra_psis[0]), 0],
                  [0, 0]])

    print('\t zs', z, ' <-> ', zs_both_desired)
    p = k * (zs_both_desired - z)

    speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
    speed_follower = np.matmul(np.linalg.inv(G), (p - np.matmul(F, speed_robot)))
    print('\t', speed_follower)
    speed_follower[0] = max(min(0.5 * speed_follower[0], max_speed), -max_speed)
    speed_follower[1] = max(min(0.4 * speed_follower[1], max_angular), -max_angular)
    speed_follower[2] = max(min(0.5 * speed_follower[2], max_speed), -max_speed)
    speed_follower[3] = max(min(0.4 * speed_follower[3], max_angular), -max_angular)

    vel_msg = Twist()
    # if cnt < 500:
    #     vel_msg = stop_msg
    #     cnt += 500
    # else:
    #     vel_msg.linear.x = speed_follower[0]
    #     vel_msg.angular.z = speed_follower[1]

    vel_msg.linear.x = speed_follower[0]
    vel_msg.angular.z = speed_follower[1]

    print('\t, follower ' + str(0) + ' speed:', vel_msg.linear.x, vel_msg.angular.z)
    print()
    print()

    f_publishers[0].publish(vel_msg)

    vel_msg.linear.x = speed_follower[2]
    vel_msg.angular.z = speed_follower[3]
    print('\t, follower ' + str(1) + ' speed:', vel_msg.linear.x, vel_msg.angular.z)
    print()
    print()

    f_publishers[1].publish(vel_msg)

    rate_limiter.sleep()


if __name__ == '__main__':
  run()
