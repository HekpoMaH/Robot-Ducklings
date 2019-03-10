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

# Import the potential_field.py code rather than copy-pasting.
# directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
# sys.path.insert(0, directory)
# try:
import rrt_improved as rrt
# except ImportError:
#   raise ImportError('Unable to import potential_field.py. Make sure this file is in "{}"'.format(directory))

SPEED = .2
EPSILON = .1

X = 0
Y = 1
YAW = 2


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


class SLAM(object):
  def __init__(self):
    rospy.Subscriber('/turtlebot03/map', OccupancyGrid, self.callback)
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

  def update(self):
    # Get pose w.r.t. map.
    a = 'occupancy_grid'
    b = 'turtlebot03/base_link'
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

  @property
  def ready(self):
    return self._occupancy_grid is not None and not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose

  @property
  def occupancy_grid(self):
    return self._occupancy_grid


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


def run(args):
  rospy.init_node('rrt_navigation')

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/turtlebot03/cmd_vel', Twist, queue_size=5)
  while True:
      l_publisher = rospy.Publisher('/turtlebot03/cmd_vel', Twist, queue_size=5)
      vel_msg_l = Twist()
      vel_msg_l.linear.x = .5
      l_publisher.publish(vel_msg_l)
      rate_limiter.sleep()
  path_publisher = rospy.Publisher('/path', Path, queue_size=1)
  slam = SLAM()
  goal = GoalPose()
  frame_id = 0
  current_path = []
  previous_time = rospy.Time.now().to_sec()

  # Stop moving message.
  stop_msg = Twist()
  stop_msg.linear.x = 0.
  stop_msg.angular.z = 0.

  # Make sure the robot is stopped.
  i = 0
  while i < 10 and not rospy.is_shutdown():
    publisher.publish(stop_msg)
    rate_limiter.sleep()
    i += 1

  while not rospy.is_shutdown():
    slam.update()
    current_time = rospy.Time.now().to_sec()

    # Make sure all measurements are ready.
    # Get map and current position through SLAM:
    # > roslaunch exercises slam.launch
    if not goal.ready or not slam.ready:
      rate_limiter.sleep()
      continue

    goal_reached = np.linalg.norm(slam.pose[:2] - goal.position) < .2
    if goal_reached:
      publisher.publish(stop_msg)
      rate_limiter.sleep()
      continue

    # Follow path using feedback linearization.
    position = np.array([
      slam.pose[X] + EPSILON * np.cos(slam.pose[YAW]),
      slam.pose[Y] + EPSILON * np.sin(slam.pose[YAW])], dtype=np.float32)
    v = get_velocity(position, np.array(current_path, dtype=np.float32))
    u, w = feedback_linearized(slam.pose, v, epsilon=EPSILON)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Update plan every 1s.
    time_since = current_time - previous_time
    if current_path and time_since < 2.:
      rate_limiter.sleep()
      continue
    previous_time = current_time

    # Run RRT.
    start_node, final_node = rrt.rrt(slam.pose, goal.position, slam.occupancy_grid)
    current_path = get_path(final_node)
    if not current_path:
      print('Unable to reach goal position:', goal.position)

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

    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs RRT navigation')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
