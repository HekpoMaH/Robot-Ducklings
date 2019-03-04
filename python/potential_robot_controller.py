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

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
EPSILON = 0.001
MAX_SPEED = .5

ROSPY_RATE = 50

X = 0
Y = 1
YAW = 2

LEADER = 'tb3_0'
FOLLOWERS = ['tb3_1', 'tb3_2']
FOLLOWER_1 = 'tb3_1'
FOLLOWER_2 = 'tb3_2'

def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v

def feedback_linearized(pose, velocity, epsilon):
  # TODO Not a todo, just note that things are relative to robot's frame
  pose[YAW] = 0.
  # TODO
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # MISSING: Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.

  u = velocity[X] * np.cos(pose[YAW]) + velocity[Y] * np.sin(pose[YAW])
  w = (1 / epsilon) * (-velocity[X] * np.sin(pose[YAW]) + velocity[Y] * np.cos(pose[YAW]))

  return u, w

def dist_to_obstacle(position, obstacle_position, obstacle_radius):
  # gets the distance to the obstacle's wall
  dist = vector_length(position-obstacle_position)
  dist -= obstacle_radius
  return dist

def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii, q_star=0.30):
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
    
    # The position is inside obstacle, where the potential is inf,
    # therefore the gradient inside an obstacle is 0
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
    vec *= 2.5*(q_star-d)/d

    vec = cap(vec, MAX_SPEED)
    v += vec

  # NOT MISSING: Compute the velocity field needed to avoid the obstacles
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). Make sure to not create
  # speeds that are larger than max_speed for each obstacle. Both obstacle_positions
  # and obstacle_radii are lists.

  # In the end, cap the sum of obstacles, in order not to burn the robot's
  # motors
  # I DONT WANT IT CAPPED HERE
  v = cap(v, 500*MAX_SPEED)
  return v

class SimpleLaser(object):
  def __init__(self, robot_name, angles=[0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.], cone_view=10.):
    rospy.Subscriber('/'+robot_name+'/scan', LaserScan, self.callback)
    self._angles = angles
    self._width = np.pi / 180. * cone_view  # 10 degrees cone of view.
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

zs_desired = {FOLLOWERS[0]: [0.4, 5*np.math.pi/4.],
              FOLLOWERS[1]: [0.4, 3*np.math.pi/4.]}

# right triangle, two sides 0.4
#                  l12,  psi12          , l13,   l23
zs_both_desired = [0.4, 5.*np.math.pi/4., 0.4, np.sqrt(0.32)]
#              psi13,            psi23
extra_psis =  [3.*np.math.pi/4., np.math.pi/2.]

def set_distance_and_bearing(robot_name, dist, bearing):
    """ Bearing is always within [0; 2pi], not [-pi;pi] """
    global zs_desired
    zs_desired[robot_name] = [dist, bearing]

def get_potential_speed(pose, angles, measurements):


    tot = np.array([0., 0.])
    for alpha, lm in zip(angles, measurements):

        if np.isnan(lm):
            continue

        obstacle_position = pose[:2] + lm*np.array([np.cos(alpha), np.sin(alpha)])
        vec = get_velocity_to_avoid_obstacles(pose[:2], [obstacle_position], [.0])
        tot += vec

    tot = cap(tot, MAX_SPEED)
    print('\t total', tot)
    up, wp = feedback_linearized(pose, tot, .2)
    up *= .8
    wp *= .6
    return up, wp

def run():
    global zs_desired
    rospy.init_node('potential_robot_controller')

    l_publisher = rospy.Publisher('/' + LEADER + '/cmd_vel', Twist, queue_size=5)
    f_publishers = [None] * len(FOLLOWERS)
    for i, follower in enumerate(FOLLOWERS):
        f_publishers[i] = rospy.Publisher('/' + follower + '/cmd_vel', Twist, queue_size=5)

    slam = SLAM()

    rate_limiter = rospy.Rate(ROSPY_RATE)

    stop_msg = Twist()
    stop_msg.linear.x = 0.
    stop_msg.angular.z = 0.

    leader_laser = SimpleLaser(LEADER)

    angles = np.linspace(-np.math.pi, np.math.pi - EPSILON, 60)
    f1_laser = SimpleLaser(FOLLOWERS[0], angles=angles, cone_view=4)
    f2_laser = SimpleLaser(FOLLOWERS[1], angles=angles, cone_view=4)

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
    k2 = np.array([0.45, 0.04])

    cnt = 0

    while not rospy.is_shutdown():
        if not leader_laser.ready or not f1_laser.ready or not f2_laser.ready:
            rate_limiter.sleep()
            continue
        
        print('measurments', leader_laser.measurements)
        # u, w = rule_based(*leader_laser.measurements)
        u, w = obstacle_avoidance.braitenberg(*leader_laser.measurements)
        print('vels', u, w)
        vel_msg_l = Twist()
        max_speed = 0.5
        max_angular = 0.35
        vel_msg_l.linear.x = max(min(u, max_speed-0.25), -max_speed+0.25)
        vel_msg_l.angular.z = max(min(w, max_angular-0.1), -max_angular+0.1)
        l_publisher.publish(vel_msg_l)
        leader_pose = slam.get_pose(LEADER)

        # chance of this happening if map-merge has not converged yet (or maybe some other reason)
        if leader_pose is None:
          rate_limiter.sleep()
          continue


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
            f1_pose[YAW] += 2*np.math.pi

        if f2_pose[YAW] < 0.:
            f2_pose[YAW] += 2*np.math.pi

        print('\t follower1_pose:', f1_pose)
        print('\t follower2_pose:', f2_pose)

        z = np.array([0., 0., 0., 0.])
        z[0] = vector_length(leader_pose[:-1] - f1_pose[:-1])
        z[1] = get_alpha(np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])]),
                         f1_pose[:-1] - leader_pose[:-1])
        z[2] = vector_length(leader_pose[:-1] - f2_pose[:-1])
        z[3] = vector_length(f1_pose[:-1] - f2_pose[:-1])

        speed_follower = [0., 0., 0., 0.]

        if z[0] < 2 * zs_both_desired[0] and z[2] < 2 * zs_both_desired[0]:
            #         g12
            gammas = [leader_pose[YAW] - f1_pose[YAW] + z[1]]
            #             g13
            gammas.append(leader_pose[YAW] - f2_pose[YAW] + extra_psis[0])
            #             g23
            gammas.append(f1_pose[YAW] - f2_pose[YAW] + extra_psis[1])
            # beta = 
            # gamma = beta + z[1]

            G=np.array([[np.cos(gammas[0]), d*np.sin(gammas[0]), 0, 0],
                        [-np.sin(gammas[0])/z[0], d*np.cos(gammas[0])/z[0], 0, 0],
                        [0, 0, np.cos(gammas[1]), d*np.sin(gammas[1])],
                        [0, 0, np.cos(gammas[2]), d*np.sin(gammas[2])]])
            F=np.array([[-np.cos(z[1]), 0],
                        [np.sin(z[1])/z[0], -1],
                        [-np.cos(extra_psis[0]), 0],
                        [0, 0]])
        
            print('\t zs', z, ' <-> ', zs_both_desired)
            p = k * (zs_both_desired-z)

            speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
            speed_follower = np.matmul(np.linalg.inv(G), (p-np.matmul(F, speed_robot)))
            print('\t', speed_follower)


        else:
            speed_leader = [vel_msg_l.linear.x, vel_msg_l.angular.z]
            z = np.array([0., 0.])
            z[0] = vector_length(leader_pose[:-1] - f1_pose[:-1])
            z[1] = get_alpha(np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])]),
                             f1_pose[:-1] - leader_pose[:-1])

            beta = leader_pose[YAW] - f1_pose[YAW]
            gamma = beta + z[1]

            G=np.array([[np.cos(gamma), d*np.sin(gamma)],
                        [-np.sin(gamma)/z[0], d*np.cos(gamma)/z[0]]])
            F=np.array([[-np.cos(z[1]), 0],
                        [np.sin(z[1])/z[0], -1]])

            p = k2 * (zs_desired[FOLLOWERS[0]]-z)
            _speed_follower = np.matmul(np.linalg.inv(G), (p-np.matmul(F, speed_leader)))
            speed_follower[0] = _speed_follower[0]
            speed_follower[1] = _speed_follower[1]


            z[0] = vector_length(leader_pose[:-1] - f2_pose[:-1])
            z[1] = get_alpha(np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])]),
                             f2_pose[:-1] - leader_pose[:-1])

            beta = leader_pose[YAW] - f2_pose[YAW]
            gamma = beta + z[1]

            G=np.array([[np.cos(gamma), d*np.sin(gamma)],
                        [-np.sin(gamma)/z[0], d*np.cos(gamma)/z[0]]])
            F=np.array([[-np.cos(z[1]), 0],
                        [np.sin(z[1])/z[0], -1]])

            p = k2 * (zs_desired[FOLLOWERS[1]]-z)
            _speed_follower = np.matmul(np.linalg.inv(G), (p-np.matmul(F, speed_leader)))
            speed_follower[2] = _speed_follower[0]
            speed_follower[3] = _speed_follower[1]

        speed_follower[0] = max(min(0.6*speed_follower[0], max_speed), -max_speed)
        speed_follower[1] = max(min(0.4*speed_follower[1], max_angular), -max_angular)
        speed_follower[2] = max(min(0.6*speed_follower[2], max_speed), -max_speed)
        speed_follower[3] = max(min(0.4*speed_follower[3], max_angular), -max_angular)
        ut, wt = get_potential_speed(f1_pose, angles, f1_laser.measurements)

        vel_msg = Twist()
        # if cnt < 500:
        #     vel_msg = stop_msg
        #     cnt += 500
        # else:
        #     vel_msg.linear.x = speed_follower[0]
        #     vel_msg.angular.z = speed_follower[1]

        vel_msg.linear.x = max(min(speed_follower[0] + ut, max_speed), -max_speed)
        vel_msg.angular.z = max(min(speed_follower[1] + wt, max_angular), -max_angular)

        print('\t, follower ' + str(0) + ' potential:',  ut, wt)
        print('\t, follower ' + str(0) + ' speed:',  vel_msg.linear.x, vel_msg.angular.z)
        print()
        print()

        f_publishers[0].publish(vel_msg)
            
        ut, wt = get_potential_speed(f2_pose, angles, f2_laser.measurements)

        vel_msg.linear.x = max(min(speed_follower[2] + ut, max_speed), -max_speed)
        vel_msg.angular.z = max(min(speed_follower[3] + wt, max_angular), -max_angular)
        print('\t, follower ' + str(1) + ' potential:',  ut, wt)
        print('\t, follower ' + str(1) + ' speed:',  vel_msg.linear.x, vel_msg.angular.z)
        print()
        print()

        f_publishers[1].publish(vel_msg)

        rate_limiter.sleep()

if __name__ == '__main__':
    run()
