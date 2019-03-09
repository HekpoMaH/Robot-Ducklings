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

from sim import vector_length, get_alpha, dot_prod
# that's just for prototyping
import obstacle_avoidance

FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
EPSILON = 0.001

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

ZS_DESIRED = {FOLLOWERS[0]: [0.4, 3. * np.math.pi / 4.],
              FOLLOWERS[1]: [0.4, 5.*np.math.pi/4.]}

z_obstacle_desired = {FOLLOWERS[0]: [0.8, 0.30],
                      FOLLOWERS[1]: [0.8, 0.30]}

def set_distance_and_bearing(robot_name, dist, bearing):
    """ Bearing is always within [0; 2pi], not [-pi;pi] """
    global ZS_DESIRED
    zs_desired[robot_name] = [dist, bearing]


def getR(theta):
    """ Gets rotation matrix """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def find_virtual_robot(robot_pose, leader_pose, angles, measurements, min_idx):
    
    # print(robot_pose)
    # print(leader_pose)
    # print(angles)
    # print(measurements)
    # print(min_idx, measurements[min_idx])

    # robot_dir is a unit vector, min_dir = vector to the min point
    robot_dir = np.array([np.cos(robot_pose[YAW]), np.sin(robot_pose[YAW])])
    print('\t \t robot dir', robot_dir)
    leader_dir = np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])])
    print('\t \t leader dir', leader_dir)
    min_dir = np.matmul(getR(angles[min_idx]), robot_dir) * measurements[min_idx]
    print('\t \t min dir', min_dir)
    print('\t \t min pos', robot_pose[:-1]+min_dir)

    # first neighbour and first tangent
    n1_idx = (len(angles)+min_idx-1)%len(angles)
    n1_angle = angles[n1_idx]
    print('\t \t n1 angle', n1_angle)
    n1_dir = np.matmul(getR(n1_angle), robot_dir) * measurements[n1_idx]
    print('\t \t n1 dir', n1_dir)
    print('\t \t n1 pos', robot_pose[:-1]+n1_dir)
    t1_dir = n1_dir - min_dir
    print('\t \t t1 dir', t1_dir)



    # second neighbour and first tangent
    n2_idx = (len(angles)+min_idx+1)%len(angles)
    n2_angle = angles[n2_idx]
    print('\t \t n2 angle', n2_angle)
    n2_dir = np.matmul(getR(n2_angle), robot_dir) * measurements[n2_idx]
    print('\t \t n2 dir', n2_dir)
    print('\t \t n2 pos', robot_pose[:-1]+n2_dir)
    t2_dir = n2_dir - min_dir
    print('\t \t t2 dir', t2_dir)

    
    print('\t \t dot_prods', dot_prod(t1_dir, leader_pose[:-1]-robot_pose[:-1]), dot_prod(t2_dir, leader_pose[:-1]-robot_pose[:-1]))
    if dot_prod(t1_dir, leader_pose[:-1]-robot_pose[:-1]) >= dot_prod(t2_dir, leader_pose[:-1]-robot_pose[:-1]):
        tangent_dir = t1_dir
    else:
        tangent_dir = t2_dir

    virtual_robot = robot_pose[:-1] + min_dir
    virtual_robot = np.append(virtual_robot, np.math.pi + np.arctan2(*tangent_dir))
    print('\t vrob', virtual_robot)
    return virtual_robot


def find_nearest(angles, angle_to_robot):
    eps = 0.2
    angles = np.abs(angles - angle_to_robot)
    print('angles', angles)
    idxs = np.argwhere(angles < eps)
    return idxs

d = 0.05
k = np.array([0.45, 0.04])
max_speed = 0.28
max_angular = 0.28

def set_vel_no_obstacle(follower, follower_pose, leader_pose, speed_leader):
    global ZS_DESIRED, d, k, max_speed, max_angular


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

    print('\t zs vs desired:', z, zs_desired[follower])
    p = k * (zs_desired[follower]-z)

    # speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
    speed_follower = np.matmul(np.linalg.inv(G), (p-np.matmul(F, speed_leader)))
    print('\t skorosta1', speed_follower)
    speed_follower[0] = max(min(speed_follower[0], max_speed), -max_speed)
    speed_follower[1] = max(min(speed_follower[1], max_angular), -max_angular)

    return speed_follower

def set_vel_with_obstacle(follower, follower_pose, leader_pose, speed_leader, virtual_pose, delta):
    global ZS_DESIRED, d, max_speed, max_angular
    k = [0.045, 0.4]

    z = np.array([0., 0.])
    z[0] = vector_length(leader_pose[:-1] - follower_pose[:-1])
    z[1] = delta
    # z[1] = 

    beta = leader_pose[YAW] - follower_pose[YAW]
    
    bearing = get_alpha(np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])]),
                        follower_pose[:-1] - leader_pose[:-1])
    gamma = beta + bearing
    gamma_obstacle = np.clip(virtual_pose[YAW] - follower_pose[YAW], -np.math.pi, np.math.pi)
    print('\t go', gamma_obstacle, virtual_pose[YAW], follower_pose[YAW])
    print('\t zs vs desired:', z, z_obstacle_desired[follower])

    G = np.array([[np.cos(gamma), d*np.sin(gamma)],
                [np.sin(gamma_obstacle), d*np.cos(gamma_obstacle)]])
    F = np.array([-speed_leader[0]*np.cos(bearing), 0])

    p = k * (z_obstacle_desired[follower]-z)

    # speed_robot = np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])

    print('\t matricata', (G))
    print('\t matricata', np.linalg.inv(G), p-F, p, F)
    speed_follower = np.matmul(np.linalg.inv(G), p-F)
    print('\t skorosta', speed_follower)
    speed_follower[0] = max(min(speed_follower[0], max_speed), -max_speed)
    speed_follower[1] = max(min(0.08*speed_follower[1], max_angular), -max_angular)

    return speed_follower

def run():
    global ZS_DESIRED, d, k
    rospy.init_node('robot_controller')

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
    #                                -np.math.pi = np.math.pi and I don't want that
    angles = np.linspace(-np.math.pi, np.math.pi - EPSILON, 360)
    f_lasers = [SimpleLaser(FOLLOWERS[0], angles, cone_view=5), SimpleLaser(FOLLOWERS[1], angles, cone_view=5)]

    previous_time = rospy.Time.now().to_sec()

    # Make sure the robot is stopped.
    i = 0
    while i < 10 and not rospy.is_shutdown():
      l_publisher.publish(stop_msg)
      for f_publisher in f_publishers:
          f_publisher.publish(stop_msg)

      rate_limiter.sleep()
      i += 1
    



    cnt = 0

    while not rospy.is_shutdown():
        if not leader_laser.ready or not f_lasers[0].ready or not f_lasers[1].ready:
            rate_limiter.sleep()
            continue
        
        print('measurments', leader_laser.measurements)
        # u, w = rule_based(*leader_laser.measurements)
        u, w = obstacle_avoidance.braitenberg(*leader_laser.measurements)
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

        if leader_pose[YAW] < 0.:
            leader_pose[YAW] += 2 * np.math.pi

        print('leader_pose', leader_pose)
        print('leader_speed', vel_msg_l.linear.x, vel_msg_l.angular.z)
        for i, follower in enumerate(FOLLOWERS):

            follower_pose = slam.get_pose(follower)
            if follower_pose[YAW] < 0.:
                follower_pose[YAW] += 2*np.math.pi

            angle_to_leader = get_alpha(
                    np.array([np.cos(leader_pose[YAW]), np.sin(leader_pose[YAW])]),
                    follower_pose[:-1] - leader_pose[:-1]
            )

            # distance to the reference point P, but our robot is round
            distances = np.array([m-d for m in f_lasers[i].measurements])
            print(len(angles), 'and', len(distances))
            angles2 = angles[~np.isnan(distances)]
            distances = distances[~np.isnan(distances)]
            # print(distances)
            min_idx = np.argmin(distances)
            
            print('\t closest obstacle', distances[min_idx])
            # proceed as normal
            if distances[min_idx] > z_obstacle_desired[follower][1]:
                print('case 1')
                speed_follower = set_vel_no_obstacle(
                        follower,
                        follower_pose,
                        leader_pose,
                        np.array([vel_msg_l.linear.x, vel_msg_l.angular.z])
                )

            else:

                print('case 2')
                virtual_pose = find_virtual_robot(
                        follower_pose,
                        leader_pose,
                        angles2,
                        distances,
                        min_idx
                )
                
                speed_follower = set_vel_with_obstacle(
                        follower,
                        follower_pose,
                        leader_pose,
                        np.array([vel_msg_l.linear.x, vel_msg_l.angular.z]),
                        virtual_pose,
                        distances[min_idx]
                )




            vel_msg = Twist()
            # if cnt < 500:
            #     vel_msg = stop_msg
            #     cnt += 500
            # else:
            #     vel_msg.linear.x = speed_follower[0]
            #     vel_msg.angular.z = speed_follower[1]

            vel_msg.linear.x = speed_follower[0]
            vel_msg.angular.z = speed_follower[1]

            print('\t, follower ' + str(i) + ' speed:',  vel_msg.linear.x, vel_msg.angular.z)
            print()
            print()

            f_publishers[i].publish(vel_msg)
            


        rate_limiter.sleep()

if __name__ == '__main__':
    run()
