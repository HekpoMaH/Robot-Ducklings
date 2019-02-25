#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion


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


def rule_based(front, front_left, front_right, left, right):
  u = 0.  # [m/s]
  omega = 0.  # [rad/s] going counter-clockwise.

  I = np.array([left, front_left, front, front_right, right])
  I = np.clip(I, 0, 5)
  I = I - np.array([1.0, 1.5, 2.0, 1.5, 1.0])

  # so far similar as before (just weights are fine-tuned)

  # I'm implementing a mini subsumption architecture.
  # The base case (and lowest level) is wandering, which just drives us forward.
  # then we have checks for how close an obstacle is, what's on the left/right, etc.

  # wander case
  phi = [0.45, 0.45] # wheel's speeds
  
  # approaching sth on the front (front is defined as a weighted average of the three front sensors)
  if 1/4. * (I[1] + 2*I[2] + I[3]) < 0:
    # is it TOO close on the front
    if I[2] < -1.7:
      phi = [-0.3, -0.3] # reverse
      phi[1] += np.random.uniform(low=-0.5, high=0.3, size=1) # and a bit more (or less) on one side
      # so we could turm (NB. Sampling interval is [-0.5, 0.3) -> we will always reverse)

    # nothing on the left
    elif I[0] + I[1] > 0.0:
      phi = [0.1, 0.4] # left wheel goes slower, i.e. we go on left
    
    # nothing on the right
    elif I[3] + I[4] > 0.0:
      phi = [0.4, 0.1] # going on the right
    
    # there is an obstacle both on left and right
    else:
      # proceeding as in the first case
      phi = [-0.3, -0.3]
      phi[1] += np.random.uniform(low=-0.5, high=0.3, size=1)

  u = (phi[1] + phi[0]) / 2.
  omega = phi[1] - phi[0]

  return u, omega


class SimpleLaser(object):
  def __init__(self):
    rospy.Subscriber('/scan', LaserScan, self.callback)
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


class GroundtruthPose(object):
  def __init__(self, name='Robot1'):
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    self._name = name

  def callback(self, msg):
    idx = [i for i, n in enumerate(msg.name) if n == self._name]
    if not idx:
      raise ValueError('Specified name "{}" does not exist.'.format(self._name))
    idx = idx[0]
    self._pose[0] = msg.pose[idx].position.x
    self._pose[1] = msg.pose[idx].position.y
    _, _, yaw = euler_from_quaternion([
        msg.pose[idx].orientation.x,
        msg.pose[idx].orientation.y,
        msg.pose[idx].orientation.z,
        msg.pose[idx].orientation.w])
    self._pose[2] = yaw

  @property
  def ready(self):
    return not np.isnan(self._pose[0])

  @property
  def pose(self):
    return self._pose
  

def run(args):
  rospy.init_node('obstacle_avoidance')
  avoidance_method = globals()[args.mode]

  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
  laser = SimpleLaser()
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = GroundtruthPose()
  pose_history = []
  with open('/tmp/gazebo_exercise.txt', 'w'):
    pass

  print(args)
  while not rospy.is_shutdown():
    print(laser.ready, groundtruth.ready)
    # Make sure all measurements are ready.
    if not laser.ready or not groundtruth.ready:
      rate_limiter.sleep()
      continue

    print(laser.measurements)
    u, w = avoidance_method(*laser.measurements)
    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    # Log groundtruth positions in /tmp/gazebo_exercise.txt
    pose_history.append(groundtruth.pose)
    if len(pose_history) % 10:
      with open('/tmp/gazebo_exercise.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []
    rate_limiter.sleep()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
  parser.add_argument('--mode', action='store', default='braitenberg', help='Method.', choices=['braitenberg', 'rule_based'])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
