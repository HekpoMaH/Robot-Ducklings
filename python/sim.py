from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import time


# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Drawing constants.
REFRESH_RATE = 1. / 15.


def euler(current_pose, t, dt):
  next_pose = current_pose.copy()
  u = 0.25
  w = np.cos(t)
  # Uncomment below for 1hz action-perception loop
  # w = np.cos(np.floor(t))

  x_dot = u * np.cos(current_pose[YAW])
  y_dot = u * np.sin(current_pose[YAW])
  # theta_dot = w

  dx = x_dot * dt
  dy = y_dot * dt
  dtheta = w * dt

  next_pose = np.add(current_pose, np.array([dx, dy, dtheta], dtype=np.float32))
  # NOT MISSING :P : Use Euler's integration method to return the next pose of our robot.
  # https://en.wikipedia.org/wiki/Euler_method
  # t is the current time.
  # dt is the time-step duration.
  # current_pose[X] is the current x position.
  # current_pose[Y] is the current y position.
  # current_pose[YAW] is the current orientation of the robot.
  # Update next_pose[X], next_pose[Y], next_pose[YAW].

  return next_pose

def rk4(current_pose, t, dt, u, w):
  def f(t, y): # y is the pose of our robot, t is time
    # for 1hz action perception loop replace np.cos(t)
    # with np.cos(np.floor(t)).
    # Or at least that's how I understand: " rather than computing w as cosine
    # of the time parameter, you compute it
    # based on the cosine of the floor_value of your time parameter "
    return np.array([u * np.cos(y[YAW]), u * np.sin(y[YAW]), w])
  
  next_pose = current_pose.copy()
  
  k = np.zeros((4, 3))

  k[0] = dt * f(t, current_pose)
  k[1] = dt * f(t + dt/2, current_pose + k[0]/2)
  k[2] = dt * f(t + dt/2, current_pose + k[1]/2)
  k[3] = dt * f(t + dt, current_pose + k[2])
  next_pose += 1/6 * (k[0] + 2*k[1] + 2*k[2] + k[3])

  return next_pose


def vector_length(v):
  return np.sqrt(v[0]**2 + v[1]**2)

def dot_prod(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def cross_prod(v1, v2):
  return v1[0]*v2[1] - v2[0]*v1[1]

def get_alpha(v1, v2):
  alpha = np.arccos(min(max(dot_prod(v1, v2) / (vector_length(v1) * vector_length(v2)), -1.), 1.))
  if cross_prod(v1, v2) < 0:
    alpha = 2*np.math.pi-alpha
  
  return alpha


def main(args):
  print('Using method {}'.format(args.method))
  integration_method = globals()[args.method]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.ion()  # Interactive mode.
  plt.grid('on')
  plt.axis('equal')
  plt.xlim([-0.5, 2])
  plt.ylim([-0.75, 1.25])
  plt.show()
  colors = colors_from('jet', len(args.dt))

  z_desired = [0.4, np.math.pi]

  # hardcoded? TODO
  d=0.05
  k=np.array([1, 1])
  # Show all dt.
  for color, dt in zip(colors, args.dt):
    print('Using dt = {}'.format(dt))

    # Initial robot pose (x, y and theta).
    robot_pose = np.array([0., 0., 0.], dtype=np.float32)
    follower_pose = np.array([1, -1, np.math.pi/2])
    robot_drawer = RobotDrawer(ax, robot_pose, color=color, label='dt = %.3f [s]' % dt)
    follower_drawer = RobotDrawer(ax, follower_pose, color=colors_from('jet', 4)[3], label='madafaka')
    if args.animate:
      fig.canvas.draw()
      fig.canvas.flush_events()

    # Simulate for 10 seconds.
    last_time_drawn = 0.
    last_time_drawn_real = time.time()
    for t in np.arange(0., 40., dt):
      speed_robot = [0.25, 0.01]
      if t>20:
          speed_robot=[0.15, -.25]
      speed_follower = [0.25, -0.01]
      z = np.array([0., 0.])

      z[0] = vector_length(robot_pose[:-1]- follower_pose[:-1])
      z[1] = get_alpha(np.array([np.cos(robot_pose[YAW]), np.sin(robot_pose[YAW])]),
                       follower_pose[:-1]-robot_pose[:-1])


      beta = robot_pose[YAW] - follower_pose[YAW]
      gamma = beta +z[1]
      G=np.array([[np.cos(gamma), d*np.sin(gamma)],
                  [-np.sin(gamma)/z[0], d*np.cos(gamma)/z[0]]])
      F=np.array([[-np.cos(z[1]), 0],
                  [np.sin(z[1])/z[0], -1]])
      
      p = k * (z_desired-z)

      speed_follower = np.matmul(np.linalg.inv(G), (p-np.matmul(F, speed_robot)))
      speed_follower[0]=min(speed_follower[0], 0.5)

      robot_pose = integration_method(robot_pose, t, dt, *speed_robot)
      follower_pose = integration_method(follower_pose, t, dt, *speed_follower)
      
      plt.title('time = %.3f [s] with dt = %.3f [s]' % (t + dt, dt))
      robot_drawer.update(robot_pose)
      follower_drawer.update(follower_pose)

      # Do not draw too many frames.
      time_drawn = t
      if args.animate and (time_drawn - last_time_drawn > REFRESH_RATE):
        # Try to draw in real-time.
        time_drawn_real = time.time()
        delta_time_real = time_drawn_real - last_time_drawn_real
        if delta_time_real < REFRESH_RATE:
          time.sleep(REFRESH_RATE - delta_time_real)
        last_time_drawn_real = time_drawn_real
        last_time_drawn = time_drawn
        fig.canvas.draw()
        fig.canvas.flush_events()
    robot_drawer.done()

  plt.ioff()
  plt.title('Trajectories')
  plt.legend(loc='lower right')
  plt.show(block=True)


# Simple class to draw and animate a robot.
class RobotDrawer(object):

  def __init__(self, ax, pose, radius=.05, label=None, color='g'):
    self._pose = pose.copy()
    self._radius = radius
    self._history_x = [pose[X]]
    self._history_y = [pose[Y]]
    self._outside = ax.plot([], [], 'b', lw=2)[0]
    self._front = ax.plot([], [], 'b', lw=2)[0]
    self._path = ax.plot([], [], c=color, lw=2, label=label)[0]
    self.draw()

  def update(self, pose):
    self._pose = pose.copy()
    self._history_x.append(pose[X])
    self._history_y.append(pose[Y])
    self.draw()

  def draw(self):
    a = np.linspace(0., 2 * np.pi, 20)
    x = np.cos(a) * self._radius + self._pose[X]
    y = np.sin(a) * self._radius + self._pose[Y]
    self._outside.set_data(x, y)
    r = np.array([0., self._radius])
    x = np.cos(self._pose[YAW]) * r + self._pose[X]
    y = np.sin(self._pose[YAW]) * r + self._pose[Y]
    self._front.set_data(x, y)
    self._path.set_data(self._history_x, self._history_y)

  def done(self):
    self._outside.set_data([], [])
    self._front.set_data([], [])


def colors_from(cmap_name, ncolors):
    cm = plt.get_cmap(cmap_name)
    cm_norm = matplotlib.colors.Normalize(vmin=0, vmax=ncolors - 1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=cm_norm, cmap=cm)
    return [scalar_map.to_rgba(i) for i in range(ncolors)]


def positive_floats(string):
  values = tuple(float(v) for v in string.split(','))
  for v in values:
    if v <= 0.:
      raise argparse.ArgumentTypeError('{} is not strictly positive.'.format(v))
  return values


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Launches a battery of experiments in parallel')
  parser.add_argument('--method', action='store', default='rk4', help='Integration method.', choices=['euler', 'rk4'])
  parser.add_argument('--dt', type=positive_floats, action='store', default=(0.05,), help='Integration step.')
  parser.add_argument('--animate', action='store_true', default=True, help='Whether to animate.')
  args = parser.parse_args()
  main(args)
