# !/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np


WALL_OFFSET = 2.
# CYLINDER_POSITION = np.array([.3, .2], dtype=np.float32)

CYLINDER_POSITION = np.array([.5, .0], dtype=np.float32)
CYLINDER_POSITION2 = np.array([.0, .5], dtype=np.float32)
CYLINDER_RADIUS = .3
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)

START_POSITION = np.array([-1.5, -1.5], dtype=np.float32)
MAX_SPEED = .5

def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v

def vector_length(v):
  return np.sqrt(v[0]**2 + v[1]**2)

def get_velocity_to_reach_goal(position, goal_position):

  v = np.zeros(2, dtype=np.float32)
  v = goal_position - position

  if vector_length(v) > MAX_SPEED:
    v = cap(v, MAX_SPEED)

  # NOT MISSING: Compute the velocity field needed to reach goal_position
  # assuming that there are no obstacles.

  return v

def dist_to_obstacle(position, obstacle_position, obstacle_radius):
  # gets the distance to the obstacle's wall
  dist = vector_length(position-obstacle_position)
  dist -= obstacle_radius
  return dist

def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
  v = np.zeros(2, dtype=np.float32)

  # If an obstacle is further away, (more than Q*) it should not
  # have any repulsive potential
  q_star = 0.75

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
    # towards infinity. 0.35 is just re-scaling factor
    vec *= 0.35*(q_star-d)/d

    # if it is an virtual force it is 2 times weaker,
    # so as to avoid pushing us into an obstacle
    for pos in extra_positions:
      if np.array_equal(obstacle, pos):
        vec /= 2

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
  v = cap(v, MAX_SPEED)
  return v


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n



extra_positions = [None]
extra_radii = [None]

def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(
      position,
      # I could only have one extra_position at a time.
      # For parts (e)-(f) cylinder 2 from above was added
      [CYLINDER_POSITION, CYLINDER_POSITION2] + extra_positions,
      [CYLINDER_RADIUS, CYLINDER_RADIUS] + extra_radii)
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid
  return cap(v, max_speed=MAX_SPEED)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='obstacle', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots()

  # Plot environment.
  ax.add_artist(plt.Circle(CYLINDER_POSITION, CYLINDER_RADIUS, color='gray'))
  # part (e)-(f)
  ax.add_artist(plt.Circle(CYLINDER_POSITION2, CYLINDER_RADIUS, color='gray'))
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.01
  x = START_POSITION
  positions = [x]
  eps = 1e-2

  for t in np.arange(0., 20., dt):
    v = get_velocity(x, args.mode)
    # <saddle> and <local minimum>
    if args.mode == 'all' and vector_length(v) < eps and vector_length(x-GOAL_POSITION) > eps:
        # Initially, I used the approach of adding noise, which looked like:
        # v += np.random.uniform(-0.1, 0.1, 2)

        # I'll always have exactly one extra force 
        extra_positions[0] = (x+np.random.uniform(-0.25, 0.25, 2))
        extra_radii[0] = 0
        # and now calculate velocity again
        v = get_velocity(x, args.mode)
    # </saddle> and </local minimum>
    x = x + v * dt
    positions.append(x)
  positions = np.array(positions)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

  # Plot field.
  X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.show()
  
