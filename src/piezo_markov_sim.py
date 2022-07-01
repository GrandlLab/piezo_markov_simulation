import numpy as np
import pandas as pd
import random
import math
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import numba
from numba import jit, prange

#####################################################################################################
def generate_transition_matrix(T, T50, slope, tau):

    '''used to generate a probability matrix of channel transition probabilities to determine the starting equilibrium states of channels. 
    Output is used for the equilibriumStates function.
    
    Input:
      T: tension being applied to each channel. For determineing equilibrium states, this value should be 0mN/m
      T50: The tension at which half of the channels in a population will open. For piezo1, use 1.4 mN/m
      slope: the slope at the T50 value of a Boltzman fit of normalized current v tension plot. For piezo1, this value is 0.8
      tau:How seconds per iteration. To prevent oscillations, needs to be at minimum 1/10**5 for tensions 1-5mN/m, 1/10**6 for tension 6-7Mn/N, and 1/10**7 for tension 8mN/m 
      
      Outputs:
        np.matrix: matrix of transition probabilties for each channel state'''

    c_o = 5.1 * tau
    c_i = ((5.1 * np.exp(1.25 * T) * 8.0 * 34.6 * np.exp(-1.25 * T)) / (5 * np.exp(T50/slope) * 0.4)) * tau
    c_i2 = 0
    c_c = 1-(c_o + c_i + c_i2)
    
    o_c = 5 * np.exp(T50/slope)*tau
    o_i = 8 * tau
    o_i2 = 4 * tau
    o_o = 1-(o_c + o_i + o_i2)
    
    i_o = 0.4 * tau
    i_c = 34.6 * np.exp(-1.25*T) * tau
    i_i2 = 0
    i_i = 1-(i_o + i_c + i_i2)
    
    i2_c = 0
    i2_o = 0.6 * tau
    i2_i = 0
    i2_i2 = 1- i2_o
    
    return np.matrix([[c_c, c_o, c_i, c_i2],[o_c, o_o, o_i, o_i2],[i_c, i_o, i_i, i_i2],[i2_c, i2_o, i2_i, i2_i2]])

def equilibriumStates(T):

    '''Used to determine the starting probabilities of channels in a given state. As a whole, the populaiton of channels will start at an equlibrated state.
    used in tandem with the function generate_transition_matrix.
  
    Inputs:
      T: matrix of transition probabilties for each channel state. Generated from the function generate_transition_matrix.
    
    Outputs:
      Matrix of probabilties for each channel state at equilibrium'''

    nStates = np.shape(T)[1]
    a = np.vstack([T.T - np.eye(4), np.ones(4)])
    b = np.vstack(np.append(np.zeros(4), [1], axis=0))
    return np.linalg.solve(a.T * a, a.T * b)


def generate_reference_tension(x_size, y_size, diff_coeff, poke_radius):

  '''determines a reference point to normalize the tension profile to. This function finds the pixel with the highest tension
  in the area outside of the tension clamped stimulus and it's value can be used to normalize the other tension pixels in the following functions
  
  Inputs: 
    x_size: the size of the membrane in the x dimension
    y_size:the size of the membrane in the x dimension
    diff_coeff: how quickly tension diffuses in pixels/s
    poke_radius: the radius of the tension clamped stimuli
    
  Output:
    reference pixel: location on the membrane/array where the reference pixel is found'''

  membrane = np.zeros((y_size, x_size))
  for j in range(y_size):
    for k in range(x_size):
      time = 1
      y_distance_from_center = abs((j + 1 - (0.5 * y_size))) 
      x_distance_from_center = abs((k + 1 - (0.5 * x_size))) 
      radius = math.sqrt((x_distance_from_center**2) + (y_distance_from_center**2)) - poke_radius
      tension_value = ((math.e ** (-radius**2 / (4 * diff_coeff * time))) /(4 * math.pi * diff_coeff * time)) 
      membrane[j, k] = tension_value
  flat_reference_pixel = np.argmax(membrane[:, :])
  y_reference_pixel = math.floor(flat_reference_pixel / y_size)
  x_reference_pixel = flat_reference_pixel - (y_reference_pixel * y_size)
  reference_pixel = [y_reference_pixel, x_reference_pixel]
  return(reference_pixel)


@jit(nopython = True)
def channel_update_pre(x, y, state_array, sampling_interval, seed_num):

  '''determines transition of each channel from t=frame 0 to t=frame1. Only used for the first step. Called one time for every channel passed
  through the function do_timestep_pre (below)
  
  Inputs:
    x: the x position of the channel on the membrane/array
    y: the y position of the channel on the membrane/array
    state_array: the first array that has channels scattered throughout in different states. Only used for the first timesetp
    sampling_interval: How seconds per iteration. To prevent oscillations, needs to be at minimum 1/10**5 for tensions 1-5mN/m, 1/10**6 for tension 6-7Mn/N, and 1/10**7 for tension 8mN/m 
    seed_num: seed number used for any random functions
    
    Outputs:
      statex: what state that one specific channel will transition to for that timestep'''

  channel = state_array[x, y]

  #importing the transition probabilities when no tension is added
  c_o0 = (5.1 * math.e ** (1.25 * 0)) * sampling_interval
  c_i0 = ((5.1 * math.e ** (1.25 * 0) * 8.0 * 34.6 * math.e **(-1.25 * 0)) / (5 * math.e**(1.4/0.8) * 0.4)) * sampling_interval
  o_i0 = 8 * sampling_interval
  o_c0 = 5 * math.e**(1.4/0.8)* sampling_interval
  o_i20 = 4 * sampling_interval
  i_o0 = 0.4 * sampling_interval
  i_c0 = (34.6 * math.e**(-1.25 * 0)) * sampling_interval
  i2_o0 = 0.6 * sampling_interval


  if channel == 5:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([10, 15, 5], [c_o0, c_o0 + c_i0, 1]):
      if draw <= probability:
        break
    return statex

  elif channel == 10:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([5, 15, 20, 10], [o_c0, o_c0 + o_i0, o_c0 + o_i0 +o_i20,  1]):
      if draw <= probability:
        break
    return statex

  if channel == 15:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([10, 5, 15], [i_o0, i_o0 + i_c0, 1]):
      if draw <= probability:
        break
    return statex

  if channel == 20:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([10, 20], [i2_o0, 1]):
      if draw <= probability:
        break
    return statex


def do_timestep_pre(state_array, sampling_interval, seed_num, channel_locs):

  '''Used to determine the transitions of all channels in a starting array from t = timestep0 to t = timestep1. Only used for the first step
  
  Inputs: 
    state_array: the first array that has channels scattered throughout in different states. Only used for the first timesetp
    sampling_interval: How seconds per iteration. To prevent oscillations, needs to be at minimum 1/10**5 for tensions 1-5mN/m, 1/10**6 for tension 6-7Mn/N, and 1/10**7 for tension 8mN/m 
    seed_num: seed number used for any random functions
    channel_locs: x,y coordinates of where the channels are located. Prevents having to cycle through each individual pixel and rather isolates only where channels are located
    
  Outputs:
    frame_array: an array with all of the channels in their new states. Same locations as state_array just with the updated channel states'''

  frame_array = np.zeros((state_array.shape[0], state_array.shape[1]))
  for j, k in zip(channel_locs[0], channel_locs[1]):
    updated_channel = channel_update_pre(j, k, state_array, sampling_interval, seed_num)
    frame_array[j, k] = updated_channel
  return frame_array

#Getting the tension value for a specific location
@jit(nopython=True)
def find_probability(x, y, t, poke_radius, diff_coeff, x_size, y_size):

  '''Determines the tension value at channel location. Called for every pixel passed through the function generate)tension (below)
  
  Inputs:
    x: the x position of the channel on the membrane/array where tension will be calculated
    y: the y position of the channel on the membrane/array where tension will be calculated
    t: time post stimuli onset
    poke_radius: the radius of the tension clamped stimuli
    diff_coeff: how quickly tension diffuses in pixels/s
    x_size: the size of the membrane in the x dimension
    y_size:the size of the membrane in the x dimension'''

  x_distance_from_center = abs((x + 1 - (0.5 * x_size))) 
  y_distance_from_center = abs((y + 1 - (0.5 * y_size))) 
  radius = math.sqrt((x_distance_from_center**2) + (y_distance_from_center**2)) - poke_radius
  tension_value = math.e ** (-radius**2 / (4 * diff_coeff * t)) /(4 * math.pi * diff_coeff * t)
  return tension_value


def generate_tension(x_size, y_size, tension, timepoint, poke_size, diff_coeff, channel_locs, reference_tension_pixel):

  '''generates an array, every timepoint, for tension values at channel locations in the array/membrane. These pixels are normalized so that the tension clamped stimuli
  is always 1 and then the tension that diffuses from the edge of the clamped stimuli is a fraction of that value based on the the rate of diffusion. These values are
  then all multiplied by an absolute tension, ranging from 1-8 mN/m, to generate the tension vlaues that channels will experience. For each pixel, calls the function find_probability.
  
  Inputs:
    x_size: the size of the membrane in the x dimension
    y_size:the size of the membrane in the x dimension
    tension: the absolute tension that channels will be exposed to in the tension clamped region of the membrane
    timepoint:time post stimuli onset
    poke_size: the radius of the tension clamped stimuli
    diff_coeff: how quickly tension diffuses in pixels/s
    channel_locs: x,y coordinates of where the channels are located. Prevents having to cycle through each individual pixel and rather isolates only where channels are located
    reference_tension_pixel: pixel used to normalize values outside the tension clamped region. Determined by the function generate_reference_tension

  Outputs:
    normalized_membrane: array of tension values at each pixel where a channel is located'''

  membrane = np.zeros((x_size, y_size))
  normalized_membrane = np.zeros((x_size, y_size))
  for j, k in zip(channel_locs[0], channel_locs[1]):
      tension_value = find_probability(j, k, timepoint, poke_size, diff_coeff, x_size, y_size)
      membrane[j, k] = tension_value
  #making a normalized probability distribution
  maxElement = find_probability(reference_tension_pixel[0], reference_tension_pixel[1], timepoint, poke_size, diff_coeff, x_size, y_size)
  max_ratio = 1 / maxElement
  normalized_membrane[:, :] = membrane[:, :] * max_ratio
  for j, k in zip(channel_locs[0], channel_locs[1]):
        x_distance_from_center = abs((j + 1 - (0.5 * x_size))) 
        y_distance_from_center = abs((k + 1 - (0.5 * y_size))) 
        radius = math.sqrt((x_distance_from_center**2) + (y_distance_from_center**2))
        if radius < poke_size:
          normalized_membrane[j, k] = 1
  normalized_membrane = normalized_membrane * tension
  return normalized_membrane

@jit(nopython=True)
def channel_update_post(x, y, state_array, sampling_interval, tension_array, seed_num):

  '''determines transition of each channel for all timesteps after the first one. Called one time for every channel passed
  through the function do_timestep_post (below). For this function, channel states are influenced by tension.
  
  Inputs:
    x: the x position of the channel on the membrane/array
    y: the y position of the channel on the membrane/array
    state_array: the  array that has channels scattered throughout in different states that was generated in the previous timestep
    sampling_interval: How seconds per iteration. To prevent oscillations, needs to be at minimum 1/10**5 for tensions 1-5mN/m, 1/10**6 for tension 6-7Mn/N, and 1/10**7 for tension 8mN/m 
    tension_array: array with tension values for locations where channels are located. These values are used to determine transition probabilities from one state to another using a Markov model
    seed_num: seed number used for any random functions
    
    Outputs:
      statex: what state that one specific channel will transition to for that timestep'''

  tension_value = tension_array[x, y]

  c_o = (5.1 * math.e ** (1.25 * tension_value)) * sampling_interval
  c_i = ((5.1 * math.e ** (1.25 * tension_value) * 8.0 * 34.6 * math.e**(-1.25 * tension_value)) / (5 * math.e**(1.4/0.8) * 0.4)) * sampling_interval

  o_i = 8 * sampling_interval
  o_c = 5 * math.e**(1.4/0.8) * sampling_interval
  o_i2 = 4 * sampling_interval

  i_o = 0.4 * sampling_interval
  i_c = (34.6 * math.e**(-1.25 * tension_value)) * sampling_interval

  i2_o = 0.6 * sampling_interval
  channel = state_array[x, y]
  if channel == 5:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([10, 15, 5], [c_o, c_o + c_i, 1]):
      if draw <= probability:
        break
    return statex

  elif channel == 10:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([5, 15, 20, 10], [o_c, o_c + o_i, o_c + o_i +o_i2,  1]):
      if draw <= probability:
        break
    return statex

  if channel == 15:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([10, 5, 15], [i_o, i_c + i_o, 1]):
      if draw <= probability:
        break
    return statex

  if channel == 20:
    random.seed(seed_num)
    draw = np.random.rand()
    for statex, probability in zip([10, 20], [i2_o, 1]):
      if draw <= probability:
        break
    return statex
                    
def do_timestep_post(state_array, sampling_interval, tension, timepoint, pokesize, diff_coeff, seed_num, channel_locs, reference_tension_pixel):

  '''Used to determine the transitions of all channels for all timepoitns after the first transition.
  
  Inputs: 
    state_array: the first array that has channels scattered throughout in different states. Only used for the first timesetp
    sampling_interval: How seconds per iteration. To prevent oscillations, needs to be at minimum 1/10**5 for tensions 1-5mN/m, 1/10**6 for tension 6-7Mn/N, and 1/10**7 for tension 8mN/m 
    tension: the absolute tension that channels will be exposed to in the tension clamped region of the membrane
    timepoint:time post stimuli onset
    poke_size: the radius of the tension clamped stimuli
    diff_coeff: how quickly tension diffuses in pixels/s
    seed_num: seed number used for any random functions
    channel_locs: x,y coordinates of where the channels are located. Prevents having to cycle through each individual pixel and rather isolates only where channels are located
    reference_tension_pixel: pixel used to normalize values outside the tension clamped region. Determined by the function generate_reference_tension

  Outputs:
    frame_array: an array with all of the channels in their new states'''

  frame_array = np.zeros((state_array.shape[0], state_array.shape[1]))
  tension_array = generate_tension(state_array.shape[0], state_array.shape[1], tension, timepoint, pokesize, diff_coeff, channel_locs, reference_tension_pixel)
  for j, k in zip(channel_locs[0], channel_locs[1]):
      updated_channel = channel_update_post(j, k, state_array, sampling_interval, tension_array, seed_num)
      frame_array[j, k] = updated_channel
  return frame_array

@jit(nopython = True)
def channel_state(s_time, channel_locs):

  '''Used to count the number of channels in each state at a given time.
  
  Inputs:
    s_time: Array of updated channel states for a given time.
    channel_locs: x,y coordinates of where the channels are located. Prevents having to cycle through each individual pixel and rather isolates only where channels are located
  
  Outputs:
    open: number of channels in the open state 
    closed: number of channels in the closed state 
    inactive: number of channels in the inactive state 
    inactive2: number of channels in the slow inactive state '''

  open = 0
  closed = 0
  inactive = 0
  inactive2 = 0
  for j, k in zip(channel_locs[0], channel_locs[1]):
    channel = s_time[j, k]
    if channel == 5:
      closed += 1
    if channel == 10:
      open += 1
    if channel == 15:
      inactive += 1
    if channel == 20:
      inactive2 += 1
  return open, closed, inactive, inactive2

##############################################################################################

#functions above, code below

##############################################################################################

#Imput variables:
tension = 5
x_size = 550
y_size = 550
fps = 10**5
equilibration_time = int(0.01 * fps)
time = int(0.05 * fps) # in frames. Will change based on fps
diffusion_coeff_list = [0.0001, 240, 2400, 24000]

##############################################################################################
equil_probabilities =  generate_transition_matrix(0, 1.4, 0.8, 1/fps)
equil_values_array = equilibriumStates(equil_probabilities)
equil_c = equil_values_array[0, 0]
equil_o = equil_values_array[1, 0]
equil_i = equil_values_array[2, 0]
equil_i2 =equil_values_array[3, 0]
data_array = np.zeros((len(diffusion_coeff_list), 20))
seed_list = [8, 34, 56, 32, 9, 65, 54, 90, 97, 13, 35, 67, 5467, 123, 567, 234, 567, 324, 65 , 2]
for diffusionx in range(len(diffusion_coeff_list)):
  for n in range(20):
    seed_num = seed_list[n]
    sampling_interval = 1/fps
    diff_coeff =diffusion_coeff_list[diffusionx]
    time_list = []
    open_list = []
    closed_list = []
    inactive_list = []
    inactive2_list = []
    poke_size = 200
    state_array = np.zeros((x_size, y_size))
    for i in range(state_array.shape[0]):
      for j in range(state_array.shape[1]):
        random.seed(seed_num)
        draw = np.random.rand()
        if draw <= 0.01:
          random.seed(seed_num)
          draw2 = np.random.rand()
          for statex, probability in zip([5, 10, 15, 20], [equil_c, equil_c + equil_o, equil_c + equil_o + equil_i, 1]):
            if draw2 <= probability:
              break
          state_array[i, j] = statex


    reference_tension_pixel = generate_reference_tension(x_size, y_size, diff_coeff, poke_size)
    channel_locs = np.where(state_array != 0)
    timepoint = 1/fps
    for t in range(time):
      if t < int(equilibration_time): 
        s_time = do_timestep_pre(state_array, sampling_interval, seed_num, channel_locs)
        open_sum, closed_sum, inactive_sum, inactive2_sum = channel_state(s_time, channel_locs)
        time_list.append(t)
        open_list.append(open_sum)
        closed_list.append(closed_sum)
        inactive_list.append(inactive_sum)
        inactive2_list.append(inactive2_sum)
        state_array = s_time
      else:
        s_time = do_timestep_post(state_array, sampling_interval, tension, timepoint, poke_size, diff_coeff, seed_num, channel_locs, reference_tension_pixel)
        open_sum, closed_sum, inactive_sum, inactive2_sum = channel_state(s_time,channel_locs)
        time_list.append(t)
        open_list.append(open_sum)
        closed_list.append(closed_sum)
        inactive_list.append(inactive_sum)
        inactive2_list.append(inactive2_sum)
        state_array = s_time
        timepoint += sampling_interval
    print(len(channel_locs))
    open_list = [x/3025 for x in open_list]
    closed_list = [x/3025 for x in closed_list]
    inactive_list = [x/3025 for x in inactive_list]
    inactive2_list = [x/3025 for x in inactive2_list]
    plt.plot(time_list, open_list)
    plt.plot(time_list, closed_list)
    plt.plot(time_list, inactive_list)
    plt.plot(time_list, inactive2_list)
    plt.ylabel('Percent Channels')
    plt.xlabel('Time(1e-5s)')
    plt.show()

    max_current = max(open_list)
    index = open_list.index(max_current)
    delay = (index-equilibration_time) * ((1/fps) * 10**3)
    data_array[diffusionx, n] = delay
    print(data_array)
