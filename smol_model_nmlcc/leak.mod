NEURON {
  SUFFIX leak
  NONSPECIFIC_CURRENT ileak
  RANGE conductance, eleak
}

PARAMETER {
  conductance = 0.00001 (uS)
  eleak = 10 (mV)
}

BREAKPOINT {
  LOCAL g

  g = conductance
  ileak = g * (v + -1 * eleak)
}

