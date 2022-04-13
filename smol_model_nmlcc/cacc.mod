NEURON {
  SUFFIX cacc
  USEION ca READ cai
  NONSPECIFIC_CURRENT icl
  RANGE conductance, ecl
}

PARAMETER {
  conductance = 0.00001 (uS)
  ecl = -45 (mV)
}

BREAKPOINT {
  LOCAL gates_m_steadyState_x, g

  gates_m_steadyState_x = (1 + exp(1.1111111405455043 * (3.700000047683716 + -1000000 * cai)))^-1
  g = conductance * gates_m_steadyState_x
  icl = g * (v + -1 * ecl)
}

