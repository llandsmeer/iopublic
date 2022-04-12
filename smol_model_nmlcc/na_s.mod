NEURON {
  SUFFIX na_s
  USEION na WRITE ina READ ena
  RANGE conductance
}

PARAMETER {
  conductance = 0.00001 (uS)
}

STATE { gates_h_q }

INITIAL {
  LOCAL gates_h_steadyState_x

  gates_h_steadyState_x = (1 + exp(0.17241378743356547 * (70 + v)))^-1
  gates_h_q = gates_h_steadyState_x
}

DERIVATIVE dstate {
  LOCAL gates_h_steadyState_x, gates_h_timeCourse_t

  gates_h_steadyState_x = (1 + exp(0.17241378743356547 * (70 + v)))^-1
  gates_h_timeCourse_t = 3 * exp(-0.030303030303030304 * (40 + v))
  gates_h_q' = (gates_h_steadyState_x + -1 * gates_h_q) * gates_h_timeCourse_t^-1
}

BREAKPOINT {
  SOLVE dstate METHOD cnexp
  LOCAL gates_m_steadyState_x, gates_m_fcond, fopen0, g

  gates_m_steadyState_x = (1 + exp(-0.18181818181818182 * (30 + v)))^-1
  gates_m_fcond = gates_m_steadyState_x * gates_m_steadyState_x * gates_m_steadyState_x
  fopen0 = gates_h_q * gates_m_fcond
  g = conductance * fopen0
  ina = g * (v + -1 * ena)
}

