NEURON {
  SUFFIX h
  NONSPECIFIC_CURRENT ih
  RANGE conductance, eh
}

PARAMETER {
  conductance = 0.00001 (uS)
  eh = -43 (mV)
}

STATE { gates_n_q }

INITIAL {
  LOCAL gates_n_steadyState_x

  gates_n_steadyState_x = (1 + exp(0.25 * (80 + v)))^-1
  gates_n_q = gates_n_steadyState_x
}

DERIVATIVE dstate {
  LOCAL gates_n_steadyState_x, gates_n_timeCourse_t

  gates_n_steadyState_x = (1 + exp(0.25 * (80 + v)))^-1
  gates_n_timeCourse_t = (exp(-14.600000381469727 + -0.0860000029206276 * v) + exp(-1.8700000047683716 + 0.07000000029802322 * v))^-1
  gates_n_q' = (gates_n_steadyState_x + -1 * gates_n_q) * gates_n_timeCourse_t^-1
}

BREAKPOINT {
  SOLVE dstate METHOD cnexp
  LOCAL g

  g = conductance * gates_n_q
  ih = g * (v + -1 * eh)
}

