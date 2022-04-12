NEURON {
  SUFFIX k
  NONSPECIFIC_CURRENT ik
  RANGE conductance, ek
}

PARAMETER {
  conductance = 0.00001 (uS)
  ek = -75 (mV)
}

STATE { gates_n_q }

INITIAL {
  LOCAL gates_n_reverseRate_r, gates_n_forwardRate_x, gates_n_forwardRate_r, gates_n_inf

  gates_n_reverseRate_r = 1.690000057220459 * exp(-0.0125 * (35 + v))
  gates_n_forwardRate_x = 0.1 * (25 + v)
  if (gates_n_forwardRate_x != 0) {
    gates_n_forwardRate_r = 1.2999999523162842 * gates_n_forwardRate_x * (1 + -1 * exp(-1 * gates_n_forwardRate_x))^-1
  } else {
    if (gates_n_forwardRate_x == 0) {
      gates_n_forwardRate_r = 1.2999999523162842
    } else {
      gates_n_forwardRate_r = 0
    }
  }
  gates_n_inf = gates_n_forwardRate_r * (gates_n_forwardRate_r + gates_n_reverseRate_r)^-1
  gates_n_q = gates_n_inf
}

DERIVATIVE dstate {
  LOCAL gates_n_reverseRate_r, gates_n_forwardRate_x, gates_n_forwardRate_r, gates_n_inf, gates_n_tau

  gates_n_reverseRate_r = 1.690000057220459 * exp(-0.0125 * (35 + v))
  gates_n_forwardRate_x = 0.1 * (25 + v)
  if (gates_n_forwardRate_x != 0) {
    gates_n_forwardRate_r = 1.2999999523162842 * gates_n_forwardRate_x * (1 + -1 * exp(-1 * gates_n_forwardRate_x))^-1
  } else {
    if (gates_n_forwardRate_x == 0) {
      gates_n_forwardRate_r = 1.2999999523162842
    } else {
      gates_n_forwardRate_r = 0
    }
  }
  gates_n_inf = gates_n_forwardRate_r * (gates_n_forwardRate_r + gates_n_reverseRate_r)^-1
  gates_n_tau = (gates_n_forwardRate_r + gates_n_reverseRate_r)^-1
  gates_n_q' = (gates_n_inf + -1 * gates_n_q) * gates_n_tau^-1
}

BREAKPOINT {
  SOLVE dstate METHOD cnexp
  LOCAL gates_n_fcond, g

  gates_n_fcond = gates_n_q * gates_n_q * gates_n_q * gates_n_q
  g = conductance * gates_n_fcond
  ik = g * (v + -1 * ek)
}

