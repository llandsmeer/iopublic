NEURON {
  SUFFIX cah
  USEION ca WRITE ica READ eca
  RANGE conductance
}

PARAMETER {
  conductance = 0.00001 (uS)
}

STATE { gates_r_q }

INITIAL {
  LOCAL gates_r_forwardRate_r, gates_r_reverseRate_x, gates_r_reverseRate_r, gates_r_inf

  gates_r_forwardRate_r = 1.7000000476837158 * (1 + exp(-0.07194244801754432 * (-5 + v)))^-1
  gates_r_reverseRate_x = -0.2 * (8.5 + v)
  if (gates_r_reverseRate_x != 0) {
    gates_r_reverseRate_r = 0.10000000149011612 * gates_r_reverseRate_x * (1 + -1 * exp(-1 * gates_r_reverseRate_x))^-1
  } else {
    if (gates_r_reverseRate_x == 0) {
      gates_r_reverseRate_r = 0.10000000149011612
    } else {
      gates_r_reverseRate_r = 0
    }
  }
  gates_r_inf = gates_r_forwardRate_r * (gates_r_forwardRate_r + gates_r_reverseRate_r)^-1
  gates_r_q = gates_r_inf
}

DERIVATIVE dstate {
  LOCAL gates_r_forwardRate_r, gates_r_reverseRate_x, gates_r_reverseRate_r, gates_r_inf, gates_r_tau

  gates_r_forwardRate_r = 1.7000000476837158 * (1 + exp(-0.07194244801754432 * (-5 + v)))^-1
  gates_r_reverseRate_x = -0.2 * (8.5 + v)
  if (gates_r_reverseRate_x != 0) {
    gates_r_reverseRate_r = 0.10000000149011612 * gates_r_reverseRate_x * (1 + -1 * exp(-1 * gates_r_reverseRate_x))^-1
  } else {
    if (gates_r_reverseRate_x == 0) {
      gates_r_reverseRate_r = 0.10000000149011612
    } else {
      gates_r_reverseRate_r = 0
    }
  }
  gates_r_inf = gates_r_forwardRate_r * (gates_r_forwardRate_r + gates_r_reverseRate_r)^-1
  gates_r_tau = (0.20000000298023224 * (gates_r_forwardRate_r + gates_r_reverseRate_r))^-1
  gates_r_q' = (gates_r_inf + -1 * gates_r_q) * gates_r_tau^-1
}

BREAKPOINT {
  SOLVE dstate METHOD cnexp
  LOCAL gates_r_fcond, g

  gates_r_fcond = gates_r_q * gates_r_q
  g = conductance * gates_r_fcond
  ica = g * (v + -1 * eca)
}

