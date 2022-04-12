NEURON {
  SUFFIX cal
  USEION ca WRITE ica READ eca
  RANGE conductance
}

PARAMETER {
  conductance = 0.00001 (uS)
}

STATE { gates_k_q gates_l_q }

INITIAL {
  LOCAL gates_l_steadyState_x, gates_k_steadyState_x

  gates_l_steadyState_x = (1 + exp(0.11764705882352941 * (85.5 + v)))^-1
  gates_k_steadyState_x = (1 + exp(-0.23809524890787256 * (61 + v)))^-1
  gates_k_q = gates_k_steadyState_x
  gates_l_q = gates_l_steadyState_x
}

DERIVATIVE dstate {
  LOCAL gates_l_steadyState_x, gates_l_timeCourse_t, gates_k_steadyState_x

  gates_l_steadyState_x = (1 + exp(0.11764705882352941 * (85.5 + v)))^-1
  gates_l_timeCourse_t = 35 + 20 * (1 + exp(0.13698629779067634 * (84 + v)))^-1 * exp(0.03333333333333333 * (160 + v))
  gates_k_steadyState_x = (1 + exp(-0.23809524890787256 * (61 + v)))^-1
  gates_k_q' = gates_k_steadyState_x + -1 * gates_k_q
  gates_l_q' = (gates_l_steadyState_x + -1 * gates_l_q) * gates_l_timeCourse_t^-1
}

BREAKPOINT {
  SOLVE dstate METHOD cnexp
  LOCAL gates_k_fcond, fopen0, g

  gates_k_fcond = gates_k_q * gates_k_q * gates_k_q
  fopen0 = gates_k_fcond * gates_l_q
  g = conductance * fopen0
  ica = g * (v + -1 * eca)
}

