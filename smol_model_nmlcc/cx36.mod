NEURON {
    JUNCTION_PROCESS cx36
    NONSPECIFIC_CURRENT i
    RANGE g
}
INITIAL {}
PARAMETER {
  g = 1
}
BREAKPOINT {
    LOCAL v_diff
    v_diff = v - v_peer
    i = (g*v_diff * exp( v_diff * v_diff  * (-0.01)))*0.8 + (g*v_diff)*0.2
    if (i != i) {
        i = 0
    }
}
