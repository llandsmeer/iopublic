NEURON {
    JUNCTION_PROCESS cx36averaged
    NONSPECIFIC_CURRENT i
    RANGE g
}
INITIAL {}
PARAMETER {
  g = 1
  peer_vmean = -50
}
BREAKPOINT {
    LOCAL v_diff
    v_diff = v - peer_vmean
    i = (g*v_diff * exp( v_diff * v_diff  * (-0.01)))*0.8 + (g*v_diff)*0.2
    if (i != i) {
        i = 0
    }
}
