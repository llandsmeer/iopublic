NEURON {
    JUNCTION_PROCESS cx36temp
    NONSPECIFIC_CURRENT i
    GLOBAL stopAfter
    GLOBAL enableAfter
    RANGE g
}
INITIAL {
    totalTime = 0
    rates()
}
PARAMETER {
    g = 1
    stopAfter = 0
    enableAfter = 100000000
}
ASSIGNED {
    rate_totalTime
}
STATE {
    totalTime ? ms
}
BREAKPOINT {
    SOLVE states METHOD cnexp
    if (totalTime > stopAfter && totalTime < enableAfter) { ? please give us events
        i = 0
    } else {
        i = (g*(v - v_peer) * exp((v - v_peer) * (v - v_peer) * (-0.01)))*0.8 + (g*(v - v_peer))*0.2
    }
}
DERIVATIVE states {
    rates()
    totalTime' = rate_totalTime
}

PROCEDURE rates() {
    rate_totalTime = 1
}
