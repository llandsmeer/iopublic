NEURON {
  SUFFIX ca_conc
  USEION ca READ ica WRITE cai
  RANGE initialConcentration
}

PARAMETER {
  diam
  initialConcentration = 0 (mM)
}

STATE { cai }

INITIAL {
  cai = initialConcentration
}

DERIVATIVE dstate {
  LOCAL surfaceArea, effectiveRadius, eqshellDepth, innerRadius, shellVolume

  surfaceArea = 0.7853975296020508 * diam * diam
  effectiveRadius = 0.25 * diam
  eqshellDepth = 0.0010000000474974513 + -0.0000010000000949949049 * effectiveRadius^-1
  innerRadius = effectiveRadius + -1 * eqshellDepth
  shellVolume = -4.1887868245442705 * innerRadius * innerRadius * innerRadius + 4.1887868245442705 * effectiveRadius * effectiveRadius * effectiveRadius
  cai' = -0.030000001144409223 * cai + -0.009999999776482582 * ica * surfaceArea * (0.19297059375 * shellVolume)^-1
}

BREAKPOINT {
  SOLVE dstate METHOD cnexp
  if (cai < 0) {
    cai = 0
  }

}

