# Within-pad contrast (train-era wells; test untouched)

wells 4,858 on 1,416 pads with >=2 targeted siblings and real exposure variation

exposure = kernel-weighted draining-neighbour count
  BETWEEN pads (naive, confounded): slope +0.9 pts per unit
  WITHIN pads (rock held fixed):    slope -2.8 pts per unit  [95% CI -9.6 .. +1.8]
  physics predicts negative; confound predicts positive

exposure = kernel-weighted withdrawal
  BETWEEN pads (naive, confounded): slope +1.9 pts per unit
  WITHIN pads (rock held fixed):    slope -5.3 pts per unit  [95% CI -11.4 .. -0.5]
  physics predicts negative; confound predicts positive
