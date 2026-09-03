# Counterfactual summary (Phase 10)

wells scored: 45,511   in training domain: 10,810
isolated wells' context penalty (must be ~0): max |0.0000| pts

context (depletion) penalty, pts of peer-P50 pace — in-domain wells:
    1-3 nbrs: n=   458  median  -23.2  p25/p75 -53.1/-4.8  negative share 83%
    4-8 nbrs: n=   936  median  -49.4  p25/p75 -105.0/-21.1  negative share 93%
   9-12 nbrs: n=   828  median  -66.1  p25/p75 -118.3/-34.6  negative share 94%
    >12 nbrs: n= 8,115  median  -93.5  p25/p75 -150.0/-53.5  negative share 96%

MARGINAL depletion penalty (withdrawal history erased, bag kept) — in-domain:
    1-3 nbrs: n=   458  median   +0.0 pts  p25/p75 +0.0/+3.2  negative share 17%
    4-8 nbrs: n=   936  median   +0.0 pts  p25/p75 -1.4/+5.1  negative share 40%
   9-12 nbrs: n=   828  median   +0.0 pts  p25/p75 -3.1/+5.6  negative share 47%
    >12 nbrs: n= 8,115  median   -0.0 pts  p25/p75 -4.6/+3.9  negative share 51%
  in volume terms: median 0 boe/yr per in-domain well

within-pad proximity penalty (deviation model, pad-relative pts):
  in-domain median +16.73  p25/p75 +7.26/+31.35

READ ME FIRST: penalty_ctx prices the neighbourhood's withdrawal
history (the identified channel), NOT bare proximity. Negative values
are printed, not hidden: they mark residual good-rock confounding in
levels. penalty_prox is small by honest necessity. No number without
in_training_domain=True should ever be quoted.
