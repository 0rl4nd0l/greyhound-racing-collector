# Use one coherent ordered finish forecast

The canonical model will produce one probability distribution over possible finishing orders rather than independent win, place, and exotic-market predictions. Win, top-N, exacta, trifecta, and runner-ranking outputs will be derived from that distribution so they cannot contradict one another; the first implementation should use a Plackett–Luce-style sequential model behind a replaceable forecast contract.
