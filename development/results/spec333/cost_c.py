"""One pinned-core seed-0 cost pair, warmed without timing compilation."""
import json
import os
import sys
import time
from eval.predictive import run_one
from harness import journal
from models.factor import CANONICAL, make_factor

if __name__ == '__main__':
    m = make_factor(CANONICAL, 5, 2, 4)
    x = [0., 0., 1., 0., 0.]
    for k in range(7):
        y = [float(k % 2), 0., 1., 0., 0.]
        m.predict(x, 0)
        m.learn(x, 0, y)
        x = y
    results = {}
    for world in ('s-2x3', 's-4x3'):
        t = time.monotonic()
        _, seed, out = run_one(world, [CANONICAL], 0, 600, 'dscale')
        val = out[CANONICAL]['sec']
        results[world] = val
        journal.append(sys.argv[1] + 'l', {'k': [os.environ['PG_TREE'], seed, f'{CANONICAL}|C|{world}|600'],
                         'm': val, 'dt': time.monotonic() - t})
    results['ratio'] = results['s-4x3'] / results['s-2x3']
    with open(sys.argv[1], 'w') as f:
        json.dump(results, f)
        f.write('\n')
    print('C', results, flush=True)
