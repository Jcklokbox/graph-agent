import multiprocessing as mp
import os
import sys
import time
from eval.predictive import run_one
from harness import journal


def episode(task):
    world, seed, steps, wset, agent = task
    start = time.monotonic()
    _, _, out = run_one(world, [agent, 'marginal'], seed, steps, wset)
    loss = {k: v['ll'] for k, v in out.items()}
    gc = (loss['marginal'] - loss[agent]) / (loss['marginal'] - loss['filter'])
    return {'k': [os.environ['PG_TREE'], seed, f'{agent}|{wset}|{world}|{steps}'],
            'm': {'gc': gc, 'll': loss, 'sec': out[agent]['sec']}, 'dt': time.monotonic() - start}


if __name__ == '__main__':
    wset, steps, name, path, *pairs = sys.argv[1:]
    todo = [(k, int(i), int(steps), wset, name) for k, i in (x.split(':') for x in pairs)]
    with mp.get_context('fork').Pool(2) as pool:
        for rec in pool.imap(episode, todo, chunksize=1):
            journal.append(path, rec)
            print(rec['k'][1:], rec['m']['gc'], round(rec['dt'], 2), flush=True)
