"""Ordered process mapping for evaluation jobs."""
import os


def pmap(fn, jobs):
    import multiprocessing as mp
    jobs = list(jobs)
    procs = int(os.environ.get("PG_PROCS", os.cpu_count() or 1))
    if procs <= 1 or len(jobs) <= 1:
        return [fn(*job) for job in jobs]
    with mp.get_context("fork").Pool(min(procs, len(jobs))) as pool:
        return pool.starmap(fn, jobs, chunksize=1)
