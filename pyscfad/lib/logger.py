# pylint: skip-file
import jax
from pyscf.lib import logger
from pyscf.lib.logger import *
from pyscfad import util

def flush(rec, msg, *args):
    def _flush(*args):
        rec.stdout.write(msg % args)
        rec.stdout.write('\n')
        rec.stdout.flush()

    if any(util.is_tracer(arg) for arg in args):
        jax.debug.callback(_flush, *args)
    else:
        _flush(*args)

def timer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0 is None:
        wall0 = rec._w0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec' % (msg, rec._t0-cpu0))
        return rec._t0

def get_t0(rec):
    return (rec._t0, rec._w0)

# FIXME monkey patch
logger.flush = flush
logger.timer = timer
logger.Logger.timer = timer
logger.Logger.get_t0 = get_t0

