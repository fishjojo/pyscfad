# pylint: skip-file
from pyscf.lib import logger
from pyscf.lib.logger import *
from pyscfad import ops

def flush(rec, msg, *args):
    args_list = []
    for arg in args:
        if ops.is_array(arg):
            arg = ops.to_numpy(arg)
        args_list.append(getattr(arg, 'val', arg))
    rec.stdout.write(msg % tuple(args_list))
    rec.stdout.write('\n')
    rec.stdout.flush()

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

logger.flush = flush
logger.timer = timer
logger.Logger.timer = timer
