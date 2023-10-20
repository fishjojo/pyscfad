import sys
from functools import wraps
from pyscf.lib import logger
from pyscf.lib.logger import (
    DEBUG1        as DEBUG1,
    DEBUG         as DEBUG,
    INFO          as INFO,
    NOTE          as NOTE,
    NOTICE        as NOTICE,
    WARN          as WARN,
    WARNING       as WARNING,
    ERR           as ERR,
    ERROR         as ERROR,
    QUIET         as QUIET,
    TIMER_LEVEL   as TIMER_LEVEL,
    process_clock as process_clock,
    perf_counter  as perf_counter,
)
from pyscfad import ops

def flush(rec, msg, *args):
    arg_list = []
    for arg in args:
        if ops.is_tensor(arg):
            arg = ops.convert_to_numpy(arg)
        arg_list.append(arg)
    logger.flush(rec, msg, *arg_list)

def log(rec, msg, *args):
    if rec.verbose > QUIET:
        flush(rec, msg, *args)

def error(rec, msg, *args):
    if rec.verbose >= ERROR:
        flush(rec, '\nERROR: '+msg+'\n', *args)
    #sys.stderr.write('ERROR: ' + (msg%args) + '\n')

def warn(rec, msg, *args):
    if rec.verbose >= WARN:
        flush(rec, '\nWARN: '+msg+'\n', *args)
        #if rec.stdout is not sys.stdout:
        #    sys.stderr.write('WARN: ' + (msg%args) + '\n')

def info(rec, msg, *args):
    if rec.verbose >= INFO:
        flush(rec, msg, *args)

def note(rec, msg, *args):
    if rec.verbose >= NOTICE:
        flush(rec, msg, *args)

def debug(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)

def debug1(rec, msg, *args):
    if rec.verbose >= DEBUG1:
        flush(rec, msg, *args)

def timer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0 is None:
        wall0 = rec._w0
    rec._t0, rec._w0 = process_clock(), perf_counter()
    if rec.verbose >= TIMER_LEVEL:
        flush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
              % (msg, rec._t0-cpu0, rec._w0-wall0))
    return rec._t0, rec._w0

class Logger(logger.Logger):
    log = log
    error = error
    warn = warn
    note = note
    info = info
    debug  = debug
    debug1 = debug1
    timer = timer

@wraps(logger.new_logger)
def new_logger(rec=None, verbose=None):
    if isinstance(verbose, Logger):
        log = verbose
    elif isinstance(verbose, int):
        if getattr(rec, 'stdout', None):
            log = Logger(rec.stdout, verbose)
        else:
            log = Logger(sys.stdout, verbose)
    else:
        log = Logger(rec.stdout, rec.verbose)
    return log
