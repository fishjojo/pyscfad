# pylint: skip-file
import re
import jax
from pyscf.lib import logger
from pyscf.lib.logger import *
from pyscfad import util

def _partial_eval_msg(msg, args):
    format_specifier = re.compile(r'%(?:\d+\$)?[#0\-+ ]?(?:\d+)?(?:\.\d+)?[hlL]?[a-zA-Z]')
    matches = list(format_specifier.finditer(msg))
    partially_evaluated_msg = ''
    remaining_args = list(args)
    tracer_args = []
    last_end = 0

    for match in matches:
        start, end = match.span()
        spec = match.group(0)
        partially_evaluated_msg += msg[last_end:start]

        if remaining_args:
            arg = remaining_args.pop(0)
            if not util.is_tracer(arg):
                partially_evaluated_msg += spec % arg
            else:
                partially_evaluated_msg += spec
                tracer_args.append(arg)
        else:
            partially_evaluated_msg += spec

        last_end = end

    partially_evaluated_msg += msg[last_end:]
    return partially_evaluated_msg, tracer_args

def flush(rec, msg, *args):
    msg, args = _partial_eval_msg(msg, args)

    def _flush(*args):
        rec.stdout.write(msg % args)
        rec.stdout.write('\n')
        rec.stdout.flush()

    if args:
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

