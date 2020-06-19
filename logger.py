"""============================================================================
Logger that logs to file or prints, depending on configuration.
============================================================================"""

import logging


# -----------------------------------------------------------------------------

class Logger:

    def __init__(self, directory):
        """Initialize logger with handler.
        """
        handler = logging.FileHandler(f'{directory}/out.txt')
        self.logger = logging.getLogger('logger.main')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.directory = directory
        self.log('=' * 80)

    def log(self, msg, with_hline=False):
        """Log or print depending on context.
        """
        if with_hline:
            self.log_hline()
        self.logger.info(msg)
        print(msg)

    def log_pair(self, key, val):
        """Log or print key-value pair depending on context.
        """
        key = key_formatter(key)
        if is_number(val):
            val = format_number(val)
        self.log(f'{key}: {val}')

    def log_hline(self, bold=False):
        """Print horizontal line.
        """
        mark = '=' if bold else '-'
        self.log(mark * 80, with_hline=False)

    def log_args(self, args):
        """Print arguments passed to script.
        """
        self.log('SCRIPT ARGS')
        self.log_hline()
        fields = [f for f in vars(args)]
        longest = max(fields, key=len)
        format_str = '{:>%s}  {:}' % len(longest)
        for f in fields:
            msg = format_str.format(f, getattr(args, f))
            self.log(msg)


# -----------------------------------------------------------------------------

def key_formatter(value, length=11):
    """Return string with fixed length `length`.
    """
    assert(length > 3)
    if len(value) > length:
        value = list(value)[:length]
        value[length-3:] = '...'
        return ''.join(value).encode('utf-8')
    return f'{value: <{length}}'


def format_number(value):
    """Print number with rounding and commas every three digits.
    """
    return f'{value:.6f}'


def is_number(val):
    """Return True if val is number.
    """
    try:
        float(val)
        return True
    except TypeError:
        return False
