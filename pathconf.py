import re, os, sys


PATH_CONF_PATH = "conf/path_default.conf"
PATH_CONF_PATH = os.path.join(os.path.dirname(__file__), PATH_CONF_PATH)


def get_path(key):
    return attributes[key]


class InvalidConfError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, linenum, line, type=0):
        self.linenum = linenum
        self.line = line
        self.type = type

    def __str__(self):
        res = 'Conf file "%s" line %d\n  %s\nInvalidConfError' % (PATH_CONF_PATH, self.linenum, self.line)
        if self.type == 1:
            res += ': path not exist'
        return res


comm_regex = re.compile(r"#.*")
quot_regex = re.compile(r"\"[^\"]*\"")

attributes = {}
with open(PATH_CONF_PATH, 'r') as f:
    for linenum, ol in enumerate(f):
        l = ol.strip()
        l = comm_regex.sub("", l)
        paths = quot_regex.findall(l)
        if len(paths) > 1:
            raise InvalidConfError(linenum + 1, ol, 0)
        if len(l) != 0:
            s = [a.strip() for a in l.split('=')]

            if len(s) != 2:
                raise InvalidConfError(linenum+1, ol,0)
            if len(paths) == 1:
                path = paths[0][1:-1]
            else:
                path = s[1]
            if not os.path.exists(path):
                raise InvalidConfError(linenum + 1, ol, 1)
            attributes[s[0]] = path

