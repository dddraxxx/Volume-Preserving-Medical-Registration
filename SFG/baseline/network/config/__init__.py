class Reader:
    def __init__(self, obj, full_attr=""):
        self._object = obj
        self._full_attr = full_attr
    
    def __getattr__(self, name):
        if self._object is None:
            ret = None
        else:
            ret = self._object.get(name, None)
        return Reader(ret, self._full_attr + '.' + name)

    def get(self, default=None):
        if self._object is None:
            # print in green
            print('\033[92m', end='')
            print('Default FLAGS.{} to {}'.format(self._full_attr, default))
            print('\033[0m', end='')
            return default
        else:
            return self._object

    @property
    def value(self):
        return self._object