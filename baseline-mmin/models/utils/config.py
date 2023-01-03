import sys

class OptConfig(object):
    def __init__(self):
        pass

    def load(self, config_dict):
        if sys.version > '3':
            for key, value in config_dict.items():
                if not isinstance(value, dict):
                    setattr(self, key, value)
                else:
                    self.load(value)
        else:
            for key, value in config_dict.iteritems():
                if not isinstance(value, dict):
                    setattr(self, key, value)
                else:
                    self.load(value)