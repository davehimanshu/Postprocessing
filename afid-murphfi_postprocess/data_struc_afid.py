class data_struc_fields:
    def __init__(self):

        # define variables to store this class
        self._temp = None
        self._vx = None
        self._vy = None
        self._vz = None
        self._phi = None
        self._tempr = None
        self.cord_info = {}

    #-----------------------------------------------------------------#
    # Define the coordinate information as a dictionary
    #-----------------------------------------------------------------#
    def set_cord_info(self, cord_info):
        for key, value in cord_info.items():
            setattr(self, key, value)
            self.cord_info[key] = value
    def __getattr__(self, name):
        if name in self.cord_info:
            return self.cord_info[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    # Define the getter, setter and deleter for the class variables
    #-----------------------------------------------------------------#   
    # phase field variable (phi)
    @property
    def phi(self):
        return self._phi   
    @phi.setter
    def phi(self, value):
        self._phi = value
    @phi.deleter
    def phi(self):
        del self._phi
    # temperature variable (temp)
    @property
    def temp(self):
        return self._temp
    @temp.setter
    def temp(self, value):
        self._temp = value
    @temp.deleter
    def temp(self):
        del self._temp
    # temperature variable (tempr)
    @property
    def tempr(self):
        return self._tempr
    @tempr.setter
    def tempr(self, value):
        self._tempr = value
    @tempr.deleter
    def tempr(self):
        del self._tempr
    # velocity variables (vx, vy, vz)
    @property
    def vx(self):
        return self._vx
    @vx.setter
    def vx(self, value):
        self._vx = value
    @vx.deleter
    def vx(self):
        del self._vx
    @property
    def vy(self):
        return self._vy
    @vy.setter
    def vy(self, value):
        self._vy = value
    @vy.deleter
    def vy(self):
        del self._vy
    @property
    def vz(self):
        return self._vz
    @vz.setter
    def vz(self, value):
        self._vz = value
    @vz.deleter
    def vz(self):
        del self._vz
    #-----------------------------------------------------------------#