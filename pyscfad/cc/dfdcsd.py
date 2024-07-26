from pyscfad.cc import dfccsd

class RDCSD(dfccsd.RCCSD):
    @property
    def dcsd(self):
        return True
