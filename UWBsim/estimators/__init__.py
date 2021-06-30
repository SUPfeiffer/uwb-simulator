import yaml

from UWBsim.estimators.mhe import MHE_Params
from UWBsim.estimators.ekf import EKF_Params


class EstimatorParams(yaml.YAMLObject):
    """Parameter Structure for the estimators

    Structure for passing and saving parameters used by the estimators. 
    Inherits from YAMLObject to allow saving and loading parameters from yaml files.
    
    Attributes:
        yaml_tag: tag for yaml representation
    """

    yaml_tag = u'!EstimatorParams'
    def __init__(self, mhe=MHE_Params(), ekf=EKF_Params()):
        """ Initializes EstimatorParams

        Collection of the parameter structures for the different estimators
        Args:
            mhe: Parameters for the Numerical Position MHE
            ekf: Parameters for the EKF
        """

        self.mhe = mhe
        self.ekf = ekf
        
    def __repr__(self):
        return "%s(mhe=%r, ekf=%r)" % (
                self.__class__.__name__, self.mhe, self.ekf)