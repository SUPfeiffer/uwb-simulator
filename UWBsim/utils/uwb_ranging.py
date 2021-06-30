import random
import numpy as np
import scipy.optimize
import scipy.special
import math
import yaml
from enum import IntEnum

#import UWBsim
from UWBsim.utils import dataTypes

class RangingType(IntEnum):
    NONE = 0
    TWR = 1
    TDOA = 2
    

class RangingSource(IntEnum):
    NONE = 0
    LOG = 1
    GENERATE_GAUSS = 2
    GENERATE_HT_CAUCHY = 3
    GENERATE_HT_GAMMA = 4

class RangingParams(yaml.YAMLObject):
    """Parameter Structure for uwb ranging

    Structure for passing and saving parameters used for generating,
    uwb ranging data, specifically how and what data is generated or
    read from the logs (if available). Inherits from YAMLObject to 
    allow saving and loading parameters from yaml files.

    Attributes:
        yaml_tag: tag for yaml representation
    """

    yaml_tag = u'!RangingParams'
    def __init__(self, source=int(RangingSource.GENERATE_HT_CAUCHY), rtype=int(RangingType.TWR), 
                    anchor_positions=[[0.0,0.0,0.0]], anchor_enable=[False], 
                    interval=0.01, gauss_sigma=0.05, htc_gamma=0.1, htc_ratio=1.0,
                    htg_mu=0.1, htg_lambda=3.5, htg_k=2, htg_scale=1, outlier_chance=0.0):
        """ Initializes RangingParams

        Args:
            source: One of the sources specified in the RangingSource Enum Class:
                    0=NONE, 1=LOG, 2=GENERATE_GAUSS, 3=GENERATE_HT_CAUCHY = 3
                    4=GENERATE_HT_GAMMA
            rtype: Ranging type (specified in RangingType Enum): 0=NONE,
                    1=TWR, 2=TDOA
            anchor_positions: list with positions of all the uwb anchors (index in
                    list corresponds to anchor id)
            anchor_enable: list of booleans, (true = anchor at same index
                    in anchor_positions is enabled)
            interval: When measurements are generated, average time between measurements
            gauss_sigma: Standard deviation for gaussian parts of the noise models
            htc_gamma: Gamma parameter for Cauchy distribution part of the Cauchy-based
                    heavy-tailed noise model
            htg_mu: Mean for the Gamma-based heavy-tailed noise model
            htg_lambda: lambda parameter for the Gamma distribution
            htg_scale: Factor of heavy-tailedness of the Gamma-based heavy-tailed
                    noise model (0:Gaussian, >0: Heavy-tailed)
            outlier_chance: additional chance for generating outliers
        """
        self.source = source
        self.rtype = rtype
        self.anchor_positions = anchor_positions
        self.anchor_enable = anchor_enable
        
        self.interval = interval
        
        # Gaussian Params
        self.gauss_sigma = gauss_sigma

        # Noise is modeled as combination of a Gaussian (x<0) and 
        # a Cauchy distribution (x>0)
        self.htc_gamma = htc_gamma
        self.htc_ratio = htc_ratio

        # Noise is modelled as the sum of a gaussian, 
        # and a Gamma Distribution:
        # https://ieeexplore-ieee-org.tudelft.idm.oclc.org/abstract/document/7891540?casa_token=TDHYZfl0pg4AAAAA:_obI3xcZ4xprTWSCtgvRuWssHtSfgd3BgFi_kn7-QrnMjf6B66190djtuyRYkb_XSHq4Awzl0fc
        self.htg_mu = htg_mu
        self.htg_lambda = htg_lambda
        self.htg_k = htg_k
        self.htg_scale = htg_scale

        # Outlier Chance
        self.outlier_chance = outlier_chance

    def __repr__(self):
        return "%s(source=%d, rtype=%d, anchor_positions=%r, anchor_enable=%r, interval=%r, \
            gauss_sigma=%r, htc_gamma=%r, htc_ratio=%r, htg_mu=%r, htg_lambda=%r, htg_k=%r, htg_scale=%r, \
            outlier_chance=%r)" % (
                self.__class__.__name__, self.source, self.rtype, self.anchor_positions, self.anchor_enable,
                self.interval, self.gauss_sigma, self.htc_gamma, self.htc_ratio, self.htg_mu, self.htg_lambda, self.htg_k,
                self.htg_scale, self.outlier_chance
            )


class UWBGenerator:
    def __init__(self, params: RangingParams):
        self.params = params

        self.anchor_position = params.anchor_positions
        self.anchor_enable = params.anchor_enable
        if self.anchor_enable is not None:
            self.N_anchors = len(self.anchor_enable)
        else:
            self.N_anchors = 0
        self.interval = params.interval
        self.interval_diff = 0.1*self.interval # add randomness to intervals(between 10% longer and 10% shorter)
        # Gauss
        self.gauss_sigma = params.gauss_sigma

        # Cauchy
        self.cauchy_gamma = params.htc_gamma
        self.cauchy_ratio = params.htc_ratio
        self.cauchy_alpha = (2*math.pi*self.cauchy_gamma) / (math.sqrt(2*math.pi*self.gauss_sigma**2) + math.pi*self.cauchy_gamma)

        # Gamma
        self.gamma_mu = params.htg_mu
        self.gamma_lambda = params.htg_lambda
        self.gamma_k = params.htg_k
        self.gamma_scale = params.htg_scale

        self.outlier_chance = params.outlier_chance

        # next_meas is used to determine when an anchor should generate its next measurement
        if params.rtype == RangingType.TWR:
            self.next_meas = np.zeros(self.N_anchors)
            for i in range(self.N_anchors):
                self.next_meas[i] += self.interval + np.random.uniform(-self.interval_diff, self.interval_diff)
        else:  #params.rtype == RangingType.TDOA:
            self.next_meas = np.zeros((self.N_anchors,self.N_anchors))
            for i in range(self.N_anchors):
                for j in range(self.N_anchors):
                    self.next_meas[i][j] += self.interval + np.random.uniform(-self.interval_diff, self.interval_diff)



    def generate_twr(self, position, anchor_id, time):
        if (self.anchor_enable[anchor_id]) and (time >= self.next_meas[anchor_id]):
            self.next_meas[anchor_id] += self.interval + np.random.uniform(-self.interval_diff, self.interval_diff)
            
            a = self.anchor_position[anchor_id]
            dx = position[0] - a[0]
            dy = position[1] - a[1]
            dz = position[2] - a[2]
            
            d = math.sqrt(dx*dx + dy*dy + dz*dz) + self.noise()
            return dataTypes.TWR_meas(a, anchor_id, d, self.gauss_sigma, time)
        else:
            return None

    def generate_tdoa(self, position, anchor0_id, anchor1_id, time):
        if (self.anchor_enable[anchor0_id]) and (self.anchor_enable[anchor1_id]) and (time >= self.next_meas[anchor0_id][anchor1_id]):
            self.next_meas[anchor0_id][anchor1_id] += self.interval + np.random.uniform(-self.interval_diff, self.interval_diff)
            
            a0 = self.anchor_position[anchor0_id]
            a1 = self.anchor_position[anchor1_id]
            
            dx0 = position[0] - a0[0]
            dy0 = position[1] - a0[1]
            dz0 = position[2] - a0[2]
            dx1 = position[0] - a1[0]
            dy1 = position[1] - a1[1]
            dz1 = position[2] - a1[2]
            
            d0 = math.sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0)
            d1 = math.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            dd = d1-d0 + self.noise()
            return dataTypes.TDOA_meas(a0, a1, anchor0_id, anchor1_id, 
                                            dd, self.gauss_sigma, time)
        else:
            return None
    
    def noise(self):
        if self.params.source == RangingSource.GENERATE_GAUSS:
            return np.random.normal(loc=0.0, scale=self.params.gauss_sigma)
        
        elif self.params.source == RangingSource.GENERATE_HT_CAUCHY:
            if self.params.rtype == RangingType.TWR:
                CDF_limit = (2-self.cauchy_alpha)*0.5
                tmp = random.random()
                if tmp < CDF_limit:
                    # Gaussian CDF: F(x) = (1/2) * (1 + erf( (x-mu)/sqrt(2*sig^2) ))
                    return math.sqrt(2*self.gauss_sigma**2)*scipy.special.erfinv(2*tmp/(2-self.cauchy_alpha)-1) #pylint: disable=no-member
                else:
                    # Cauchy CDF: F(x) = (1/pi) * arctan((x-x0)/gamma) + 1/2
                    # Because of scaling with alpha (Area under CDF no longer 1), need to also start from -inf and then flip
                    tmp = 1-tmp
                    return -self.cauchy_gamma*math.tan(math.pi*(tmp/self.cauchy_alpha-0.5))
            elif self.params.rtype == RangingType.TDOA:
                # Sum of gaussian and cauchy
                tmp = random.random()
                res = scipy.optimize.fsolve(self._ht_cauchy_cdf, 0, args=[tmp], xtol=0.00001)
                return res[0]

        elif (self.params.source == RangingSource.GENERATE_HT_GAMMA):
            tmp = random.random()
            res = scipy.optimize.fsolve(self._ht_gamma_cdf, 0, args=[tmp], xtol=0.00001)
            return res[0]
        else:
            return 0

    def _ht_gamma_cdf(self, x, *args):
        tmp = 0.5*(1+scipy.special.erf( (x-self.gamma_mu)/(math.sqrt(2)*self.gauss_sigma)))/(1+self.gamma_scale)
        if x>0:
            tmp += scipy.special.gammainc(self.gamma_k, self.gamma_lambda*x)* self.gamma_scale/(1+self.gamma_scale)
        return tmp - args[0]

    def _ht_cauchy_cdf(self, x, *args):
        gauss = 0.5*(1+scipy.special.erf( x/(math.sqrt(2)*self.gauss_sigma))) * (1-self.cauchy_ratio)
        cauchy = ((1/np.pi) * math.atan(x/self.cauchy_gamma) + 0.5) * self.cauchy_ratio
        return gauss + cauchy - args[0]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params = RangingParams()
    params.source = RangingSource.GENERATE_HT_CAUCHY
    params.rtype = RangingType.TDOA
    params.gauss_sigma = 0.25
    params.htc_gamma = 0.4
    params.htc_ratio = 1
    generator = UWBGenerator(params)

    noise = []
    out = 0
    for i in range(100000):
        tmp = generator.noise()
        if tmp > 2:
            out += 1
        elif tmp < -2:
            out += 1
        else:
            noise.append(tmp)
    print("outliers: {:.2f}%".format(100*out/100000))
    plt.hist(noise, bins=1000)
    plt.show()