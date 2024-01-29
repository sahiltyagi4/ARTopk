import numpy as np
from scipy.optimize import curve_fit

from pymoo.core.problem import ElementwiseProblem


class ModelMultiObjectiveOptimization(ElementwiseProblem):
    def __init__(self, candidateCRs, comm_times, compress_gains):
        # lower and upper bounds on cr are min. and max. CR respectively
        super().__init__(n_var=1, n_obj=2, xl=np.array([min(candidateCRs)]), xu=np.array([max(candidateCRs)]))
        self.candidate_crs = candidateCRs
        self.comm_times = comm_times
        self.compress_gains = compress_gains

    def fitCommunicationTimes(self):
        # model communication cost as an exponential function of CR
        def exponential_fn(x, v1, v2):
            y = v2 * np.exp(v1 * x) - 1
            return y

        x2 = np.array(list(self.candidate_crs))
        y2 = np.array(list(self.comm_times))
        params, _ = curve_fit(exponential_fn, x2, y2)
        v1_fit, v2_fit = params
        return v1_fit, v2_fit

    def fitCompressionGains(self):
        # model compression gain as a hyperbolic tangent function of CR
        def hyperbolic_tan(x, v1, v2, v3, v4, v5, v6):
            y = (np.exp(v1 * x) - np.exp(v2 * -1 * x) + v5) / (np.exp(v3 * x) + np.exp(v4 * -1 * x) + v6)
            return y

        temp_gains = [g for g in self.compress_gains]
        temp_crs = [cr for cr in self.candidate_crs]
        temp_gains.append(1.0)
        temp_crs.append(1.0)
        temp_gains.append(0.0)
        temp_crs.append(0.0)
        x3 = np.array(temp_crs)
        y3 = np.array(temp_gains)
        params, _ = curve_fit(hyperbolic_tan, x3, y3)
        v1, v2, v3, v4, v5, v6 = params
        return v1, v2, v3, v4, v5, v6

    def _evaluate(self, x, out, *args, **kwargs):
        v1_commtime, v2_commtime = self.fitCommunicationTimes()
        f1 = v2_commtime * np.exp(v1_commtime * x) - 1

        v1_gain, v2_gain, v3_gain, v4_gain, v5_gain, v6_gain = self.fitCompressionGains()
        f2 = (np.exp(v3_gain * x) + np.exp(v4_gain * -1 * x) + v6_gain) / (np.exp(v1_gain * x) - np.exp(v2_gain * -1 * x) + v5_gain)
        out['F'] = [f1, f2]