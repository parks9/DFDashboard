import numpy as np


__all__ = ['flat_field_cosine_model', 'CosineFlatField']



def flat_field_cosine_model(x, bias, lightlevel, xc, A, w1, B, w2):
    term_1 = A * np.cos((x - xc) / w1)**2
    term_2 = B * np.cos((x - xc) / w2)**4
    return bias + lightlevel * (term_1 + term_2)


class ProbModel(object):

    def __init__(self, x, y, y_err):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.y_err = np.asarray(y_err)

    def ln_likelihood(self, theta):
        raise NotImplementedError()

    def ln_prior(self, theta):
        raise NotImplementedError()

    def ln_posterior(self, theta):
        lnp = self.ln_prior(theta)

        if np.isinf(lnp):
            return lnp

        lnL = self.ln_likelihood(theta)
        lnprob = lnp + lnL

        if np.isnan(lnprob):
            return -np.inf

        return lnprob

    def __call__(self, theta):
        return self.ln_posterior(theta)


class CosineFlatField(ProbModel):

    def model(self, x, theta):
        return flat_field_cosine_model(x, *theta)

    def ln_likelihood(self, theta):
        obs = self.y
        exp = self.model(self.x, theta)
        return -0.5 * np.sum((obs - exp)**2 / self.y_err**2)

    def ln_prior(self, theta):
        return 0.0
