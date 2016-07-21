from numpy import dot


class Node(object):
    parents = None
    children = None
    plates = None
    name = ''

    def __init__(self, parents, name='', plates=None):
        self.parents = parents
        self.name = name
        self.plates = plates


class EFDistribution(object):
    """
    Exponential Distribution Class
    """
    f = None
    g = None
    phi = None
    u = None


    def __init__(self):
        """
        logpdf  = u(x) * phi + g(phi) + f(x)
        :return:
        """

    # check whether this method should be static
    # check phi and u should have the same dimensions
    def get_logpdf_given_x(self, phi):
        """
        u and f are set before
        :param phi:
        :return:
        """
        self.phi = phi.copy()
        self.compute_g()
        ll = self.f + self.g + dot(self.u.T, self.phi)
        return ll

    def get_logpdf_given_phi(self, x):
        """
        phi and g are set before
        :param x:
        :return:
        """
        self.compute_u(x)
        self.compute_f()
        ll = self.f + self.g + dot(self.u.T, self.phi)
        return ll

    def get_logpdf(self, x, phi):

        self.phi = phi.copy()
        self.compute_g()

        self.u = self.compute_u(x)
        self.compute_f()

        ll = self.f + self.g + dot(self.u.T, self.phi)
        return ll

    def set_x(self, x):

        self.u = self.compute_u(x)
        self.compute_f()

    def set_phi(self, phi, convert=False):

        if convert:
            phi = self.convert_to_natural_parameter(phi)
        self.phi = phi.copy()
        self.compute_g()

    def compute_g(self):
        raise NotImplementedError("compute_g not implemented for "
                                  "%s"
                                  % (self.__class__.__name__))

    def compute_f(self):
        raise NotImplementedError("compute_f not implemented for "
                                  "%s"
                                  % (self.__class__.__name__))

    def compute_u(self, x):
        raise NotImplementedError("compute_u not implemented for "
                                  "%s"
                                  % (self.__class__.__name__))

    def convert_to_natural_parameter(self):
        raise NotImplementedError("convert_to_natural_parameter "
                          "%s"
                          % (self.__class__.__name__))

    def convert_to_natural_parameter(self, p):
        raise NotImplementedError("convert_to_natural_parameter "
                          "%s"
                          % (self.__class__.__name__))
