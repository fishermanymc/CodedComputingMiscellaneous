class LTEncoder:
    def __init__(self, k, dtype='uint8'):
        self.k = k
        self.dtype = dtype
        self.density = self.densityCalculation()
        self.dGenerator = nonUniRandInt(self.density)
        self.idxList = list(range(self.k))

    def densityCalculation(self):
        '''
        When LT code is used, the number of data packets XORed in each
        coded packet follows a special distribution. This function initilze
        a random number generator according to this distribution.
        '''
        mu = np.zeros(self.k)  # discrete density distribution
        c = 0.04  # free parameter between 0 and 1
        delta = 0.0001  # free parameter between 0 and 1
        S = c * np.log(self.k / delta) * np.sqrt(self.k)
        sum_density = 0.0
        for d in range(1, self.k + 1):
            # decide the default density rho
            rho = 0.0
            if d == 1:
                rho = 1 / self.k
            else:
                rho = 1 / d / (d - 1)
            # decide the adjustment density tau
            tau = 0
            if d <= self.k / S - 1:
                tau = S / self.k / d
            else:
                if d <= self.k / S:
                    tau = S * np.log(S / delta) / self.k
            # final density value is the sum of rho and tau
            mu[d - 1] = rho + tau
            sum_density += mu[d - 1]
        # normalize the density so that it sums up to 1
        mu /= sum_density
        return mu

    def getCoeff(self):
        coeff = np.zeros(self.k, dtype=self.dtype)
        d = self.dGenerator.getNumber() + 1
        # idx = random.sample(self.idxList, d)
        validComb = decompose2D(d, self.numRow, self.numCol)
        r, c = validComb[np.random.randint(len(validComb))]
        xList = random.sample(list(range(self.numRow)), r)
        yList = random.sample(list(range(self.numCol)), c)
        idx = idxFlatten(xList, yList, self.numCol)
        for i in idx:
            coeff[i] = 1
        return coeff
