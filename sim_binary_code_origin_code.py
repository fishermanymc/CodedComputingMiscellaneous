import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
EPSILON = 1e-10  # error tolerence due to floating number calculation


def dec_xor(vec1, vec2, factor, dtype):
    # This function updates vec1 by subtracting vec2 * factor from vec1
    # The operation is reduced to binary XOR if dtype is "uint8"
    assert len(vec1) == len(vec2)
    if dtype == 'float64':
        vec1 -= vec2 * factor
        return
    # Otherwise, XOR
    assert dtype == 'uint8'
    # L = len(vec1)
    # for i in range(L):
    #     vec1[i] ^= vec2[i]
    vec1 = np.bitwise_xor(vec1, vec2)


def findPivot(coeff):
    # This function finds the location and value of
    # the first non-zero entry of a vector.
    for i in range(len(coeff)):
        if abs(coeff[i]) > EPSILON:  # use EPSILON instead of 0
            return i, coeff[i]  # pivot location and value
    return -1, 0


def idxFlatten(xList, yList, numCol=1):
    # Given a matrix with numCol columns, we count the entries row-wise, so
    # that each entry has a unique 1D index. E.g., when numCol = 4, we have
    # [0, 1, 2, 3,
    #  4, 5, 6, 7
    # ...........]
    # This function lists the 1D index of all the 2D (x, y) for every pair of
    # x and y listed in xList and yList.
    if max(yList) >= numCol:
        raise ValueError("y index exceeds column width!")
    idx = []
    for x in xList:
        for y in yList:
            idx.append(x * numCol + y)
    return idx


def idxRecover(index1D, numCol=1):
    # The inverse function of indexFlatten function. From 1D index to 2D index.
    xList = list(set([int(np.floor(idx, numCol)) for idx in index1D]))
    yList = list(set([int(np.mod(idx, numCol)) for idx in index1D]))
    return xList, yList


def primes(n):
    # return the prime factorization of an integer n
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def decompose2D(d, numRow=1, numCol=1):
    # Decompose d into d = r * c, such that r <= numRow and c <= numCol.
    # If not possible, reduce d by 1 until solved.
    # Return all the valid combination
    if d > numRow * numCol:
        raise ValueError("input d exceeds numRow * numCol !")
    while d >= 1:
        validComb = []
        for r in range(1, numRow + 1):
            c = d / r
            if np.mod(c, 1) < EPSILON and int(c + EPSILON) <= numCol:
                validComb.append((r, int(c + EPSILON)))
        if len(validComb) > 0:
            return validComb
        d -= 1


class RLNCEncoder:
    # systematic binary random linear network coding with 50% encoding prob.
    # inputs:
    #         k: the number of original tasks
    #         sysPhase: if True, will generate the k original tasks first,
    #                            i.e., will proceed the systematic phase.
    #                   if False, will skip and only generated coded tasks.
    #         dtype: if "uint8", encoding and decoding are under binary field
    #                if "float64", then under the real number field.
    def __init__(self, numRow=1, numCol=1, sysPhase=True, dtype='uint8'):
        self.numRow = numRow
        self.numCol = numCol
        self.k = numRow * numCol
        # Calculate the prob. that each Ai and Bi will be chosen
        self.dimension = int(numRow > 1) + int(numCol > 1)
        self.prob = 0.5 ** (1 / self.dimension) * 100
        if sysPhase:
            self.counter = 0
        else:
            self.counter = self.k
        self.dtype = dtype

    def getCoeff(self):
        coeff = np.zeros(self.k, dtype=self.dtype)
        if self.counter < self.k:  # systematic phase
            coeff[self.counter] = 1
            self.counter += 1
            return coeff
        else:  # coded phase
            while True:
                xList = []  # pick Ai
                yList = []  # pick Bi
                for r in range(self.numRow):
                    if np.random.randint(100) < self.prob:
                        xList.append(r)
                for c in range(self.numCol):
                    if np.random.randint(100) < self.prob:
                        yList.append(c)
                if len(xList) > 0 and len(yList) > 0:  # nonempty selections
                    idx = idxFlatten(xList, yList, self.numCol)
                    coeff = np.zeros(self.k, dtype=self.dtype)
                    for i in idx:
                        coeff[i] = 1
                    self.counter += 1
                    return coeff


class LTEncoder:
    def __init__(self, numRow=1, numCol=1, dtype='uint8'):
        self.numRow = numRow
        self.numCol = numCol
        self.k = numRow * numCol
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


class nonUniRandInt:
    # This function generates random numbers with a nonuniform distribution.
    # The algorithm we use is to decompose the nonuniform distribution into
    # binary distributions. The complexity of the algorithm is constant, i.e.,
    # its complexity does not grow with longer distribution.
    # Main reference:
    # https://oroboro.com/non-uniform-random-numbers/
    def __init__(self, distribution, resolution=100):
        assert abs(sum(distribution) - 1) < EPSILON
        self.resolution = resolution
        self.dis = []
        self.nonZeroNums = []
        for i in range(len(distribution)):
            if distribution[i] > 0:
                self.dis.append(distribution[i])
                self.nonZeroNums.append(i)
        self.disWork = self.dis[:]
        self.lenNonZero = len(self.nonZeroNums)
        self.binaryDis = []
        self.firstChoice = []
        self.secondChoice = []
        thres = 1.0 / (self.lenNonZero - 1)
        for i in range(self.lenNonZero - 1):
            for j in range(self.lenNonZero):
                if self.disWork[j] > 0 and self.disWork[j] <= thres:
                    break
            remain = thres - self.disWork[j]
            self.binaryDis.append(self.disWork[j] / (self.disWork[j] + remain))
            self.disWork[j] = 0
            self.firstChoice.append(self.nonZeroNums[j])
            if remain == 0:  # j occupies the whole thres
                self.secondChoice.append(j)  # it will thus always be chosen
            else:
                for j in range(self.lenNonZero):
                    if self.disWork[j] > 0 and self.disWork[j] >= remain:
                        break
                self.secondChoice.append(self.nonZeroNums[j])
                self.disWork[j] = self.disWork[j] - remain
        self.resolution = self.lenNonZero * 100

    def getNumber(self):
        binIdx = np.random.randint(self.lenNonZero - 1)
        if np.random.randint(self.resolution) < \
           self.binaryDis[binIdx] * self.resolution:
            return self.firstChoice[binIdx]
        return self.secondChoice[binIdx]


class Decoder:
    # General linear decoding class. It checks the usefulness of
    # incoming linear coeff vectors, and indicate the decodability
    # of original tasks.
    # TODO: the actual data decoding function.
    def __init__(self):
        self.initialized = False

    def initialize(self, coeff):
        # meaningfully initialize the decoder after receiving a coeff
        assert np.dtype(coeff[0]) in ['uint8', 'float64']
        self.k = len(coeff)
        self.dtype = np.dtype(coeff[0])
        self.coeffMatrix = np.zeros((self.k, self.k), dtype=self.dtype)
        self.numBasis = 0  # no. of orthogonal basis (useful coeff) received
        self.initialized = True

    def receive(self, coeff):
        if not self.initialized:
            self.initialize(coeff)
        if self.numBasis == self.k:  # we have already received all the k basis
            return False, True  # coeff is useless, the block can decode
        # forward Gaussian elimination
        for i in range(self.k):
            if self.coeffMatrix[i, i] == 1 and coeff[i] > 0:
                dec_xor(coeff, self.coeffMatrix[i], coeff[i], self.dtype)
        pivot, value = findPivot(coeff)
        if value == 0:  # this coeff is linearly dependent of previous basis
            return False, False  # coeff is useless, the block cannot decode
        if self.dtype == 'float64':
            coeff /= value
        # coeff becomes a new basis.
        # Use it to backward Gaussian eliminate other basis
        for i in range(self.k):
            if not self.coeffMatrix[i, pivot] == 0:
                dec_xor(self.coeffMatrix[i], coeff,
                        self.coeffMatrix[i, pivot], self.dtype)
        self.coeffMatrix[pivot] = coeff
        self.numBasis += 1
        if self.numBasis < self.k:
            return True, False  # coeff is usefull, the block cannot decode
        return True, True  # coeff is usefull, the block can decode


def basicTest():
    numRow = 30
    numCol = 30
    k = numRow * numCol  # number of original tasks
    N = k * 10  # number of workers
    # dtype = 'uint8'  # binary operation
    dtype = 'float64'  # real number operation
    # enc = RLNCEncoder2(numRow, numCol, sysPhase=False, dtype=dtype)
    enc = LTEncoder(numRow, numCol, dtype=dtype)
    gMatrix = []  # the generator matrix
    for i in range(N):
        coeff = enc.getCoeff()  # use this coeff for the i-th worker
        # print(coeff)
        gMatrix.append(coeff)
    orderCompletion = np.random.permutation(N)  # random completion order

    dec = Decoder()
    decodable = False  # a flag indicating whether the k tasks can be decoded
    counter = 0
    decMatrix = []
    while not decodable and counter < N:
        # get the coeff of the counter-th completed worker
        coeff = gMatrix[orderCompletion[counter]]
        useful, decodable = dec.receive(coeff)
        counter += 1
        if useful:  # this worker's job is useful, keep it
            decMatrix.append(coeff)
        # By the end of the process, decMatrix will be a k*k full-rank matrix.
        # Each row of decMatrix is a coeff vector.
        # We can verify it (only under "float64") by calculating its inverse.
    assert len(decMatrix) == k
    print("number of useless worker is:", counter - k)


def simulations(k=15, n=22, testNum=1000, dtype='float64'):
    # Log the test parameters
    delta = {}
    delta['k'] = k
    delta['n'] = n
    delta['testNum'] = testNum
    delta['dtype'] = dtype

    # Test RLNC
    print("Testing RLNC")
    # NOTE that if sysPhase=True, then
    # each of the first k workers will only receive 1 units of information,
    # and each of the remaining n - k workers will receive
    # k/2 units of information on average.
    enc = RLNCEncoder(numRow=k, numCol=1, sysPhase=True, dtype=dtype)
    delta['RLNC'] = np.zeros(testNum)
    for i in range(testNum):
        print(i)
        gMatrix = [enc.getCoeff() for j in range(n)]
        order = np.random.permutation(n)
        dec = Decoder()
        decodable = False
        count = 0
        recCoeff = []
        while (not decodable and count < n):
            coeff = gMatrix[order[count]]
            useful, decodable = dec.receive(coeff.copy())
            count += 1
            if useful:
                recCoeff.append(coeff)
        if decodable:  # evaluate the rank of the coeff matrix, should be full.
            assert np.linalg.matrix_rank(recCoeff) == k
        delta['RLNC'][i] = count - k

    # Test LT code
    print("Testing LT")
    # With LT code, each worker will receive log(k) units of information
    # on average. There is no systematic part.
    enc = LTEncoder(numRow=k, numCol=1, dtype=dtype)
    delta['LT'] = np.zeros(testNum)
    for i in range(testNum):
        print(i)
        gMatrix = [enc.getCoeff() for j in range(n)]
        order = np.random.permutation(n)
        dec = Decoder()
        decodable = False
        count = 0
        recCoeff = []
        while (not decodable and count < n):
            coeff = gMatrix[order[count]]
            useful, decodable = dec.receive(coeff.copy())
            count += 1
            if useful:
                recCoeff.append(coeff)
        if decodable:  # evaluate the rank of the coeff matrix, should be full.
            assert np.linalg.matrix_rank(recCoeff) == k
        delta['LT'][i] = count - k

    print("RLNC code average delta:", np.mean(delta['RLNC']))
    print("LT code average delta:", np.mean(delta['LT']))

    # Save the result
    with open("results" + str(k) + "_" + str(n) + ".pickle", 'wb') as handle:
        pickle.dump(delta, handle)

    # Read the result
    with open("results" + str(k) + "_" + str(n) + ".pickle", 'rb') as handle:
        delta = pickle.load(handle)

    # Plot the CDF
    cdf = np.linspace(0, 100, delta['testNum'])
    dRLNC = np.sort(delta['RLNC'])  # sort in descending order
    dLT = np.sort(delta['LT'])  # sort in descending order
    plt.plot(dRLNC, cdf, linewidth=3, label='RLNC code')
    plt.plot(dLT, cdf, linewidth=3, label='LT code')
    plt.xlim((0, delta['n'] - delta['k']))
    plt.xlabel("Number of redundant workers ($\delta$)", fontsize=20)
    plt.ylabel("CDF (%)", fontsize=20)
    plt.title('Redundancy CDF of RLNC and LT code', fontsize=20)
    plt.legend(loc='center right', fontsize=16)
    plt.grid()
    plt.show()


simulations(k=15, n=22, testNum=100, dtype='float64')
