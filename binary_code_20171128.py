import numpy as np
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
    L = len(vec1)
    for i in range(L):
        vec1[i] ^= vec2[i]


def findPivot(coeff):
    # This function finds the location and value of
    # the first non-zero entry of a vector.
    for i in range(len(coeff)):
        # use EPS instead of 0 to aviod floating point residuals
        if abs(coeff[i]) > EPSILON:
            return i, coeff[i]  # pivot location and value
    return -1, 0


class RLNCEncoder:
    # systematic binary random linear network coding with 50% encoding prob.
    # inputs:
    #         k: the number of original tasks
    #         sysPhase: if True, will generate the k original tasks first,
    #                            i.e., will proceed the systematic phase.
    #                   if False, will skip and only generated coded tasks.
    #         dtype: if "uint8", encoding and decoding are under binary field
    #                if "float64", then under the real number field.
    def __init__(self, k=5, sysPhase=True, dtype='uint8'):
        self.k = k
        if sysPhase:
            self.counter = 0
        else:
            self.counter = k
        self.dtype = dtype

    def getCoeff(self):
        coeff = np.zeros(self.k, dtype=self.dtype)
        if self.counter < self.k:
            coeff[self.counter] = 1  # systematic phase
        else:  # coded phase
            zeroCoeff = True
            while zeroCoeff:
                for i in range(self.k):
                    # each original task has 50% prob. of being encoded
                    coeff[i] = np.random.randint(0, 2) % 2
                    if coeff[i] == 1:
                        zeroCoeff = False
        self.counter += 1
        return coeff


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


if __name__ == '__main__':
    # np.random.seed(1351)
    k = 12  # number of original tasks
    N = 22  # number of workers
    # dtype = 'uint8'  # binary operation
    dtype = 'float64'  # real number operation
    enc = RLNCEncoder(k, sysPhase=True, dtype=dtype)
    gMatrix = []  # the generator matrix
    for i in range(N):
        coeff = enc.getCoeff()  # use this coeff for the i-th worker
        gMatrix.append(coeff)
    print(np.array(gMatrix).T)
    print(np.array(gMatrix).sum(axis=1))
    acc = 0
    testNum = 10000
    for i in range(testNum):
        orderCompletion = np.random.permutation(N)  # random order of worker completion
        dec = Decoder()
        decodable = False  # a flag indicating whether the k tasks can be decoded
        counter = 0
        decMatrix = []
        while not decodable:
            # get the coeff of the counter-th completed worker
            coeff = gMatrix[orderCompletion[counter]]
            useful, decodable = dec.receive(coeff.copy())
            counter += 1
            if useful:  # this worker's job is useful, keep it
                decMatrix.append(coeff)
                # By the end of the process, decMatrix will be a k*k full-rank matrix.
                # Each row of decMatrix is a coeff vector.
                # We can verify it (only under "float64") by calculating its inverse.
        assert len(decMatrix) == k
        decMatrix = np.array(decMatrix).T
        invMatrix = np.linalg.inv(decMatrix)
        #print("number of useless worker is:", counter - k)
        acc += (counter - k)
    print(acc / testNum)
