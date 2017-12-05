import binary_code as bc
import numpy as np
from matplotlib import pyplot as plt
import pickle
import time


def singleTest(numRow, numCol):
    k = numRow * numCol  # number of original tasks
    dtype = 'uint8'  # binary operationration
    # dtype = 'float64'  # real number operation
    # print('rlnc')
    enc1 = bc.RLNCEncoder(numRow, numCol, sysPhase=True, dtype=dtype)
    dec1 = bc.Decoder()
    decodable = False  # a flag indicating whether the k tasks can be decoded
    counter1 = 0
    n = k * 3
    order = np.random.permutation(n)
    coeffs = [enc1.getCoeff() for i in range(n)] 
    while not decodable:
        useful, decodable = dec1.receive(coeffs[counter1])
        counter1 += 1
    # print('lt')
    enc2 = bc.LTEncoder(numRow, numCol, dtype=dtype)
    dec2 = bc.Decoder()
    decodable = False  # a flag indicating whether the k tasks can be decoded
    counter2 = 0
    while not decodable:
        useful, decodable = dec2.receive(enc2.getCoeff())
        counter2 += 1
    return counter1 - k, counter2 - k


def simulations(testNum):
    kTest = range(100, 1001, 100)
    validComb = {}
    for k in kTest:
        validComb[k] = bc.decompose2D(k, k, k)
        print(validComb[k])
    print("initializaed")

    redun = {}
    redun['rlnc'] = {}
    redun['lt'] = {}
    start = time.time()
    for k in kTest:
        print(k)
        redun['rlnc'][k] = np.zeros(testNum)
        redun['lt'][k] = np.zeros(testNum)
        combs = validComb[k]
        lenCombs = len(combs)
        for t in range(testNum):
            print(t)
            numRow, numCol = combs[np.random.randint(0, lenCombs)]
            r1, r2 = singleTest(numRow, numCol)
            redun['rlnc'][k][t] = r1
            redun['lt'][k][t] = r2
        print(np.mean(redun['rlnc'][k]), np.mean(redun['lt'][k]))
    print("duration: ", time.time() - start)

    #############################################
    # numRow = range(5, 41, 5)
    # redun = {}
    # redun['rlnc'] = {}
    # redun['lt'] = {}
    # testNum = 10
    # start = time.time()
    # for r in numRow:
    #     redun['rlnc'][r ** 2] = []
    #     redun['lt'][r ** 2] = []
    #     print(r ** 2)
    #     for t in range(testNum):
    #         print(t)
    #         redun1, redun2 = singleTest(r, r)
    #         redun['rlnc'][r ** 2].append(redun1)
    #         redun['lt'][r ** 2].append(redun2)
    # print("duration: ", time.time() - start)
    #############################################

    redun['kList'] = kTest
    with open('20171204_binary.pickle', 'wb') as handle:
        pickle.dump(redun, handle)

    assert False
    kTest = range(100, 1001, 100)
    with open('./results/20171129_float.pickle', 'rb') as handle:
        redun1 = pickle.load(handle)
    with open('20171204_binary.pickle', 'rb') as handle:
        redun2 = pickle.load(handle)
    # with open('20171129_binary.pickle', 'rb') as handle:
    #     redun2 = pickle.load(handle)
    plt.plot(redun1['kList'], [(np.mean(redun1['rlnc'][k]) * 50 + np.mean(redun1['rlnc'][k]) * 10) / 60 for k in redun1['kList']],
             label='RLNC - real', linewidth=3)
    # plt.plot(redun2['kList'], [np.mean(redun2['rlnc'][k]) for k in redun2['kList']],
    #          label='RLNC - binary', linewidth=3)
    plt.plot(redun1['kList'], [(np.mean(redun1['lt'][k]) * 50 + np.mean(redun1['rlnc'][k]) * 10) / 60 for k in redun1['kList']],
             label='LT code - real', linewidth=3)
    # plt.plot(redun2['kList'], [np.mean(redun2['lt'][k]) for k in redun2['kList']],
    #          label='LT code - binary', linewidth=3)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("$K=s*t$", fontsize=18)
    plt.ylabel('Average number of extra workers $\delta$', fontsize=18)
    plt.title("Synthesis Binary RLNC and LT codes over $\mathbb{R}$\n" +
              "can both asymptotically achieve the recovery threshold $K$ " +
              "as $K+\delta$", fontsize=18)
    plt.xlim(0, 1000)
    plt.ylim(0, 10)
    plt.grid()
    plt.show()

testNum = 10
# simulation(testNum)
# numRow = range(5, 41, 5)
# with open('r1.pickle', 'rb') as handle:
#     redun = pickle.load(handle)

# for key in ['rlnc', 'lt']:
#     plt.plot(redun['kList'], [np.mean(redun[key][k]) for k in redun['kList']],
#              label=key)
# plt.legend(loc='best')
# plt.xlabel("$r^2$")
# plt.ylabel('average number of extra workers')
# plt.grid()
# plt.show()
# with open('r1.pickle', 'rb') as handle:
#     redun = pickle.load(handle)
# # print(redun[1225])
# numRow = range(5, 40, 5)
# kList = [r ** 2 for r in numRow[1:]]
# plt.plot(kList, [np.mean(redun[k]) / k for k in kList])
# plt.show()
