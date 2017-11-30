import numpy as np
import binary_code_20171128 as bc


# m = np.array([[0., 1., 0., 1., 0.],
#               [1., 1., 0., 0., 0.],
#               [0., 1., 1., 0., 1.],
#               [1., 0., 1., 1., 1.],
#               [0., 0., 1., 0., 1.]], dtype='float64')

# # m = np.array([[0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
# #               [1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1.],
# #               [0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1.],
# #               [1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.],
# #               [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
# #               [1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
# #               [0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0.],
# #               [0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.],
# #               [1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0.],
# #               [0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0.],
# #               [1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.],
# #               [0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1.]],
# #              dtype='float64')

# m = m.transpose()
# # print(m)

# print(np.linalg.matrix_rank(m))
# dec = bc.Decoder()
# for mm in m:
#     print(mm)
#     useful, decodable = dec.receive(mm)
#     print(dec.coeffMatrix)
#     # print(dec.coeffMatrix)
#     # print(str(useful), str(decodable))
#     # print(dec.coeffMatrix)
# print(dec.receivedCoeff)

k = 5
enc = bc.RLNCEncoder(k)