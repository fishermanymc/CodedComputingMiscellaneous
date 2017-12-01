import cProfile
import binary_code as bc


def testCode(dType):
    numRow = 30
    numCol = 30
    enc = bc.RLNCEncoder2(numRow, numCol, sysPhase=False, dtype=dType)
    dec = bc.Decoder()
    decodable = False
    while not decodable:
        useful, decodable = dec.receive(enc.getCoeff())


cProfile.run('testCode("uint8")')
