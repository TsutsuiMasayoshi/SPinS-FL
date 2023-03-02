import matplotlib.pyplot as plt
FLOAT_SIZE = 4 # Bytes

class TrafficTracker: # number is Byte
    def __init__(self) -> None:
        self.sentCom_round = []
        self.loadedCom_round = []

    def send(self, paramsNum: int):
        self.sentCom_round.append(paramsNum * FLOAT_SIZE)

    def load(self, paramsNum: int):
        self.loadedCom_round.append(paramsNum * FLOAT_SIZE)

    def plot(self):
        rounds = list(range(len(self.sentCom_round)))
        comm_per_round = [self.sentCom_round[i] + self.loadedCom_round[i]
                            for i in range(len(self.sentCom_round))]
        plt.plot(rounds, comm_per_round)
        plt.title('Communication vs Rounds')
        plt.xlabel('Rounds')
        plt.ylabel('Communication')
        plt.savefig('commLog/FLCommunication.png')