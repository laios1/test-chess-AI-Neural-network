

def get_data(trainingSize = 200,EvaluationSize = 0):
    training_set = []
    evaluation_set = []
    with open("database/chessData.csv","r") as dataFile : 
        dataFile.readline()
        for i in range(trainingSize):
            training_set.append(dataFile.readline()[:-1].split(","))
        for j in range(EvaluationSize):
            evaluation_set.append(dataFile.readline()[:-1].split(","))
    return training_set,evaluation_set


a,b = get_data(10000)
for i in range(1):
    print(a[i])
import chess.engine

def centipawn_to_proba2(strCenti): #prend en entree ce qu'il y a en 2eme colonne
    if strCenti[0] == '#' :
        if strCenti[1] == '+' :
            return 1
        else : #strCenti[1] == '-'
            return 0
    else :
        if strCenti[0] == '+' :
            return 1/(1+10**(-int(strCenti[1:])/400))
        else : #strCenti[0] == '-'
            return 1/(1+10**(-int(strCenti)/400))
"""
for i in range(10000):
    print(a[i][1])
    print(centipawn_to_proba2(a[i][1]))
    print()
    """
""" 
def stockfish_evaluation(board, time_limit = 0.01):
    engine = chess.engine.SimpleEngine.popen_uci("Stockfish-sf_16/src/stockfish")
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    return result['score']

board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
result = stockfish_evaluation(board)
print(result)
board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
result = stockfish_evaluation(board)
print(result) """