
import numpy as np 
import json


###################################################Neural Network#################################################################
def print_graphe_graphviz(SSize,w,b,X,results) : 
    to_print = "digraph G {\nnode [shape=record,width=.1,height=.1];\n"
    k = 0
    to_print = to_print + "subgraph cluster0 {\n"
    for i in range(len(X)):
        to_print = to_print + """node%d [label = "{<n>%d|%.2f|a<p>}"];\n""" % (k,i,X[i])
        k += 1
    to_print = to_print + "}\n"

    for i in range(len(SSize)-1) :
        to_print = to_print + "subgraph cluster%d {\n" % (i+1)
        for j in range(SSize[i+1]):
            to_print = to_print + """node%d [label = "{<n>%d/%d|%.2f|%.2f|a<p>}"];\n""" % (k,i,j,b[i][j],results[i][j])
            k += 1
        to_print = to_print + "}\n"
    
    totSSize = 0
    for i in range(len(SSize)-1) :
        for j in range(SSize[i]):
            for l in range(SSize[i+1]):
                to_print = to_print + """node%d -> node%d [label = %.2f];\n""" % (j+totSSize,l+totSSize+SSize[i],w[i][j][l])
        totSSize += SSize[i]
    to_print = to_print+ "}"
    return to_print

class neural_network(object):
    """
    crée un réseau de neurone : 

    SSize = [#input_layer, #hidden_layer_1,..., #hidden_layer_n, #output_layer]

    /!\ peut ne pas avoir d'hidden layer

    LR(Learning rate) ; LR = 1 par default 

    act (activation function) ; act = "sigmoid" par defaut ; (pour l'instant seulement sigmoid and ReLU)
    """
    def __init__(self,SSize,LR = 1,act = "sigmoid"):
        self.Size = SSize
        self.ActivationType = act

        self.w = [] #les poids/ taille : (nombre_de_couches-1)*(taille de la couche i)*(taille de la couche i+1)
        self.bias = [] #les correctifs d'érreurs / taille : (nb_couches-1)*(taille de la couche i)
        for i in range(len(self.Size)-1) : 
            self.w.append(np.random.randn(self.Size[i],self.Size[i+1])) #on met les poids en random 
            self.bias.append(np.random.randn(self.Size[i+1])) #on assigne les correctifs de chaques neurones 
        #self.bias.append(np.random.randn(self.Size[-1])) #on assigne les correctifs de la derniere couche 
        
        
        self.learningRate = LR 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []
        self.dError  = []


    #-------------------------------fonctions d'activations--------------------------------------

    def identite(self,s):
        return np.array(s)

    def identiteDerive(self,s):
        return np.array([1 for i in s])

    def ReLU(self,s):
        return np.array([i if i >= 0 else 0.1*i for i in s ])

    def ReLUDerive(self,s):
        return np.array([1 if i >= 0 else 0.1 for i in s ])

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidDerive(self,s):
        return self.sigmoid(s)*(1-self.sigmoid(s))

    def activation(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoid(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLU(s)
        elif self.ActivationType == "id" :
            return self.identite(s)

    def activationDerive(self,s):
        if self.ActivationType == "sigmoid" :
            return self.sigmoidDerive(s)
        elif self.ActivationType == "ReLU" :
            return self.ReLUDerive(s)
        elif self.ActivationType == "id" :
            return self.identiteDerive(s)
    
    
    #-------------------evaluation de la fonction reseau de neurones en X--------------------------------
    def forward(self,X):
        #self.preAct = 0 
        self.preAct = [] # valeur de la preactivation (combinaison linéaire des poids et des valeurs)
        self.results = []

        self.preAct.append(np.dot(X,self.w[0]) + self.bias[0]) #produit matricielle 
        self.results.append(self.activation(self.preAct[0])) #activation du neurone (lissage des données)
        
        for i in range(1,len(self.Size)-1) : #on itere pour chaque neurone 
            self.preAct.append(np.dot(self.results[i-1],self.w[i]) + self.bias[i])
            #print("preact : ",self.preAct[i])
            self.results[i:] = [self.activation(self.preAct[i])]

        return self.results[-1]


    #---------------------------Entrainement--------------------------------------

    def backward(self,X,y,o):
        """
        X : valeur d'entree 

        y : valeur attendu par le modele 

        o : valeur retourne par le model pour une entree donne
        """
        self.dError = [] #calcul de l'erreur 


        list.insert(self.dError,0, 2*(o-y) * self.activationDerive(self.preAct[-1]))
        for i in range(1,len(self.Size)-1) : 
            #print(self.sigmoidDerive(self.preAct[-(i+1)]))
            list.insert(self.dError,0, np.dot(self.dError[0],self.w[-i].T) * self.activationDerive(self.preAct[-(i+1)]))

        #print(np.array(X,ndmin=2),np.array(self.dError[0],ndmin=2))
        self.w[0] -= np.dot(np.array(X,ndmin=2).T, np.array(self.dError[0],ndmin=2))*self.learningRate
        self.bias[0] -= self.dError[0]*self.learningRate
        

        for i in range(1,len(self.Size)-1) : 
            self.w[i] -= np.dot(np.array(self.results[i-1],ndmin=2).T, np.array(self.dError[i],ndmin=2))*self.learningRate
            self.bias[i] -= self.dError[i]*self.learningRate
        

    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)
        return np.abs(o-y)


    def error(self,X,y):
        o = self.forward(X)
        return (o-y)**2


    def predict(self,xIncunue):
        print(xIncunue)
        print(tuple(self.forward(xIncunue)))



    #--------------------------sauvegarder, charger et afficher le reseau de Neurones-----------------------------
    def print_NN(self,X):
        o = self.forward(X)
        print(print_graphe_graphviz(self.Size,self.w,self.bias,X,self.results))

    def save_NN(self, name = "Neural_Network_save"):
        with open(name+'.json', 'w', encoding='utf-8') as f:
            
            w2 = [self.w[i].tolist() for i in range(len(self.w))]
            b2 = [self.bias[i].tolist() for i in range(len(self.bias))]
            activation_f = self.ActivationType
            dic = {"weight" : w2, "bias" : b2,"act_f" : activation_f}
            json.dump(dic, f, ensure_ascii=False, indent=4)  

    def load_NN(self,name):
        with open(name+'.json') as f:
            data_loaded = json.load(f)
            self.w = [np.array(data_loaded["weight"][i]) for i in range(len(data_loaded["weight"]))]
            self.bias = [np.array(data_loaded["bias"][i]) for i in range(len(data_loaded["bias"]))]
            self.ActivationType = data_loaded["act_f"]

def print_shape(array):
    for i in range(len(array)):
        print(array[i].shape)



####################################################################################################################################
import chess
import itertools
"""
board = chess.Board()
print(list(board.legal_moves),"\n")
a = []
for mv in board.legal_moves :
    a.append(mv.to_square)
print(a)

move = chess.Move(chess.E2,chess.E4)
print(move)

king_square_index = board.pieces(chess.ROOK,chess.BLACK)

print(king_square_index)
print(list(king_square_index))
"""



"""TODO : 
1. lire le jeu OK
2. savoir quels coup sont légaux OK
3. jouer un coup ~
4. savoir quand il y a mat 
"""



def read_board(board):
    """
    nomenclature : 

    blancs : 
        rois
        reinnes 
        tours 
        fous 
        cavaliers 
        pions

    Noirs : 
        rois
        reinnes 
        tours 
        fous 
        cavaliers
        pions
    """
    entree = []
    #blancs
    #roi
    wk = board.pieces(chess.KING,chess.WHITE)
    wk = list(wk)
    wkb = [0 for i in range(64)]
    for k in wk : 
        wkb[k] = 1

    entree.append(wkb)

    #reine 
    wq = list(board.pieces(chess.QUEEN,chess.WHITE))
    wqb = [0 for i in range(64)]
    for q in wq : 
        wqb[q] = 1

    entree.append(wqb)

    #tours 
    wr = list(board.pieces(chess.ROOK,chess.WHITE))
    wrb = [0 for i in range(64)]
    for r in wr : 
        wrb[r] = 1

    entree.append(wrb)

    #fous 
    wb = list(board.pieces(chess.BISHOP,chess.WHITE))
    wbb = [0 for i in range(64)]
    for b in wb : 
        wbb[b] = 1

    entree.append(wbb)

    #cavalier 
    wkn = list(board.pieces(chess.KNIGHT,chess.WHITE))
    wknb = [0 for i in range(64)]
    for k in wkn : 
        wknb[k] = 1

    entree.append(wknb)

    #pions
    wp = list(board.pieces(chess.PAWN,chess.WHITE))
    wpb = [0 for i in range(64)]
    for k in wp : 
        wpb[k] = 1

    entree.append(wpb)


    #noirs
    #roi
    wk = board.pieces(chess.KING,chess.BLACK)
    wk = list(wk)
    wkb = [0 for i in range(64)]
    for k in wk : 
        wkb[k] = 1

    entree.append(wkb)

    #reine 
    wq = list(board.pieces(chess.QUEEN,chess.BLACK))
    wqb = [0 for i in range(64)]
    for q in wq : 
        wqb[q] = 1

    entree.append(wqb)

    #tours 
    wr = list(board.pieces(chess.ROOK,chess.BLACK))
    wrb = [0 for i in range(64)]
    for r in wr : 
        wrb[r] = 1

    entree.append(wrb)

    #fous 
    wb = list(board.pieces(chess.BISHOP,chess.BLACK))
    wbb = [0 for i in range(64)]
    for b in wb : 
        wbb[b] = 1

    entree.append(wbb)

    #cavalier 
    wkn = list(board.pieces(chess.KNIGHT,chess.BLACK))
    wknb = [0 for i in range(64)]
    for k in wkn : 
        wknb[k] = 1

    entree.append(wknb)

    #pions
    wp = list(board.pieces(chess.PAWN,chess.BLACK))
    wpb = [0 for i in range(64)]
    for k in wp : 
        wpb[k] = 1

    entree.append(wpb)



    #on lui donne aussi c'est au tours de qui 
    if board.turn == chess.WHITE :
        entree.append([1,0])
    else :
        entree.append([0,1])

    return sum(entree,[])

# print(len(read_board(board)))
# board = chess.Board("3b1q1q/1N2PRQ1/rR3KBr/B4PP1/2Pk1r1b/1P2P1N1/2P2P2/8 b - -")

# print(board.result())


def evaluate(NN,board,depth,maxDepth):
    if depth == maxDepth :
        #print(board)
        #print(NN.forward(read_board(board)))
        return NN.forward(read_board(board))[0]
    else :

        if board.turn == chess.WHITE :
            rating = []
            for mv in board.legal_moves :
                board.push(mv)
                rating.append(evaluate(NN,board,depth+1,maxDepth))
                board.pop()
            return max(rating)
        else : 
            rating = []
            for mv in board.legal_moves :
                board.push(mv)
                rating.append(evaluate(NN,board,depth+1,maxDepth))
                board.pop()
            return min(rating)


def choose_move(NN,board,maxDepth = 1):
    if board.turn == chess.WHITE :
        rating = []
        legal = []
        for mv in board.legal_moves :
            board.push(mv)
            legal.append(mv)
            rating.append(evaluate(NN,board,1,maxDepth))
            board.pop()
        return legal[np.argmax(rating)]
    else : 
        rating = []
        legal = []
        for mv in board.legal_moves :
            board.push(mv)
            legal.append(mv)
            rating.append(evaluate(NN,board,1,maxDepth))
            board.pop()
        return legal[np.argmin(rating)]
    
    



""" 
import chess.engine

def stockfish_evaluation(board, time_limit = 0.01):
    engine = chess.engine.SimpleEngine.popen_uci("Stockfish-sf_16/src/stockfish")
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    return result['score']

board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
result = stockfish_evaluation(board)
print(result)
 """


def get_data(trainingSize = 1000,EvaluationSize = 0,from_line = 1):
    training_set = []
    evaluation_set = []
    with open("database/chessData.csv","r") as dataFile : 
        for i in range(from_line):
            dataFile.readline()
        for i in range(trainingSize):
            training_set.append(dataFile.readline()[:-1].split(","))
        for j in range(EvaluationSize):
            evaluation_set.append(dataFile.readline()[:-1].split(","))
    return training_set,evaluation_set




def centipawn_to_proba(centi):
    return 1/(1+10**(-centi/400))

def proba_to_centipawn(proba):
    """
    /!\ une proba c'est entre 0 et 1 
    """
    return 400*np.log10(proba/(1-proba))


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














import matplotlib.pyplot as plt


#TODO:
#automatiser l'entrainement 
def train_AI1(nb_train_set,nb_eval_set,nb_train,dim_NN = [770,1000,1000,1], lr = 0.001,continue_training = False,from_line = 1):
    board = chess.Board()
    NN = neural_network(dim_NN,lr)
    index = []
    cost_list = []
    training_set,evaluation_set = get_data(nb_train_set,nb_eval_set,from_line)
    if continue_training :
        NN.load_NN("Neural_Network_save")
        for j in range(nb_train):
            for i in range(nb_train_set):
                if ((i+j*(nb_train_set))*100)%(nb_train_set*nb_train) == 0 :
                    print(((i+j*(nb_train_set))*100)/(nb_train_set*nb_train),"%")
                    fen,value = training_set[i]
                    board.set_fen(fen)
                    index.append(i+len(training_set)*j)
                    cost_list.append(NN.train(read_board(board),centipawn_to_proba2(value)))
                else :
                    fen,value = training_set[i]
                    board.set_fen(fen)
                    NN.train(read_board(board),centipawn_to_proba2(value))
    else :
        for j in range(nb_train):
            for i in range(nb_train_set):
                if ((i+j*(nb_train_set))*100)%(nb_train_set*nb_train) == 0 :
                    print(((i+j*(nb_train_set))*100)/(nb_train_set*nb_train),"%")

                if(i == 100):
                    fen,value = training_set[i]
                    board.set_fen(fen)
                    index.append(i+len(training_set)*j)
                    cost_list.append(NN.train(read_board(board),centipawn_to_proba2(value)))
                else :
                    fen,value = training_set[i]
                    board.set_fen(fen)
                    NN.train(read_board(board),centipawn_to_proba2(value))

    print(cost_list,"\n")
    plt.plot(index,cost_list)
    plt.show()
    NN.save_NN()
    
    print("the end ?")
    return training_set,evaluation_set,NN

#créer une interface pour jouer 

if __name__ == '__main__':
    
    #t_s,e_s,NN = train_AI1(2000,0,200,[770,600,400,200,1],0.005,False,2000)

    
    board = chess.Board()
    NN = neural_network([770,600,400,200,1],0.01)
    NN.load_NN("Neural_Network_save2")
    

    #for i in range (10): 
    #    print(choose_move(NN,board,2))
    #    board.push(choose_move(NN,board,1))
    #    print(board)
    #    print(choose_move(NN,board,2),"\n")
    #    board.push(choose_move(NN,board,1))
    
    
    
    index = []
    cost_list = []
    random_list = []
    training_set,evaluation_set = get_data(0,1000,20000)
    for i in range(len(evaluation_set)) :
        fen,value = evaluation_set[i]
        board.set_fen(fen)
        index.append(i)
        cost_list.append(NN.error(read_board(board),centipawn_to_proba2(value)))
        random_list.append((np.random.random()-centipawn_to_proba2(value))**2)
    
    plt.ylim(0, 1)
    print(cost_list,"\n")
    m_coup = np.mean(cost_list)
    m_coup_l  = [m_coup for i in index]
    plt.plot(index,cost_list)
    plt.plot(index,m_coup_l)
    plt.show()



    plt.ylim(0, 1)
    m_coup = np.mean(random_list)
    m_coup_l  = [m_coup for i in index]
    plt.plot(index,random_list)
    plt.plot(index,m_coup_l)
    plt.show()
    
    #training 
    
    
    #tests
    """
    board = chess.Board()
    #print(len(read_board(board)))
    NN = neural_network([768,1],1)
    print(NN.forward(read_board(board)))
    print(choose_move(NN,board,1))
    """
