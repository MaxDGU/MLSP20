import pprint

def createMasterArray(Arr, B, n):
    C = list(); # C = []
    for i in range(n):
        C.extend(list(Arr[i] * B[i]));
    return C;

def padArr(Arr, lenSeq):
    n = len(Arr);
    numZeroToPad = max(lenSeq - n, 0);
    Arr.extend(numZeroToPad * [['0']])
    return Arr;

def convertToOutputList(Arr):
    # input =  Arr = [[aaa][b][cccc]]
    # output = [[a,3],[b,1],[c,4]]
    return [[e[0], len(e)] for e in Arr];

def createX(wind, maxReps):
    if (len(wind) == 0):
        return [];
    X_n = list();
    curr_list = [wind[0]]
    for curr in wind[1:]:
        print(curr_list)
        if ((curr_list[-1] == curr) and (len(curr_list) < maxReps)):
            curr_list.append(curr)
        else:
            X_n.append(curr_list)
            curr_list = [curr]
    if (curr_list):
        X_n.append(curr_list)

    return X_n

def getNumberOfRemainingElements(idx, e, C):
    count = 1
    for i in range(idx, len(C)):
        if C[i] != e:
            break;
        else:
            count += 1
    return count;
    
def createY(idx, lenPred, maxReps, C):
    Y_n = list()
    if (len(C[idx:idx+lenPred]) == 0):
        return Y_n
    Y_n = createX(C[idx:idx+lenPred], maxReps)
    lastY = Y_n[-1]
    
    if idx+lenPred >= len(C):
        return Y_n
    nextC = C[idx+lenPred]
    if nextC != lastY[0]:
        return Y_n
   
    # C has more of the same elements as lastY[0]
    lastY = Y_n.pop(-1)
    rem = getNumberOfRemainingElements(idx+lenPred, lastY[0], C)
    appendRemaining = createX((rem+len(lastY))*lastY[0], maxReps);
    return Y_n + appendRemaining;
        

def saveSetGroupBy(Arr, B, maxReps, lenSeq, lenPred): 
    C = createMasterArray(Arr, B, len(Arr));
    X = list();
    Y = list();
    for i in range(len(C)):
        X_n = createX(C[i:i+lenSeq], maxReps);
        Y_n = createY(i+lenPred, lenPred, maxReps, C);
            
        if (X_n and Y_n):
            X.append(padArr(convertToOutputList(X_n), lenSeq));
            Y.append(padArr(convertToOutputList(Y_n), lenPred));
          
    return X, Y;


def expandElements(X, Y): #enumerate out and expand all subsets of X and Y
    for i in X:
        chord = X[i][0] #string in list of lists (of lists?)
        reps = X[i][1] #int in list of lists
        for i in reps:
            X.append([chord, i])



def main():
    Arr = ['A', 'B', 'C', 'D', 'E'];
    B = [3, 4, 5, 6, 7];
    # Arr, B, maxReps, lenSeq, lenPred
    tests = [
        [['A', 'B', 'C', 'D', 'E'], [3, 4, 5, 6, 7], 4, 8, 8],
        [['A', 'B', 'C',], [6, 4, 5], 4, 8, 10],
        [['A', 'B'], [4,5], 4, 8, 10],
        [['A'], [9], 2, 8, 8],
        [[], [3], 2, 4, 4]
    ];
    for idx, t in enumerate(tests):
        print('\n\nrunning test #', idx+1);
        X, Y = saveSetGroupBy(t[0], t[1], t[2], t[3], t[4]);
        pprint.pprint(X);
        pprint.pprint(Y);
        
    print('finished running tests');
    return;
    
main();





