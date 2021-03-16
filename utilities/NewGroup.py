"""
Created on Tue June 30 3:49:20 2020

"""

import pprint
from utilities import dataImport, chordUtil
import torch 


chordList = [
    'A:maj'  ,
    'B:maj' , 
    'C:maj' ,
    'D:maj' , 
    'E:maj' ,
    'F:maj' ,
    'G:maj' ,
    'A:min'  ,
    'B:min' ,
    'C:min'  ,
    'D:min'  ,
    'E:min'  ,
    'F:min'  ,
    'G:min'  ,
    'A#:maj'  ,
    'B#:maj'  ,
    'C#:maj' ,
    'D#:maj'  ,
    'E#:maj'  ,
    'F#:maj'  ,
    'G#:maj'  ,
    'A#:min'  ,
    'B#:min'  ,
    'C#:min'  ,
    'D#:min'  ,
    'E#:min'  ,
    'F#:min'  ,
    'G#:min'  

]
Xfull = []
Yfull = []

def populateDict(maxReps, chordList):
    count = 0
    newDict2 = {}
    for i in range(len(chordList)):
        for j in range(maxReps):
            newDict2.update({str(chordList[i]) + str(j) : count})
            count +=1
    return newDict2

def createMasterArray(Arr, reps):
    C = list(); 
    for i in range(len(Arr)):
        C.extend(list([Arr[i]] * int(reps[i])));
    return C;

def serializeArr(Arr):
    return [(i[0], i[1]) for i in Arr]
                   
                   
def padArr(Arr, lenSeq):
    n = len(Arr);
    numZeroToPad = max(lenSeq - n, 0);
    Sol = serializeArr(Arr) + numZeroToPad * ['N'];
    return Sol;

                   
def convertToOutputList(Arr):
    # input =  Arr = [[aaa][b][cccc]]
    # output = [[a,3],[b,1],[c,4]]
    return [[e[0], len(e)] for e in Arr];
                   
                   
def windowAlg(wind, maxReps):
    if (len(wind) == 0):
        return [];
    X_n = list();
    curr_list = [wind[0]]
    for curr in wind[1:]:
        if ((curr_list[-1] == curr) and (len(curr_list) < maxReps)):
            curr_list.append(curr)
        else:
            X_n.append(curr_list)
            curr_list = [curr]
    if (curr_list):
        X_n.append(curr_list)
    return X_n
                   
                   
def getNumberOfRemainingElements(idx, e, C):
    count = 0
    for i in range(idx, len(C)):
        if C[i] != e:
            break;
        else:
            count += 1
    return count;
    
                   
def createX(idx, lenSeq, maxReps, C):
    return windowAlg(C[idx: idx+lenSeq], maxReps);
    
                   
def createY(idx, lenPred, maxReps, C):
    Y_n = list()
    if (len(C[idx:idx+lenPred]) == 0):
        return Y_n
    Y_n = windowAlg(C[idx:idx+lenPred], maxReps)
    lastY = Y_n[-1]
    
    if idx+lenPred >= len(C):
        return Y_n
    nextC = C[idx+lenPred]
    if nextC != lastY[0]:
        return Y_n
   
    # else, C has more of the same elements as lastY[0]
    lastY = Y_n.pop(-1)
    rem = getNumberOfRemainingElements(idx+lenPred, lastY[0], C)
    appendRemaining = windowAlg((rem+len(lastY))*[lastY[0]], maxReps);
    return Y_n + appendRemaining;



def expandElements(Arr, maxReps, padFactor):
    Serial = createMasterArray([e[0] for e in Arr], [e[1] for e in Arr]);
    X = [[Serial[0], 0]]
    full = list()
    count = 0
    for i, e in enumerate(Serial):
        if count == maxReps or e != X[-1][0]:
            X.append([e, 1]);
            count = 1
        else:
            X[-1][1] += 1
            count += 1
        full.append(padArr(X,padFactor))
    return full;


#change this loop so Arr is just ONE chord progression, NOT an entire song's progressions

def oneHot(Seq, maxReps, lenSeq, label):

    #create dictionary
    dictChord = chordUtil.getDictChordUpgrade(chordUtil.a5
        , maxReps)[0]
    
    #the Xfull/Yfull list, 2 for each track
    if label == 'X':
    
        dictValues = list()
        X = torch.zeros(lenSeq, len(dictChord))
        #create dictValue for each chord quantity
        for i in range(len(Seq)):
            if type(Seq[i]) is tuple:
                chord = Seq[i][0][0]
                beat = Seq[i][1]
                dictKey = str(chord) + " " + str(beat)
                dictValue = dictChord[dictKey]
            else:
                dictValue = dictChord['N']
            dictValues.append(dictValue)
        for z in range(lenSeq): 
            if len(dictValues) < lenSeq:
                continue
            X[z][dictValues[z]] = 1
        return X  #full list of all tensors, looped by song 

    else:
        dictValues = list()
        Y = torch.zeros(lenSeq, len(dictChord))
        #create dictValue for each chord quantity
        for i in range(len(Seq)):
            if type(Seq[i]) is tuple:
                chord = Seq[i][0][0]
                beat = Seq[i][1]
                dictKey = str(chord) + " " + str(beat)
                dictValue = dictChord[dictKey]
            else:
                dictValue = dictChord['N']
            dictValues.append(dictValue)
        #print(dictValues)
            
        for z in range(lenSeq): 
            if len(dictValues) < lenSeq:
                continue
            Y[z][dictValues[z]] = 1
        
        return Y  #full list of all tensors, looped by song
   

def saveSetGroupBy(chords, reps, maxReps, lenSeq, lenPred): 
    #print(chords)
    C = createMasterArray(chords, reps);
    #pprint.pprint(C)
    X = list();
    Y = list();
    for i in range(len(C)):
        X_n = createX(i, lenSeq, maxReps, C);
        Y_n = createY(i+lenPred, lenPred, maxReps, C);
        #pprint.pprint(X_n)
        if (X_n and Y_n):
            if i == 0:
                X.extend(expandElements(convertToOutputList(X_n), maxReps, lenSeq))
                Y.extend(padArr(convertToOutputList(Y_n), lenPred))
            else:
                X.append(padArr(convertToOutputList(X_n), lenSeq))
                Y.append(padArr(convertToOutputList(Y_n), lenPred))
  
    

    for i in range(len(X)): 
        Xshort = oneHot(X[i], maxReps, lenSeq, 'X')
        #pprint.pprint(Xshort.size())
       # pprint.pprint(X.size())
       
        Xfull.append(Xshort)

        Yshort = oneHot(Y[i], maxReps, lenPred, 'Y')
        #pprint.pprint(Yshort.type())
      
        Yfull.append(Yshort)
        #pprint.pprint(Yshort.size(0))
   # pprint.pprint(Yfull)
    return Xfull, Yfull

#def transfToOld(output, lenPred):


















