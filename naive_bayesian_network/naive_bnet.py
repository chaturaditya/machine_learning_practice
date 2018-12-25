from library import *
from numpy import *
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
    col = theData[:,root]
    length = len(col)
    for i in range(noStates[root]):
        occ = (col == i).sum()
        prior[i] = occ/float(length)

    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
 
    for i in  range(noStates[varP]):
        indices = (theData[:,varP] == i)
        dataReduced = theData[indices,varC]
        
        for j in range(noStates[varC]):
            occ = (dataReduced == j).sum()
            cPT[j,i] = occ/float(len(dataReduced))
    return cPT

# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )

    lenRow = len(theData[:,varRow])
    for i in  range(noStates[varCol]):
        indices = (theData[:,varCol] == i)
        dataReduced = theData[indices,varRow]
        for j in range(noStates[varRow]):
            N = (dataReduced == j).sum()
            jPT[j,i] = N/float(lenRow)
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    col = 1
    tot = sum(row[col] for row in aJPT)
    print(tot)	
    return aJPT

# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)


noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
AppendString("results.txt","Coursework One Results by Aditya Chaturvedi")
AppendString("results.txt","") #blank line

AppendString("results.txt","The prior probability of node 0")
prior = Prior(theData, 0, noStates)
AppendList("results.txt", prior)

AppendString("results.txt","The conditonal probabilty P(2|0)")
cPT = CPT(theData, 2, 0, noStates)
AppendArray("results.txt", cPT)

AppendString("results.txt","The conditonal probabilty P(2&0)")
jPT = JPT(theData, 2, 0, noStates)
AppendArray("results.txt", jPT)

AppendString("results.txt","The conditonal probabilty matrix P(2|0) calculated from the joint probabilty matrix P(2&0)")
ajPT = JPT2CPT(jPT,prior)
AppendArray("results.txt", ajPT)

