from library import *
from numpy import *

# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
    sum_col = sum(jP,axis=0)
    sum_row = sum(jP,axis=1)    
    for rows in range(jP.shape[0]):
        for cols in range(jP.shape[1]):
            if(jP[rows,cols]!=0):
                mi += (jP[rows,cols]*log2(jP[rows,cols]/(sum_col[cols]*sum_row[rows])))

    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))

    for i in range(noVariables):
        for j in range(noVariables):
            jPT = JPT(theData, i, j, noStates)
            MIMatrix[i][j] = MutualInformation(jPT)
    return MIMatrix

# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]

    for rows in range(depMatrix.shape[0]):
        for cols in range(rows+1,depMatrix.shape[1]):
           depList.append([depMatrix[rows][cols],rows,cols])        
    depList.sort(reverse=True)
    return array(depList)
#
# Functions implementing the spanning tree algorithm   
def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    link = []
    new = True
    for node in depList:
        x, y = node[1], node[2]

        for i in range(len(link)):
            if(x in link[i] and y in link[i]):
                new = False
                break

        if(new):
            x_index = -1;
            y_index = -1;
            for i in range(len(link)):
                if(x in link[i]):
                    x_index = i;
                if(y in link[i]):
                    y_index = i;
                    
            if(y_index!= -1 and x_index!= -1):
                new_lst = link[x_index] + link[y_index]
                del link[max(x_index,y_index)], link[min(x_index, y_index)]
                link.append(new_lst)
                spanningTree.append(node)
                
            if(x_index == -1 and y_index != -1):
                link[y_index].append(x)
                spanningTree.append(node)
                
            if(x_index !=-1 and y_index == -1):
                link[x_index].append(y)
                spanningTree.append(node)
                
            if x_index == -1 and y_index == -1 :
                link.append([x,y])
                spanningTree.append(node)
            
        new = True    
    return array(spanningTree)

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
AppendString("results.txt","Coursework Two Results by Aditya Chaturvedi (ac2917)")
AppendString("results.txt","") #blank line

AppendString("results.txt","Dependency Matrix")
depMatrix = DependencyMatrix(theData, noVariables, noStates)
AppendArray("results.txt", depMatrix)

AppendString("results.txt","Dependency List")
depList = DependencyList(depMatrix)
AppendArray("results.txt", depList)

AppendString("results.txt","Maximally weighted spanning tree")
spanningTree = SpanningTreeAlgorithm(depList, noVariables)
AppendArray("results.txt", spanningTree)





