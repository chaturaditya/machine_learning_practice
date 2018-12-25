def SpanningTreeAlgorithm(depList, noVariables):#
    spanningTree = []
    chain = []
    flag = True
    duplicate = False
    for dep in depList:
        x = dep[1]
        y = dep[2]
        
        for i in range(0, len(chain)):
            if(x in chain[i] and y in chain[i]):
                duplicate = True
                flag = False
                break
        if not duplicate:
            for i in range(0,len(chain)):
                
                if(x not in chain[i] and y in chain[i]):
                    chain[i].append(x)
                    chain[i].append(y)
                    spanningTree.append(dep)
                    flag = False
                    break
                elif(x in chain[i] and y not in chain[i]):
                    chain[i].append(x)
                    chain[i].append(y)
                    spanningTree.append(dep)
                    flag = False
                    break
            duplicate = False
        
        if(flag):
            print("ME!")
            chain.append([x,y])
            spanningTree.append(dep)
        flag = True    
    return array(spanningTree)    
        