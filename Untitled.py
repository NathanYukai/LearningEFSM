
# coding: utf-8

# In[1091]:

#utility functions
import random, os, subprocess
import numpy as np
from shutil import copyfile, rmtree


def write_to_file(file_name,trace):
    with open(file_name,'w') as f:
        f.write(trace)

def append_to_file(file_name,trace):
    with open(file_name,'a') as f:
        f.write(trace)

#parameters are file names
def execute_clingo(trace, automata):
    script = "clingo " + trace + " " + automata
    proc = subprocess.Popen(script,stdout=subprocess.PIPE,shell=True)
    return proc

def execute_ILASP(file_name):
    script = "ILASP-b --2i {} --cached-rel={}".format(file_name,file_name+'.cache') 
    process = subprocess.Popen(script,stdout=subprocess.PIPE,shell=True)
    return process

def getILASPSpace(state_num):
    extra = ''
    for i in range(state_num):
        extra += "#constant(st,state{}).\n".format(i)
    
    tmp_file = 'useIlasp/modetmp.lp'
    copyfile('useIlasp/ilaspMode.lp', tmp_file)
    append_to_file(tmp_file,extra)
    
    script = "ILASP -s "+ tmp_file
    process = subprocess.Popen(script,stdout=subprocess.PIPE,shell=True)
    res = []
    a = process.stdout.readline()
    while(a):
        res.append(a)
        a = process.stdout.readline()
        
    os.remove(tmp_file)
    return res

def getRegILASPSpace(state_num,reg_limit):
    extra = ''
    for i in range(state_num):
        extra += "#constant(st,state{}).\n".format(i)
    for r in range(reg_limit):
        extra += "#constant(reg_value,{}).\n".format(r)
    extra += '#modeh(final_reg(const(reg_value))). \n'
    
    tmp_file = 'useIlasp/modeRegtmp.lp'
    copyfile('useIlasp/ilaspRegMode.lp', tmp_file)
    append_to_file(tmp_file,extra)
    
    script = "ILASP -s "+ tmp_file
    process = subprocess.Popen(script,stdout=subprocess.PIPE,shell=True)
    res = []
    a = process.stdout.readline()
    while(a):
        res.append(a)
        a = process.stdout.readline()
        
    os.remove(tmp_file)
    return res


def clearLogDir():
    dir_name = 'useIlasp/log'
    if(os.path.isdir(dir_name)):
        rmtree(dir_name)
    os.makedirs(dir_name)

#get the path for that automata
def getLogPath(file_name,auto_id):
    log = 'useIlasp/log'
    dirname = 'auto'+str(auto_id)
    autodir = os.path.join(log,dirname)
    if(not os.path.isdir(autodir)):
        os.makedirs(autodir)
        
    path = os.path.join(autodir,file_name)
    return path

def log(file_name,content,auto_id):
    path = getLogPath(file_name, auto_id)
    if(os.path.isfile(path)):
        append_to_file(path,content)
    else:
        write_to_file(path,content)
    
def string_to_trace(string):
    res = ''
    for i in range(len(string)):
        res += 'input({},{}).'.format(i,string[i])
    res += 'trace_length({}).\n'.format(len(string))
    return res
    
def get_random_inputs(limit):
    sections = random.randint(0,10)
    if(sections<2):
        ran_int = random.randint(0,3)
    elif(sections<5):
        ran_int = getNormalRandomWithLimit(3,10,100)
    else:
        ran_int = random.randint(0,limit)
    
    ran_string = str(bin(ran_int))
    ran_string = ran_string[2:]
    res = '0'*getNormalRandomWithLimit(1,2,5)+ran_string
    return res

def calcSTD(data):
    arr = np.array(data)
    res = np.std(arr)
    return res

import math
def mapLog(arr):
    res = [math.log(x) for x in arr]
    return res
    


# In[1092]:

#mutual exclusive condition needed for the deltas\n",
# return a list of tuple , (from state, number of MX)\n",
# e.g. [(0,2)] means state0 need MX for two state\n",
def mutual_required(deltas):
    from_states = []
    for d in deltas:
        from_states.append(d[0])
    result = []
    for f in from_states:
        count = from_states.count(f)
        if (count>1) :
            result.append((f,count))
    return list(set(result))

#only works for integers, take input limit, number of conditions needed
# return list of conditions, guaranteed to be mutually exclusive
# return [[l,u]...]
def generate_conditions(lower_bound,upper_bound,num):
#     if(upper_bound - lower_bound +1 < num):
#         print "not so possible to generate so many conditions"
#         return []
#     results = []
    
#     if(upper_bound-lower_bound +1 == num):
#         for i in range(lower_bound,upper_bound+1):
#             results.append([i,i])
#         return results
    
#     #list of ranges available : [[l,u],[l,u],[l,u]]
#     ranges = [[lower_bound,upper_bound]]
#     while(len(results)<num):
#         #choose a range from list of continuous ranges
#         chose_range = ranges[random.randint(0,len(ranges)-1)]
#         low = chose_range[0]
#         upp = chose_range[1]
#         l = random.randint(low,upp)
#         u = random.randint(l,upp)
#         results.append([l,u])

#         #update ranges
#         left_upper = l-1
#         right_lower = u+1
#         ranges.remove([low,upp])
#         if(left_upper >= low):
#             ranges.append([low,left_upper])
#         if(right_lower <= upp):
#             ranges.append([right_lower,upp])
            
#         #reset if requirement can't be met
#         possible_ranges = 0
#         for r in ranges:
#             possible_ranges += r[1]-r[0]
#         if(possible_ranges+len(results) < num):
#             results = []
#             ranges = [[lower_bound,upper_bound]]
    if(num==2):
        return [[0,0],[1,1]]
    elif(num==1):
        one_prob = 4
        zero_prob = 8
        ran = random.randint(0,10)
        if(ran < one_prob):
            return [[1,1]]
        elif(ran < zero_prob):
            return [[0,0]]
        else:
            return [[0,1]]
    else:
        print 'shoudnt happen here, generate condition'
        return []


def fillUpConditionsForDelta(deltas,automata):
    
    #generate conditions along with deltas
    #[(stateNumber,mxNumber)]
    mx_needed = mutual_required(deltas)
    
     #index corresponding to deltas
    conditions = [[]]*len(deltas)
    #index corresponding to states, elements is condition numbers
    error_conditions = [[]]*len(automata.states)

    #fill up conditions and error_conditions
    for mx in mx_needed:
        #generate that many conditions, can add a random extra number
        # but then need to pick mx[1] out of them, TODO later
        mx_conditions = generate_conditions(automata.in_low,automata.in_upp,mx[1])
        for d in range(len(deltas)):
            #if this delta need MX
            from_st = deltas[d][0]
            if (from_st == mx[0]) :
                if(len(mx_conditions) == 0):
                    print "something wrong in calculating MX?"
                conditions[d] = mx_conditions[0]
                #Delta number is corresponding to condition number
                # error condition records each state go to error state if 
                # none of it's condition are met
                error_conditions[from_st] = error_conditions[from_st]+[d]
                del mx_conditions[0]

    #fill up non-mx required conditions
    #TODO add more complex conditions
    for c in range(len(conditions)):
        if (conditions[c]==[]):
            conditions[c] = generate_conditions(automata.in_low,automata.in_upp,1)[0]
            from_st = deltas[c][0]
            error_conditions[from_st] = [c]
    return conditions,error_conditions

def draw_graph(a,graph,file_name):
#     pass
    g = nx.nx_pydot.to_pydot(graph)
    for i in range(len(a.deltas)):
        edge = g.get_edges()[i]
        delta = (int(edge.obj_dict['points'][0]),int(edge.obj_dict['points'][1]))
        con = a.deltas.index(delta)
        edge.obj_dict['attributes']['label'] = str(a.conditions[con])
    g.write_png(file_name)
   


# In[1093]:

import random
# import pkg_resources
# pkg_resources.require("networkx==2.0")
import networkx as nx
import pydot
#ASSUMPTIONS: only one input, represented by number
#input should include an trace_length 

    
cond_param = 'C'
#need to pass time parameter around,
#so that that condition is only true at that time
time_param = 'T'
def gen_states(output, states):
    for s in states:
        output += "state(" + str(s) + ")."
    output += "\n\n"
    return output 

def gen_condition_tostr(output,conditions):
    for c in range(len(conditions)):
        output += "condition(" + cond_param+ "," +str(c)+"):- "
        output += cond_param+" >= "+ str(conditions[c][0]) +','
        output += cond_param+" <= " + str(conditions[c][1]) + ", "
        output += "input(_,"+cond_param+"). \n"
    output += '\n'
    return output


def deltaToString(fr,to,cond):
    res = ''
    res += "delta("+ fr+ ","
    res += cond_param + ","
    res +=  to+ ","+cond+ "):- input(_,C). \n"
    return res

def gen_deltas_tostr(output,deltas,states):
    for d in range(len(deltas)):
        from_st = deltas[d][0]
        to_st = deltas[d][1]
        mx_conditions = []
#         output += "delta("+ states[from_st]+ "," + time_param + ","
#         output += cond_param+ ","+ states[to_st]+ ","+str(d)+ "):-"
#         output += 'input('+time_param + "," +cond_param+')' +'.\n'
        output += deltaToString(states[from_st],states[to_st],str(d))
    
    output += '\n'
    return output

def gen_state_trans(output, low, init_state):
    output += "st(" + str(low) + "," + init_state+ ").\n"
    output += "st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,C,TO,ID),condition(C,ID),input(T,C).\n\n"
    return output


def getNormalRandomWithLimit(e,std,limit):
    ran = int(random.normalvariate(e,std))
    if(limit <=0 ):
        return ran
    res = max(min(ran,limit),0)
    return res

def setStartAndEndStateForGraphGen(length):
    l = length
    if(l<=1):
        print "we only interested in graph with more than 1 state"
        return 
    din=[1]*l
    dout=[1]*l

    #add possibility for max delta 
    if(random.randint(0,100)<5):
        #start state can have zero in
        din[0] = getNormalRandomWithLimit(1,1,l)
        dout = [2]*l
        return din,dout

    #start state must have:
    #at least one out, 
    #can have zero in
    din[0] = getNormalRandomWithLimit(1,1,l)
    dout[0] = random.randint(1,2)
    
    #end state must have:
    #at least one in
    #at most 2 out
    din[l-1] = max(1,getNormalRandomWithLimit(1,1,l))
    dout[l-1] = random.randint(0,2)
    
    if(l>2):
        for i in range(l-2):
            dout[i] = (random.randint(1,2))
    
    return din,dout

def gen_graph(l):
    
    din,dout = setStartAndEndStateForGraphGen(l)
    while(sum(dout)-sum(din) < 0 ):
        din,dout = setStartAndEndStateForGraphGen(l)

    diff = sum(dout)-sum(din)
    while(diff>0):
        i = random.randint(0,l-1)
        if(din[i] < l):
            diff -= 1
            din[i] = din[i]+1
        
    D=nx.directed_configuration_model(din,dout)
    D=nx.DiGraph(D)
    return D

    
def gen_constraints(output,final_state):
    output += "accept :- st(T," + final_state + ")," + "trace_length(T).\n"
    output += ":- not accept. \n"
    return output
    
# all states can goto error state if none of it's conditions are met
def gen_error_state_conditions(output, error_conditions,states,deltas):
    output += "state(error).\n"
    start_id = len(deltas)
    for s in range(len(states)):
        cond_id = str(start_id +s)
        output += "condition("+ cond_param+ "," +cond_id+"):- "
        for c in range(len(error_conditions[s])):
            condition_number = error_conditions[s][c]
            output +=  "not condition("+ cond_param +','+str(condition_number) +"),"
        output += "input(_"  + ","+ cond_param +").\n"
        output += deltaToString(states[s],'error',cond_id)
        output += '\n'
    return output

def gen_input_complete():
    res = '1{input(T,0);input(T,1)}1:- time(T).\n'
    res += 'time(T-1):- trace_length(T).\n'
    res += 'time(T-1):- time(T), T>=1.\n'
    return res


def find_corresponding_conds(path,deltas,conditions):
    res = []
    #p is a path e.g.[0,1,3,2,4]
    #loop through all element except last one
    for i in range(len(path)-1):
        con_index = deltas.index((path[i],path[i+1]))
        cond = conditions[con_index]
        res.append(cond)
    return res
    
def find_conds_for_err(path,conditions,err_cons,in_low,in_upp):
    neg_cons_indexs = err_cons[path[-2]]
    neg_conditions = []
    #get all posible out going edges from that state
    for ind in neg_cons_indexs:
        neg_conditions.append(conditions[ind])

    #then subtract
    cond_to_error = range(in_low,in_upp+1)
    for rng in neg_conditions:
        for i in range(rng[0],rng[1]+1):
            cond_to_error.remove(i)
    return cond_to_error

   
        
def paths_to_conditions(auto,paths):
    results = []
    for p in range(len(paths)):
        conditions = find_corresponding_conds(paths[p],auto.deltas,auto.conditions)
        results.append(conditions)
    return results

def invalid_path_to_conditions(auto,paths):
    results = []
    for p in paths:
        #if it goes to error
        if(p[-1]==auto.state_num): 
            conditions = find_corresponding_conds(p[:-1],auto.deltas,auto.conditions)
            to_err = find_conds_for_err(p,auto.conditions,auto.error_cons,auto.in_low,auto.in_upp)
            conditions.append(to_err)
        #if it stops at a internal state
        else:
            conditions = find_corresponding_conds(p,auto.deltas,auto.conditions)
        
        results.append(conditions)
    return results

def getMissing_ErrorInputs_index(max_length, m_perc, m_prob, e_perc, e_prob):
    clingo_trace = ''
    missing_happen = random.randint(0,100) < m_prob*100
    error_happen = random.randint(0,100) < e_prob*100
    missing_inputs = []
    error_inputs = []
    if(missing_happen):
        clingo_trace += '%% trace below has missing inputs \n'
        m_num = int(max_length*m_perc)
        missing_inputs = random.sample(range(max_length),m_num)
    if(error_happen):
        clingo_trace += '%% trace below has error!! inputs \n'
        e_num = int(max_length*e_perc)
        error_inputs = random.sample(range(max_length), e_num)    
    return clingo_trace, missing_inputs, error_inputs

def possibleConditions_to_traces(con_traces, auto):
    result = []
    missing_num = auto.missing_num
    miss_prob = auto.missing_prob
    error_num = auto.error_num
    error_prob = auto.error_prob
    
    
    #cons: [[0,0],[1,1],[0,1]]
    for cons in con_traces:
        if ([] in cons):
            print 'this trace is impossible to complete'
            return []

        tmp_res = getMissing_ErrorInputs_index(len(cons), missing_num, miss_prob, error_num, error_prob)
        clingo_trace, missing_index, error_index = tmp_res
        
        for c in range(len(cons)):
            rng = cons[c]
            value = rng[0]
            if(len(rng)>1):
                value = random.randint(rng[0],rng[1])
            if(c in error_index):
                value = 1-value
            if(not c in missing_index):
                clingo_trace += 'input({0},{1}).'.format(c,value)
        clingo_trace += 'trace_length({0}).'.format(len(cons))
        result.append(clingo_trace)
    return result


# from [[[0,1],[1,1]],[trace],[trace]]
# to %input(), input(),...
# randomly select one of the conditions
def conditions_to_traces(path_cond,auto):
    con_traces =  path_cond
    res = possibleConditions_to_traces(con_traces,auto)
    return res

def invalid_conditions_to_traces(inv_p_c,auto):
    con_traces =  inv_p_c
    con_possible_traces = []
    
    for cons in con_traces:
        if (not [] in cons):
            con_possible_traces.append(cons)
    res = possibleConditions_to_traces(con_possible_traces,auto)
    return res
            
def graph_with_error_state(auto,graph):
    newG = graph.copy()
    err = len(auto.states)
    newG.add_node(err)
    for s in range(err):
        newG.add_edge(s,err)
    return newG
    
#ALGORITHM FOR GEN_ALL_PATH:
#first find all simple path,
#then for each circle, add path from start to circle
#and add path to tail

def getAllPossibleCycleStart(graph, fr, to, cycle):
    result = []
    for st in cycle:
        if(nx.has_path(graph,fr,st) and nx.has_path(graph,st,to)):
            result.append(st)
    return result
            
def shift(key, array):
    return array[key:]+array[:key]

def gen_all_paths(graph,fr,to,more=False):
    all_paths = []
    final_state = to
    simple = nx.all_simple_paths(graph,fr,final_state)
    circles = nx.simple_cycles(graph)
    for p in simple:
        all_paths.append(p)
    for cyc in circles:
        head = []
        tail = []
        all_start = [cyc[0]]
        if(more):
            all_start = getAllPossibleCycleStart(graph,fr,to,cyc)
        for start_point in all_start:
            c = shift(cyc.index(start_point), cyc)
            head = nx.shortest_path(graph,fr,start_point)
            tail = nx.shortest_path(graph,start_point,final_state)
            if(start_point==fr):
                p = c+c+tail
            elif(start_point==(final_state)):
                p = head + c[1:]+c+[start_point]
                p_ext = head+c[1:]+[final_state]
                all_paths.append(p_ext)
            else:
                p = head+c[1:]+tail           
            all_paths.append(p)
    return all_paths


# In[1094]:

import os,re

def check_trace_valid(trace, automata_file):
    #write trace to file
    tmp_file = './trace_tmp.lp'
    write_to_file(tmp_file, trace)
        
    #execute script, call clingo
    res_clingo = execute_clingo(tmp_file, automata_file)
    
    output = ''
    a = res_clingo.stdout.readline()
    #check result
    result = False
    while(a)  :
        output += a
        a = res_clingo.stdout.readline()
        if ("UNSATISFIABLE" in a):
            result = False
        elif("SATISFIABLE" in a ):
            result = True
    os.remove(tmp_file)
#     print output
    return result
            


def checkEdgeValid(automata,edge):
    g = automata.graph
    fin_state = len(automata.states)-1
    fr = edge[0]
    to = edge[1]
    if((fr == 0 or nx.has_path(g,0,fr)) and( to == fin_state or nx.has_path(g,to,fin_state))):
        return True
    
    return False

def getValidEdges(automata):
    res = []
    for e in automata.deltas:
        if(checkEdgeValid(automata,e)):
            res.append(e)
    return res

def getRawDelta(raw):
    res = ''
    for r in raw:
        if(not 'delta' in r):
            continue
        res += r + '\n'
    return res

def getRawTransition(raw):
    res = ''
    for r in raw:
        if(not 'st(T+1' in r):
            continue
        res += r + '\n'
    return res

def getFinalReg(raw):
    res = ''
    for r in raw:
        if(not 'final_reg' in r):
            continue
        res += r + '\n'
    return res

def getEdgesFromLearning(raw, learn_id):
    edges = []
    conditions = []
    rawOutput = ''
    
    finishedDeltas = False
    for d in raw:
        if(not 'delta' in d):
            finishedDeltas = True
        rawOutput+=d
        if(not finishedDeltas):
            m = re.search(r"delta\(state(\d+),V0,state(\d+),(\d)", d)
            groups = map(int,m.groups())
            edges.append((groups[0],groups[1]))
            conditions.append(groups[2])
        
    log('rawLearningResult',rawOutput,learn_id)
    return edges,conditions

def getTimeFromLearning(time_str):
    start = time_str.index(':')
    end = time_str.index('s')
    time = time_str[start+2:end]
    return float(time)

def conditions_to_determininistic(path_cond):
    rep = [0,1]
    modified = []
    
    remain = path_cond[:]
    while( len(remain) != 0):
        e = remain[0]
        if(rep in e):
            remain.remove(e)
            index = e.index(rep)
            replace_to_zero = e[0:index]+[[0,0]]+e[index+1:]
            replace_to_one  = e[0:index]+[[1,1]]+e[index+1:]
            modified.append(replace_to_zero)
            modified.append(replace_to_one)
            remain.append(replace_to_zero)
            remain.append(replace_to_zero)
        else:
            remain.remove(e)
            modified.append(e)
    return modified
    
                     
def gen_bare_minimum_example(auto):
    example_str = '#pos(p{0},{{st({1},state{2}),st({3},state{4})}},{{}},{{input({5},{6}).trace_length({7}).}}).'
    final_state = len(auto.states)-1
    l = len(auto.deltas)
    exps = []
    for i in range(l):
        d = auto.deltas[i]
        if(not checkEdgeValid(auto,d)):
            continue
        con = auto.conditions[i]
        
        input_time = 0
        if(d[0]!=0):
            input_time = len(nx.shortest_path(auto.graph,0,d[0]))-1
        trace_length = input_time+1
        if(d[1]!=final_state):
            trace_length = input_time + len(nx.shortest_path(auto.graph,d[1],final_state))
        e = example_str.format(i,input_time,d[0],input_time+1,d[1],input_time,con[0],trace_length)
        if(con == [0,1]):
            extra_e = example_str.format('_extra_'+str(i),input_time,d[0],input_time+1,d[1],input_time,con[1],trace_length)
            exps.append(extra_e)
        exps.append(e)
    return '\n'.join(exps)+'\n'

    
def StateInfoForClingo(auto):
    res = ''
    for i in range(auto.state_num):
        res += 'state(state{0}).\n'.format(i)
    res += ':- not st(T,state{}), trace_length(T).\n'.format(auto.state_num-1)
    return res

def getGeneratedAutomataFile(learn_id):
    parent = getLogPath('',learn_id)
    path = os.path.join(parent,'automata')
    if(not os.path.isdir(path)):
        os.makedirs(path)
    automataFile = os.path.join(path,'generatedAutomata.lp')
    return automataFile


def testAutomataAreSame(learned_auto_asp_file,learn_id,test_range,test_time):
    log_str = ''
    generatedAutomata = getGeneratedAutomataFile(learn_id)
    err_trace = ''
    for i in range(test_time):
        ran_inputs = get_random_inputs(test_range)
        ran_trace = string_to_trace(ran_inputs)
        
        r_g = check_trace_valid(ran_trace,generatedAutomata)
        r_l = check_trace_valid(ran_trace,learned_auto_asp_file)

        if(r_g!=r_l):
            log_str = 'not match,autoID: {}\n trace:{}\n, generated result : {}, leanred result : {}'
            log_str = log_str.format(learn_id,ran_trace,r_g, r_l)
            if(not r_g and r_l ):
                err_trace = ran_trace
            return False,log_str, err_trace
    return True,log_str, err_trace
    
def checkLearningIsRight_ByTest(auto,learningRes,learn_id,inductive = False):
    a = learningRes.stdout.readline()
    output = []
    while(a):
        if ("UNSATISFIABLE" in a):
            print 'learning is unsatisiable!, valid edges:', len(getValidEdges(auto))
            print 'id:', learn_id
            return False,0
        output.append(a)
        a = learningRes.stdout.readline()
        
    #get learned delta(in asp)
    if(inductive):
        template = 'useIlasp/inductiveDFATemplate.lp'
        deltas = getRawTransition(output)
    else:
        template = 'useIlasp/ilaspTemplate.lp'
        deltas = getRawDelta(output)
    time = getTimeFromLearning(output[-2])
    
    learned_auto_asp_file = getLogPath('learnedAutomata.lp',learn_id)
    
    copyfile(template,learned_auto_asp_file)
    
    append_to_file(learned_auto_asp_file, deltas)
    stateInfo = StateInfoForClingo(auto)
    append_to_file(learned_auto_asp_file, stateInfo)
    
    result,err_str,err_trace = testAutomataAreSame(learned_auto_asp_file,learn_id,1000,100)    
    log('error messages', err_str, learn_id)
    
    learnedEdges,learnedConditions =  getEdgesFromLearning(output,learn_id)
    return result, time, err_trace

    
def checkRegLearningIsRight_test(origin_file,test_range,st_num,reg_limit,learningRes,learn_id):
    a = learningRes.stdout.readline()
    output = []
    while(a):
        if ("UNSATISFIABLE" in a):
            print 'learning is unsatisiable!, valid edges'
            print 'file:', origin_file
            print 'id:', learn_id
            return False,0,''
        output.append(a)
        a = learningRes.stdout.readline()
        
    #get learned delta(in asp)
    deltas = getRawDelta(output)
    final_reg = getFinalReg(output)
    time = getTimeFromLearning(output[-2])
    
    learned_auto_asp_file = getLogPath('learnedAutomata.lp',learn_id)
    copyfile('useIlasp/RegTemplate.lp',learned_auto_asp_file)
    
    append_to_file(learned_auto_asp_file, deltas)
    append_to_file(learned_auto_asp_file, final_reg)
    stateInfo = stateRegInfoForIlaspLearning(st_num,reg_limit)
    append_to_file(learned_auto_asp_file, stateInfo)
    append_to_file(learned_auto_asp_file, ':- not accept.\n')
    
    result,err_str, err_trace = testAutomataAreSame(learned_auto_asp_file,                                                    learn_id,test_range,test_range)    
    log('error messages', err_str, learn_id)
    
    learnedEdges,learnedConditions =  getEdgesFromLearning(output,learn_id)
    return result, time, err_trace


def stateInfoForIlaspLearning(state_number):
    res = '\n'
    for i in range(state_number):
        res += "state(state{}).\n".format(i)
    res += "accept :- st(T,state{}),trace_length(T).\n".format(state_number-1)
    return res

def stateRegInfoForIlaspLearning(state_number, reg_limit):
    res = '\n'
    for i in range(state_number):
        res += "state(state{}).\n".format(i)
    res += "reg_limit({}).\n".format(reg_limit)
    res += "accept :- st(T,state{},V), final_reg(V),trace_length(T).\n".format(state_number-1)
    return res

def changeLengthOfDelta(mode,num):
    modified= []
    for m in mode[:-1]:        
        if(len(m) < 10):
            continue  
        reSearchGroup= re.search(r"delta\(state\d+,V0,state\d+,(\d)", m)
        cond_num = reSearchGroup.groups()[0]
        
        if(cond_num=='2'):
            modified.append(str(num)+m[1:])
        else:
            modified.append(m)
            
    return ''.join(modified)

def completeIlaspEncoding(ilasp_file,examples,state_num):
    ilasp_template = 'useIlasp/ilaspTemplate.lp'
    mode = getILASPSpace(state_num)
    
    mode_str = changeLengthOfDelta(mode,2)
    copyfile(ilasp_template,ilasp_file)
    append_to_file(ilasp_file,examples)
    append_to_file(ilasp_file,mode_str)
    
    stateInfo = stateInfoForIlaspLearning(state_num)
    append_to_file(ilasp_file,stateInfo)
#     append_to_file(ilasp_file,'#max_penalty({}).\n'.format(2*2*state_num+2))
    append_to_file(ilasp_file,'#max_penalty({}).\n'.format(100))
    

    
def getInductiveMode(stateNum):
    rule= '1 ~ st(T+1,state{}):- st(T,state{}),input(T,{}).\n'
    state = range(stateNum)
    mode = ''
    for fr in state:
        for to in state:
            for i in ['0','1','_']:
                mode+=(rule.format(fr,to,i))
    return mode
def completeILASPInductiveEncoding(ilasp_file,examples,state_num):
    ilasp_template = 'useIlasp/inductiveDFATemplate.lp'
    mode = getInductiveMode(state_num)
    
    copyfile(ilasp_template,ilasp_file)
    append_to_file(ilasp_file,examples)
    append_to_file(ilasp_file,mode)
    
    stateInfo = stateInfoForIlaspLearning(state_num)
    append_to_file(ilasp_file,stateInfo)
    append_to_file(ilasp_file,'#max_penalty({}).\n'.format(2*state_num+1))
    
    
def completeRegIlaspEncoding(ilasp_file,examples,state_num,reg_limit):
    ilasp_template = 'useIlasp/RegTemplate.lp'
    mode = getRegILASPSpace(state_num,reg_limit)
    
    
    mode_str = ''.join(mode)
    copyfile(ilasp_template,ilasp_file)
    append_to_file(ilasp_file,examples)
    append_to_file(ilasp_file,mode_str)
    
    stateInfo = stateRegInfoForIlaspLearning(state_num,reg_limit)
    append_to_file(ilasp_file,stateInfo)
    append_to_file(ilasp_file,'#max_penalty({}).\n'.format(100))
    
def inputs_to_example_template(head_str, trace,final_state,count=0,error=False):
    res = ''
    for t in range(len(trace)):
        #there might be some comment for the trace, for debugging only
        tmp = trace[t].split('\n')
        if(len(tmp)>1):
            res += '\n'.join(tmp[:-1]) + '\n'
            
        #real format
        head = head_str.format(t+count)
        trace_length_s = re.search(r"trace_length\((\d+)\)", tmp[-1])
        trace_length = int(trace_length_s.groups()[0])
        if(not error):    
            st_info = 'state'+str(final_state)
        else:
            st_info = '_'

        trace_end_info = '{{st({},{})}},{{}},{{'.format(trace_length,st_info)
        res += head + trace_end_info + tmp[-1] + '}).\n'
    return res

#don't need map for now, probably never tho
def inputs_to_ilasp_examples(final_state,clingo_traces,invalid_traces,invalid_int_traces):
    exp = ''
    exp += inputs_to_example_template("#pos(p{0},", clingo_traces,final_state)
    exp += inputs_to_example_template("#neg(n{0},", invalid_traces,final_state,error=True)
    exp += inputs_to_example_template("#neg(n{0},", invalid_int_traces,                                      final_state,count=len(invalid_traces))
    return exp


    


# In[1095]:

def allStateValid(auto,graph):
    final_state = auto.state_num -1
    for s in range(auto.state_num):
        check_from = (s==0) or nx.has_path(graph,0,s)
        check_to = (s==final_state) or nx.has_path(graph,s,final_state)
        if(not (check_from and check_to)):
            return False
    return True

def checkMutualPossible(deltas):
    for (s,m) in mutual_required(deltas):
        if(m > 2):
            return False
    return True
    
        
def getValidGraph(auto):
    graph = gen_graph(auto.state_num)
    deltas = list(graph.edges())
    valid_graph = checkMutualPossible(deltas)
    count = 1
    while( not (valid_graph and allStateValid(auto,graph))):
        graph = gen_graph(auto.state_num)
        count +=1
        deltas = list(graph.edges())
        valid_graph = checkMutualPossible(deltas)
                
    return graph

class Automata:

    #assume first is the initial state and 
    #the last one is the accepting state
    states = []
    #lower and upper bound of the input
    in_low = 0
    in_upp = 1
    minStates = 2
    
    missing_num=0
    missing_prob=0
    error_num = 0
    error_prob=0


    def __init__(self, state_num, in_low, in_upp):
        self.state_num = state_num
        self.states = []
        self.in_low = in_low
        self.in_upp = in_upp

    def setMissingInput(self,num,prob):
        self.missing_num=num
        self.missing_prob = prob
    
    def setErrorInput(self,num,prob):
        self.error_num = num
        self.error_prob = prob
        
    def gen_valid_paths(self):
        return gen_all_paths(self.graph,0,len(self.states)-1,True)

    def gen_invalid_paths(self):
        modified_g = self.graph.copy()
        err_graph = graph_with_error_state(self,modified_g)
        error_paths = gen_all_paths(err_graph,0,len(self.states),True) 
        
        internal_paths = []
        #traces that stops at internal state
        for i in range(0,self.state_num-1):
            if(nx.has_path(self.graph,0,i)):
                inv_ps = gen_all_paths(self.graph,0,i,True)
                internal_paths += inv_ps
        
        return error_paths, internal_paths

        
    def generate_automata(self):
        output = ""

        num_states = max(self.state_num,self.minStates)
        for i in range(num_states):
            self.states.append("state"+str(i))

        #limiting inputs
        output += "in_limit(" + str(self.in_upp) + "). \n"
        output = gen_states(output, self.states)

        
        graph = getValidGraph(self)
        deltas = list(graph.edges())
#         print "generated graph, {} times".format(count)
        
        #condition: [[l,u],...] index is corresponding to delta index
        #error_conditions: [[con,con],..],index is state number, 
        #con is the conditions that should not be satisfied
        conditions,error_conditions = fillUpConditionsForDelta(deltas,self)
        
        #output 
        output = gen_condition_tostr(output,conditions)
        output = gen_deltas_tostr(output,deltas,self.states)
        output = gen_state_trans(output,self.in_low, self.states[0])
        output = gen_constraints(output,self.states[-1])

        output += gen_input_complete()
        output += "#show st/2.\n"

        self.output = output
        self.deltas = deltas
        self.graph = graph
        self.conditions = conditions
        self.error_cons = error_conditions
        
    def summarize(self):
        res = ''
        res += "states:" + str(self.states) + '\n'
        res += "deltas:\n" + '\n'.join(str(x) for x in self.deltas) + '\n'
        res += "conditions:\n"
        for i in range(len(self.deltas)):
            res+= str(self.deltas[i]) + ": " + str(self.conditions[i])+" \n"
        res += "error_conditions:\n" + '\n'.join(str(x) for x in self.error_cons) 
        return res
        


# In[1096]:

def logAutomataEncoding(auto, clingo_traces, inv_traces,learn_id):
    #setup directory
    automataFile = getGeneratedAutomataFile(learn_id)
    write_to_file(automataFile,auto.output)
          
    
    parent = getLogPath('',learn_id)
    path = os.path.join(parent,'automata')
    results = True
    for t in range(len(clingo_traces)):
        tFile = 'valid_t'+str(t)+'.lp'
        tPath = os.path.join(path,tFile)
        write_to_file(tPath,clingo_traces[t])
        r = check_trace_valid(clingo_traces[t],automataFile)
        if( not r):
            results = False
            print "somethingwrong with this trace!:",t
    for inv in range(len(inv_traces)):
        iFile = 'invalid_t'+str(inv)+'.lp'
        iPath = os.path.join(path,iFile)
        write_to_file(iPath,inv_traces[inv])
        r = check_trace_valid(inv_traces[inv],automataFile)
        if(r):
            results = False
            print "somethingwrong with this trace!, invalid trace:",t
    return results

def sumGeneration(v_path,v_cond,inv_path,inv_cond):
    res = ''
    res += "valid path:\n" + "\n".join(str(x) for x in v_path)+'\n'
    res += "valid conditions:\n" + "\n".join(str(x) for x in v_cond) +'\n'
    res += "invalid path:\n" + "\n".join(str(x) for x in inv_path)+'\n'
    res += "invalid conditions:\n" + "\n".join(str(x) for x in inv_cond)+'\n'
    return res


# In[1097]:

def draw_reg_graph(graph_name, conMap, graph):
    deltas = list(graph.edges())
    
    g = nx.nx_pydot.to_pydot(graph)
    if(len(deltas)!=len(g.get_edges())):
        print 'wrong edge, delta number in draw_reg_graph,', len(deltas), len(g.get_edges())
    
    for i in range(len(deltas)):
        edge = g.get_edges()[i]
        delta = (int(edge.obj_dict['points'][0]),int(edge.obj_dict['points'][1]))
        cond,reg_v,op = conMap[delta]
        if(op==0):
            op_str = '-'
        elif(op==1):
            op_str = '+'
        elif(op==2):
            op_str = 'r'
        else:
            op_str = 'n'
        string = 'in: {} : reg: {}\n op: {}'.format(cond,reg_v,op_str)
        edge.obj_dict['attributes']['label'] = string
    g.write_png(graph_name)
        
def getRandomFrom(ran):
    res = random.choice(ran)
    return res

def getherMutualRequiredDelta(deltas):
    res = []
    res.append(deltas[0])
    
    for r in deltas[1:] :
        if(r[0]==res[0][0]):
            res.append(r)
    
    for r in res:
        deltas.remove(r)
    
    return res, deltas

def fillDeltaString(d,cond,reg,op):
    return 'delta(state{},V0,state{},{},{},{}):- input(_,C). \n'.format(d[0],d[1],cond,reg,op)
    
def _SplitConditions(reg_limit,mutual_req):
    condOrNum = 0
    condZeroNum = 0
    condOneNum = 0

    while(2*condOrNum+condZeroNum+condOneNum < mutual_req):
        chose_range = [0,1,2]
        cur = 2*condOrNum + condZeroNum + condOneNum
        if( cur + reg_limit*2 + 2 > mutual_req ):
            chose_range.remove(2)
        if(condZeroNum == reg_limit):
            chose_range.remove(0)
        if(condOneNum == reg_limit):
            chose_range.remove(1)

        if(len(chose_range) == 0):
            print 'something wrong , in get delta conditions!!!!!!!!!!!'
            print condOrNum, condZeroNum,condOneNum, mutual_req,reg_limit
        chose = random.choice(chose_range)

        if (chose==0):
            condZeroNum +=1
        elif(chose==1):
            condOneNum +=1
        else:
            condOrNum +=1
    return condOrNum, condZeroNum, condOneNum
            
def genRegDeltaConditions(graph,state_num,reg_limit):
    output = ''
    edges = graph.edges()
    
    #max out -  existing max
    max_out_edge = 2*(reg_limit) -2
    for (s,m) in mutual_required(edges):
        add_num = getNormalRandomWithLimit(3,1.5,max_out_edge)
        for i in range(add_num):
            dst = random.randint(0,state_num-1)
            graph.add_edge(s,dst)
          
    deltas = list(graph.edges())
    
    deltaConditionMap = {}
    
    while(len(deltas)>0):
        res, deltas = getherMutualRequiredDelta(deltas)
        reg_values = range(0,reg_limit)
        
        #register values to chose from, 
        #to keep deltas mutual exclusion
        condOneRegValues = reg_values[:]
        condZeroRegValues = reg_values[:]
        
        #number of zero, one, or conditions 
        condOrNum, condZeroNum, condOneNum = _SplitConditions(reg_limit,len(res))
        
        op_num = 3
        #do or condition first, 
        #so later dont need to get intersection
        for i in range(condOrNum):
            reg_v = getRandomFrom(condOneRegValues)
            condOneRegValues.remove(reg_v)
            condZeroRegValues.remove(reg_v)
            
            op = random.randint(0,op_num)
            output += fillDeltaString(res[0],2,reg_v,op)
            deltaConditionMap[res[0]] = (2,reg_v,op)
            res.remove(res[0])    
            
        for i in range(condOneNum):
            reg_v = getRandomFrom(condOneRegValues)
            condOneRegValues.remove(reg_v)
            
            op = random.randint(0,op_num)
            output += fillDeltaString(res[0],1,reg_v,op)
            deltaConditionMap[res[0]] = (1,reg_v,op)
            res.remove(res[0])

        for i in range(condZeroNum):
            reg_v = getRandomFrom(condZeroRegValues)
            condZeroRegValues.remove(reg_v)
            
            op = random.randint(0,op_num)
            output += fillDeltaString(res[0],0,reg_v,op)
            deltaConditionMap[res[0]] = (0,reg_v,op)
            res.remove(res[0])
               
#     print output
    return output,deltaConditionMap

def RegAutomataValid(auto_file,test_range):
    count = 0
    for i in range(test_range):
        ran_inputs = get_random_inputs(test_range)
        ran_trace = string_to_trace(ran_inputs)
        if(check_trace_valid(ran_trace,auto_file)):
            count +=1
        if(count >1):
            return True
    
    return False
        

class RegAutomata:

    #assume first is the initial state and 
    #the last one is the accepting state
    states = []
    
    minStates = 2
    
    missing_num=0
    missing_prob=0
    error_num = 0
    error_prob=0

    reg_limit = 1

    def __init__(self, state_num, reg_limit):
        self.state_num = state_num
        self.states = []
        self.reg_limit = reg_limit

    def setMissingInput(self,num,prob):
        self.missing_num=num
        self.missing_prob = prob
    
    def setErrorInput(self,num,prob):
        self.error_num = num
        self.error_prob = prob

    def generate_automata(self,dst):
        copyfile('useIlasp/RegTemplate.lp', dst)
        end_reg_value = random.randint(0,self.reg_limit)
        final_state = self.state_num-1
        
        graph = getValidGraph(self)
        
        deltas_str,conMap = genRegDeltaConditions(graph,self.state_num,self.reg_limit)
        
        self.graph = graph
        extra = ''
        for i in range(self.state_num):
            extra += 'state(state{}).\n'.format(i)
        
        extra += 'accept :- st(T,state{},{}),trace_length(T).\n'.format(final_state,end_reg_value)
        extra += 'reg_limit({}).\n'.format(self.reg_limit)
        extra += ':- not accept.'
        
        append_to_file(dst,deltas_str)
        append_to_file(dst,extra)
        
        return conMap
       
        


# In[1098]:

def getPoolFileID(path):
    i = 0
    if(not os.path.isdir(path)):
        os.makedirs(path)
    while(os.path.isfile( os.path.join(path,'id_{}.lp'.format(i) ))):
        i+=1
        
    return i

def generateRegAutoPool():
    pool = 'useIlasp/RegAutomataPool/'
    if(not os.path.isdir(pool)):
        os.makedirs('useIlasp/RegAutomataPool/')
    count = 0
    for i in range(100000):
        print i

        st_number = random.randint(2,5)
        reg_lim = random.randint(2,4)
        generatedReg_file = 'generatedReg.lp'
        auto = RegAutomata(st_number,reg_lim)
        conMap = auto.generate_automata(generatedReg_file)
        valid = RegAutomataValid(generatedReg_file,600)
        if(valid):
            count +=1
            print 'found one!'
            dir_name = 'regAutomataState_{}Reg_{}'.format(st_number,reg_lim)
            path = os.path.join(pool,dir_name)
            file_id = getPoolFileID(path)
            file_name = os.path.join(path,'id_{}.lp'.format(file_id))
            graph_name = os.path.join(path,'id_{}.png'.format(file_id))
            draw_reg_graph(graph_name, conMap,auto.graph)
            
            copyfile(generatedReg_file,file_name)
    
    print 'done!, found reg auto:', count


# In[1099]:

def flip_bit_in_inputs(inputs,index):
    flipped = str(1-int(inputs[index]))
    return inputs[:index]+flipped + inputs[index+1:]
   
    
def inputs_to_trace_with_missing_err(inputs,miss,err):
    for e in err:
        inputs = flip_bit_in_inputs(inputs,e)

    trace = ''
    for i in range(len(inputs)):
        if(not i in miss):
            trace += 'input({},{}).'.format(i,inputs[i])
    trace += 'trace_length({}).\n'.format(len(inputs))
    
    return trace
    
def get_init_valid_trace(auto,learn_id):
    valid_paths = auto.gen_valid_paths()
    path_cond = paths_to_conditions(auto,valid_paths)
    path_cond_determin = conditions_to_determininistic(path_cond)
    generated_auto_file = getGeneratedAutomataFile(learn_id)
    
    
    res = []
    for p in path_cond_determin:
        string = ''
        for i in p:
            string += str(i[0])
            
        _,miss,err = getMissing_ErrorInputs_index(len(string),                     auto.missing_num, auto.missing_prob, auto.error_num, auto.error_prob)
        
        string = inputs_to_trace_with_missing_err(string,miss,err)  
        
        res.append(string)
    return res
    
def getUnmatchingTraces(learn_id,test_range,auto,num_of_exp):
    valid = []
    invalid = []
    generatedAutomata = getGeneratedAutomataFile(learn_id)
    learned_auto_asp_file = getLogPath('learnedAutomata.lp',learn_id)
    
    for i in range(num_of_exp):
        ran_inputs = get_random_inputs(test_range)
        _,miss,err = getMissing_ErrorInputs_index(len(ran_inputs),                     auto.missing_num, auto.missing_prob, auto.error_num, auto.error_prob)
        
        missAndErrLog = getLogPath('miss&errorINdex',learn_id)
        if(len(miss)+len(err)>0):
            log_str = 'trace: {}\n miss:{} \n err: {}\n'.format(ran_inputs,miss,err)
            if(os.path.isfile(missAndErrLog)):
                append_to_file(missAndErrLog,log_str)
            write_to_file(missAndErrLog,log_str)
        
        trace = inputs_to_trace_with_missing_err(ran_inputs,miss,err)
        
        correct = check_trace_valid(trace,generatedAutomata)
        if(os.path.isfile(learned_auto_asp_file)):
            correct_learned = check_trace_valid(trace,learned_auto_asp_file)
            if(correct and (not correct_learned)):
                valid.append(trace)
            elif((not correct) and (correct_learned)):
                invalid.append(trace)
        else:
            if(correct):
                valid.append(trace)
            else:
                invalid.append(trace)
    
    return valid,invalid

def getUnmatchingTraces_checkValidFirst(learn_id,test_range,auto,num_of_exp):
    valid = []
    invalid = []
    generatedAutomata = getGeneratedAutomataFile(learn_id)
    learned_auto_asp_file = getLogPath('learnedAutomata.lp',learn_id)
    
    for i in range(num_of_exp):
        ran_inputs = get_random_inputs(test_range)
        _,miss,err = getMissing_ErrorInputs_index(len(ran_inputs),                     auto.missing_num, auto.missing_prob, auto.error_num, auto.error_prob)
        
        missAndErrLog = getLogPath('miss&errorINdex',learn_id)
        if(len(miss)+len(err)>0):
            log_str = 'trace: {}\n miss:{} \n err: {}\n'.format(ran_inputs,miss,err)
            if(os.path.isfile(missAndErrLog)):
                append_to_file(missAndErrLog,log_str)
            write_to_file(missAndErrLog,log_str)
        
        raw_trace = string_to_trace(ran_inputs)
        trace = inputs_to_trace_with_missing_err(ran_inputs,miss,err)
        
        correct = check_trace_valid(raw_trace,generatedAutomata)
        if(os.path.isfile(learned_auto_asp_file)):
            correct_learned = check_trace_valid(raw_trace,learned_auto_asp_file)
            if(correct and (not correct_learned)):
                valid.append(trace)
            elif((not correct) and (correct_learned)):
                invalid.append(trace)
        else:
            if(correct):
                valid.append(trace)
            else:
                invalid.append(trace)
    
    return valid,invalid

def traces_to_examples(valid_traces,invalid_traces, hasError =False):
    res = ''
    penalty = ''
    if(hasError):
        penalty = '@2'
        
    
    for v in range(len(valid_traces)):
        modified = '{'+valid_traces[v][:-1]+'}'
        res += "#pos(p{0},{{accept}},{{}},{1}).\n".format(str(v)+penalty,modified)
    for iv in range(len(invalid_traces)):
        modified = '{'+invalid_traces[iv][:-1]+'}'
        res += "#pos(n{0},{{}},{{accept}},{1}).\n".format(str(iv)+penalty,modified)
    return res

def learnAutomata_WithTeacher(auto,learn_id,test_range,test_num,learn_once=False,inductive=False):
    #file names
    graphFile = getLogPath('graph.png',learn_id)
    learningTask = getLogPath('learningTask.lp',learn_id)
    
    draw_graph(auto,auto.graph,graphFile)

    logAutomataEncoding(auto, [],[],learn_id)
        
    learned = False
    
    valid_trace = get_init_valid_trace(auto,learn_id)[:test_num]
    invalid_trace = []
    total_time = 0
    iteration = 0
    while(not learned):
        iteration += 1
        #return ["input(0,0).input(1,1).input(2,1).trace_length(3)","..."]
        #note in here, if contain missing or error,
        #make sure they are tested after modified the strings
        
#         new_v, new_inv = getUnmatchingTraces(learn_id,test_range,auto,test_num*2)
        new_v, new_inv = getUnmatchingTraces(learn_id,test_range,auto,test_num*2)
        
        valid_trace += new_v[:test_num]
        invalid_trace += new_inv[:test_num]
        
        exps = traces_to_examples(valid_trace,invalid_trace,auto.error_num!=0)    

        if(inductive):
            completeILASPInductiveEncoding(learningTask,exps,len(auto.states))
        else:
            completeIlaspEncoding(learningTask,exps,len(auto.states))
        
    
        learning_res = execute_ILASP(learningTask)
        learningSuccess, time, is_positive_exp, err_trace = checkLearningIsRight_ByILASP(auto,                                            learning_res,learn_id,auto.state_num)

        if(is_positive_exp):
            valid_trace.append(err_trace)
        else:
            invalid_trace.append(err_trace)
        if(learn_once):
            return learningSuccess,time, 1
        total_time += time
        learned = learningSuccess
        
    return learned, total_time, iteration


def getRegUnmatchingTraces(learn_id,test_range,reg_file,num_of_exp):
    valid = []
    invalid = []
    generatedAutomata = reg_file
    learned_auto_asp_file = getLogPath('learnedAutomata.lp',learn_id)
    
    for i in range(num_of_exp):
        ran_inputs = get_random_inputs(test_range)
        
        trace = inputs_to_trace_with_missing_err(ran_inputs,[],[])
        
        correct = check_trace_valid(trace,generatedAutomata)
        if(os.path.isfile(learned_auto_asp_file)):
            correct_learned = check_trace_valid(trace,learned_auto_asp_file)
            if(correct and (not correct_learned)):
                valid.append(trace)
            elif((not correct) and (correct_learned)):
                invalid.append(trace)
        else:
            if(correct):
                valid.append(trace)
            else:
                invalid.append(trace)
    
    return valid,invalid


def learn_RegAutomata_WithTeacher(auto_file,state_num,reg_limit,                                  learn_id,test_range,test_num,learn_once=False):
    #file names
    
    learningTask = getLogPath('learningTask.lp',learn_id)
    
    learned = False
    
    valid_trace = []
    invalid_trace = []
    total_time = 0
    iteration = 0
    
    while(len(valid_trace) ==0):
        new_v, new_inv = getRegUnmatchingTraces(learn_id,test_range,auto_file,20)
        valid_trace += new_v
    while(not learned):
        iteration += 1
        
        new_v, new_inv = getRegUnmatchingTraces(learn_id,test_range,auto_file,test_num)
        valid_trace += new_v
        invalid_trace += new_inv
        
        exps = traces_to_examples(valid_trace,invalid_trace)    

        completeRegIlaspEncoding(learningTask,exps,state_num,reg_limit)
        
        learning_res = execute_ILASP(learningTask)
        learningSuccess, time, err_trace =  checkRegLearningIsRight_test(auto_file,                                            test_range,state_num, reg_limit,learning_res,learn_id)
        if(err_trace != ''):
            invalid_trace.append(err_trace)
        if(learn_once):
            return learningSuccess,time
        total_time += time
        learned = learningSuccess
   
    return learned, total_time, iteration


def learn_RegAutomata_FromDFA_teacher(auto,state_num,reg_limit,                                  learn_id,test_range,test_num,learn_once=False):
    #file names
    graphFile = getLogPath('graph.png',learn_id)
    learningTask = getLogPath('reglearningTask.lp',learn_id)
    
    draw_graph(auto,auto.graph,graphFile)

    logAutomataEncoding(auto, [],[],learn_id)
        
    learned = False
    
    valid_trace = get_init_valid_trace(auto,learn_id)[:test_num]
    invalid_trace = []
    total_time = 0
    iteration = 0
    while(not learned):
        iteration += 1
        new_v, new_inv = getUnmatchingTraces(learn_id,test_range,auto,test_num*2)
        valid_trace += new_v[:test_num]
        invalid_trace += new_inv[:test_num]
        
        exps = traces_to_examples(valid_trace,invalid_trace,auto.error_num!=0)    

        completeRegIlaspEncoding(learningTask,exps,state_num,reg_limit)
        
        learning_res = execute_ILASP(learningTask)
        learningSuccess, time, is_positive_exp, err_trace = checkLearningIsRight_ByTest(auto,                                            learning_res,learn_id)

        if(is_positive_exp):
            valid_trace.append(err_trace)
        else:
            invalid_trace.append(err_trace)
        if(learn_once):
            return learningSuccess,time, 1
        total_time += time
        learned = learningSuccess
        
    return learned, total_time, iteration
    


# In[1100]:

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


draw_colors = ['r','b','y','g','c','m','k']

def produceMissingErrorGraph():

    clearLogDir()
    patches = []

    draw_style = '-o'

    correct_num = 0

    example_number = [10,30]
    missing_probs = [0.1,0.2]
    # missing_probs = [0]
    # example_number = [10,]
    num_states = [3,4,5]
    run_total = 2

    record = {}

    if(len(missing_probs) > len(draw_colors)):
        print "faillingsjfkls!!!!!"
    plt.clf()
    indicate_process = len(example_number)*len(missing_probs)*len(num_states)* run_total
    
    #only count successfully learned time?
    time_rec = []
    for mp in range(len(missing_probs)):
        patches.append(mpatches.Patch(color=draw_colors[mp][0],                                      label='Noisy Level '+ str(missing_probs[mp])))
        accuracy = []
        total_time = 0
        total_learned = 0
        for en in example_number:
            clearLogDir()
            correct_num = 0
            for st_num in num_states:
                for i in range(run_total):
                    print '{} left!',indicate_process
                    indicate_process -=1
                    auto = Automata(st_num,0,1)
#                     auto.setMissingInput(missing_probs[mp],1)
                    auto.setErrorInput(missing_probs[mp],1)
                    auto.generate_automata()
                    correct, time,_ = learnAutomata_WithTeacher(auto,st_num*100+i,1000,en,learn_once=True)
                    if(correct):
                        correct_num += 1
                        total_time += time
            accuracy.append(correct_num*100/(len(num_states)* run_total) ) 
            total_learned += correct_num
        time_rec.append(total_time/total_learned)
        plt.plot(example_number,accuracy,draw_colors[mp]+draw_style)
    plt.xlabel('Example number')
    plt.ylabel('Accuracy')
    plt.legend(handles=patches)
    plt.show() 
    plot_file = 'miss_different_percentage_prob_1_state3to5_run30.png'
    plt.savefig(plot_file)
    
    plt.clf()
    plt.plot(missing_probs,time_rec,'-ob')
    plt.xlabel('Noisy level')
    plt.ylabel('Average learning time')
    plot_file = 'avgLearnTimeRun_state3~5AgainstNoisyLevel.png'
    plt.savefig(plot_file)
    
    print 'finished !!!'


# In[1101]:

def getAutomataWithEdgeNum(st_num,edge_num):
    auto = Automata(st_num,0,1)
    auto.generate_automata()
    while(len(list(auto.graph.edges())) != edge_num ):
        auto = Automata(st_num,0,1)
        auto.generate_automata()
    return auto


def plotLearnTimeEdgeGraphWithFunc(st_edge_dict,func,name,patches):
    plt.clf()
    cl = draw_colors[:]
    for st_num in st_edge_dict.keys():
        edge_records = st_edge_dict[st_num]
        x = edge_records.keys()
        y = []
        err = []
        for k in x:
            rec = edge_records[k]
            y.append(func(rec))
            err.append(calcSTD(rec))
#         plt.errorbar(num_states,log_avg_learn,fmt='b'+draw_style,yerr=[logErr,logErr])
        plt.errorbar(x,y,fmt = '-o' + cl.pop(0),yerr = [err,err])
    plt.xlabel('Number of edges')
    plt.ylabel('learning time')
    plt.legend(handles = patches)
    plt.savefig(name)
    
def avg(n):
    return sum(n)/len(n)

def produceLearnTimeAgainstEdge(run_total):
    
    print 'produce for learn against edges '
    draw_style = '-o'

    clearLogDir()

    num_states = [4,5,6,7,8,9]
    
    assert(len(num_states) <= len(draw_colors))
    
    st_edge_dict = {}
    for n in num_states:
        print 'state number:', n
        record = {}
        for l in range(n-1,n+1):
            learn_time = []
            for i in range(run_total):
                auto = getAutomataWithEdgeNum(n,l)
                correct, time,_ = learnAutomata_WithTeacher(auto,n*100000+l*100+i,1000,10*n,learn_once=False)
                learn_time.append(time)
            record[l] = learn_time
        st_edge_dict[n] = record
        
# st_edge_dict:
#     {
#         st_number:{
#             edge_number:[]
#             edge_number:[]
#             ...
#         }
#         st_number:{
#             ...
#         }
        
#     }
        
    
    patches = []
    for i in range(len(num_states)):
        patches.append(mpatches.Patch(color=draw_colors[i],label='state number {}'.format(num_states[i]) ))
    
    plotLearnTimeEdgeGraphWithFunc(st_edge_dict,avg,'learn_time_against_edgesAvg_st5',patches)
    
    print 'finished !!!'


# In[1102]:


def produceLearnTimeAgainstState(run_total):
    
    print 'produce for learn against state'
    draw_style = '-o'

    clearLogDir()

    num_states = [2,3,4,5]

    record = {}
    for st_num in num_states:
        print 'for state number', st_num
        for i in range(run_total):
            auto = Automata(st_num,0,1)
            auto.generate_automata()
            num_edges = len(list( auto.graph.edges()))
            correct, time,iteration = learnAutomata_WithTeacher(auto,st_num*1000+i,1000,10,learn_once=False)
            correct, time_t,iteration = learnAutomata_WithTeacher_tmp(auto,st_num*1000+i,1000,10,learn_once=False)
            
            add_rec = (time_t-time)/time
            if(not st_num in record):
                record[st_num] = [add_rec]
            else:
                record[st_num].append(add_rec)

    err_std = []
    avg_learn=[]
    for k in record.keys():
        rec = record[k]
        avg = sum(rec) / len(rec)

        err_std.append(calcSTD(rec))
        avg_learn.append(avg)


#     plt.clf()
#     plt.xlabel('Number of States')
#     plt.ylabel('Log_e(Avg Learning time in seconds)')
#     plt.gca().axes.get_xaxis().set_ticks(num_states)
    
#     log_avg_learn = mapLog(avg_learn)
#     logErr = mapLog(err_std)
    
#     plt.errorbar(num_states,log_avg_learn,fmt='b'+draw_style,yerr=[logErr,logErr])
#     plot_file = 'learning_LOG_time_states_avg_errbar_567.png'
#     plt.savefig(plot_file)
    
    
    plt.clf()
    plt.xlabel('Number of States')
    plt.ylabel('Relative time difference')
    plt.gca().axes.get_xaxis().set_ticks(num_states)
    plt.plot(num_states,avg_learn,'r'+draw_style)
    plot_file = 'LearTime_state_fixedStatenumLimit_7COmpare.png'
    plt.savefig(plot_file)
    
    
    print 'finished !!!'


# In[1103]:

def plotDeltaNumDistribution():
    rec = {}
    for i in range(10000):
        auto = Automata(8,0,1)
        auto.generate_automata()
        delta_num = len(list(auto.graph.edges()))
        if delta_num not in rec:
            rec[delta_num] = 1
        else:
            rec[delta_num] += 1

    x = rec.keys()
    y = []
    for k in x:
        y.append(rec[k])

    plt.clf()
    plt.xlabel('Number of edges')
    plt.ylabel('Count')
    plt.plot(x,y,'-ob')
    plot_file = 'automatastate8DeltaNumberDistribution.png'
    plt.savefig(plot_file)



# In[1104]:

def pickRegisterAutomata(st_num,reg_num):
    dir_name =  './useIlasp/RegAutomataPool/regAutomataState_{}Reg_{}'.format(st_num,reg_num)
    if(not os.path.isdir(dir_name)):
        print 'no such automata when pick reg automata'
        return ''
    max_id = len(os.listdir(dir_name)) / 2 -1
    ran_id = random.randint(0,max_id)
    
    file_name = os.path.join(dir_name,'id_{}.lp'.format(ran_id))
    return file_name

def replaceFileExt(name, repl):
    return name[:name.index('.',1)] + repl

def plotRegAutomataLearnTime(run_total):
    num_state = [2,3,4]
    reg_limit = [2,3,4]
    clearLogDir()

    plt.clf()
    patches = []
    for rl_idx in range(len(reg_limit)):
        rl = reg_limit[rl_idx]
        patches.append(mpatches.Patch(color=draw_colors[rl_idx],label='Register limit:{}'.format(rl) ))
        record = {}
        to_plot = []
        for st_num in num_state:
            rec = []
            for i in range(run_total):
                learn_id = 10000*st_num+100*rl+i
                picked = pickRegisterAutomata(st_num,rl)
                graphFile = replaceFileExt(picked,'.png')
                
                generatedAutoFile = getGeneratedAutomataFile(learn_id)
                generatedGraphFile = replaceFileExt(generatedAutoFile,'.png')
                
                copyfile(picked,generatedAutoFile)
                copyfile(graphFile,generatedGraphFile)
                correct, time, ite = learn_RegAutomata_WithTeacher(                                    picked,st_num,rl,learn_id,                                    1000,10*st_num,learn_once=False)
                rec.append(time)
            longest = max(rec)
            to_plot.append(longest)
            print 'id: ', 10000*st_num+100*rl+rec.index(longest) 
            record[st_num] = rec
        plt.plot(num_state,to_plot,'-o'+draw_colors[rl_idx])
    plt.legend(handles = patches)
    plt.xlabel('Number of state')
    plt.ylabel('Learning time')
    figFile = 'Register automatat learning time.png'
    plt.savefig(figFile)
        


# In[1105]:


def compareEncoding(num_states,run_total):
    
    print 'produce for learn against state'
    draw_style = '-o'

    clearLogDir()
    record = {}
    for st_num in num_states:
        print 'for state number', st_num
        for i in range(run_total):
            auto = Automata(st_num,0,1)
            auto.generate_automata()
            num_edges = len(list( auto.graph.edges()))
            correct, time,iteration = learnAutomata_WithTeacher(auto,st_num*1000+i,1000,10,False,False)

            _, time_ind,_ = learnAutomata_WithTeacher(auto,st_num*1000+i,1000,10,False,True)
            if(not st_num in record):
                record[st_num] = [time_ind - time]
            else:
                record[st_num].append(time_ind-time)

    err_std = []
    avg_learn=[]
    for k in record.keys():
        rec = record[k]
        avg_learn.append(sum(rec))

    
    plt.clf()
    plt.xlabel('Number of States')
    plt.ylabel('Avg diff learn time in seconds')
    plt.gca().axes.get_xaxis().set_ticks(num_states)
    plt.plot(num_states,avg_learn,'r'+draw_style)
    plot_file = 'learn Time Diff.png'
    plt.savefig(plot_file)
    

    print 'finished !!!'


# In[1106]:

# #nums = [1,2,4,5,7,8,10]
# # nums = [12,15,18,21,24]

# # for n in nums:
# #     print str(bin(n))[2:]
    
# password = '2-1-4'
# pass_bin = '010010'

# pos_str_p = ['00','1','011','0101','01000','010011']
# pos_str =[]
# for p in pos_str_p:
#     pos_str.append(string_to_trace(p+pass_bin))
# neg_str = []

# for i in range(1000):
#     ran = get_random_inputs(10000)
#     if(not ran in pos_str_p):
#         neg_str.append(string_to_trace(ran))
        
                
# exps = traces_to_examples(pos_str,neg_str)
# completeIlaspEncoding('handCraftPasswordResetOnce_214.lp',exps,7)
    


# In[1107]:

# pos_str_p = range(0,10000,2)
# pos_str =[]

# for p in random.sample(pos_str_p,100):
#     pos_str.append(string_to_trace(str(bin(p))[2:] ))
# neg_str = []

# for i in range(1,1000,2):
#     neg_str.append(string_to_trace(str(bin(i))[2:]))
        
                
# exps = traces_to_examples(pos_str,neg_str)
# completeIlaspEncoding('2Dividable.lp',exps,5)
    


# In[1108]:

# results = 'delta(state0,V0,state2,1) :- input(_,V0).\
# delta(state0,V0,state3,0) :- input(_,V0).\
# delta(state2,V0,state3,1) :- input(_,V0).\
# delta(state2,V0,state1,0) :- input(_,V0).\
# delta(state3,V0,state3,0) :- input(_,V0).\
# delta(state1,V0,state2,0) :- input(_,V0).\
# delta(state3,V0,state2,1) :- input(_,V0).\
# delta(state1,V0,state1,1) :- input(_,V0).'.split('.')

    
# def getEdgeFromText(txt):
#     search = re.search(r"delta\(state(\d),V0,state(\d),(\d)\) :- input\(_,V0\)",txt)
#     return map(int,search.groups())
    
    
# graph = nx.DiGraph() 
# labels = {}
# edges = []
# for idx in range(len(results)-1):
#     edge = getEdgeFromText(results[idx])
#     graph.add_edge(edge[0],edge[1])
#     edges.append((edge[0],edge[1]))
#     labels[(edge[0],edge[1])] = edge[2]

# g = nx.nx_pydot.to_pydot(graph)
# # print labels
    
# for idx in range(len(results)-1):
#     edge = g.get_edges()[idx]
#     delta = (int(edge.obj_dict['points'][0]),int(edge.obj_dict['points'][1]))
#     edge.obj_dict['attributes']['label'] = labels[delta]
    

# g.write_png('3devidebale.png')


# In[1109]:

def getUsedExample(auto_id):
    cache_name = 'learningTask.lp.cache'
    file_name = getLogPath(cache_name,auto_id)
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def getModifiedDeltas(raw,auto_id):
    res = []
    insert_pos = 6
    for r in raw:
        if('delta' in r):
            res.append(r[:insert_pos] + str(auto_id) + ',' + r[insert_pos:])
    return ''.join(res)
    
def getConditionCode(cond):
    if(cond == [0,0]):
        return 0
    elif(cond ==[1,1]):
        return 1
    elif(cond ==[0,1]):
        return 2    

def getModifiedDeltaFromAuto(auto):
    res = []
    template = "delta(0,state{},C,state{},{}):- input(_,C).\n"
    for d in range(len(auto.deltas)):
        (fr,to) = auto.deltas[d]
        cond = getConditionCode(auto.conditions[d])
        
        filled = template.format(fr,to,cond)
        res.append(filled)
        
    return ''.join(res)


def ILASPExpStateInfo(o_stnum,limit_stnum):
    res = ''
    for i in range(limit_stnum):
        res += 'state(state{}).\n'.format(i)
    res += 'accept(0) :- st(0,T,state{}), trace_length(T).\n'.format(o_stnum-1)
    res += 'accept(1) :- st(1,T,state{}), trace_length(T).\n'.format(limit_stnum-1)
    
    return res
    
    
def ILASPExpLearnInfo(limit):
    res = ''
    for i in range(limit):
        res += '#constant(step,{}).\n'.format(i)
        res += '#constant(length,{}).\n'.format(i+1)
    return res

def checkLearningIsRight_ByILASP(auto,learningRes,learn_id, st_num_limit):
    a = learningRes.stdout.readline()
    output = []
    while(a):
        if ("UNSATISFIABLE" in a):
            print 'learning is unsatisiable!, valid edges:', len(getValidEdges(auto))
            print 'id:', learn_id
            return False,0
        output.append(a)
        a = learningRes.stdout.readline()
    
    learned_deltas = getModifiedDeltas(output,1)
    time = getTimeFromLearning(output[-2])
    origin_deltas = getModifiedDeltaFromAuto(auto)

    stateInfo = ILASPExpStateInfo(auto.state_num,st_num_limit)
    learnInfo = ILASPExpLearnInfo(20)
    
    template_file = 'useIlasp/learnCounterExpTemplate.lp'
    find_exp_file = getLogPath('findCounterTMP.lp',learn_id)
    copyfile(template_file,find_exp_file)
    
    append_to_file(find_exp_file, learned_deltas)
    append_to_file(find_exp_file, origin_deltas)
    append_to_file(find_exp_file, stateInfo)
    append_to_file(find_exp_file, learnInfo)
    

    find_exp = execute_ILASP(find_exp_file)
    a = find_exp.stdout.readline()
    trace = ''
    while(a):
        if ("UNSATISFIABLE" in a):
            return True, time, True, ''
        if('(' in a):
            trace += a
        a = find_exp.stdout.readline()
    
    trace = ''.join(e for e in trace if e!='\n') + '\n'
    generated_auto = getGeneratedAutomataFile(learn_id)
    if(check_trace_valid(trace,generated_auto)):
#       Return trace as negative example
        return False, time, True, trace
    else:
        return False, time, False, trace
    

def produceUsedExampleAgainstState(run_total):
    
    print 'produce for learn against state'
    draw_style = '-o'

    clearLogDir()

    num_states = [2,3]

    record = {}
    for st_num in num_states:
        print 'for state number', st_num
        for i in range(run_total):
            auto = Automata(st_num,0,1)
            auto.generate_automata()
            num_edges = len(list( auto.graph.edges()))
            correct, time,iteration = learnAutomata_WithTeacher(auto,st_num*1000+i,1000,1,learn_once=False)
            used_exp = getUsedExample(st_num*1000+i)
            if(not st_num in record):
                record[st_num] = [used_exp-iteration]
            else:
                record[st_num].append(used_exp-iteration)

    err_std = []
    avg_learn=[]
    for k in record.keys():
        rec = record[k]
        avg = sum(rec) / len(rec)

        err_std.append(calcSTD(rec))
        avg_learn.append(avg)


    plt.clf()
    plt.xlabel('Number of States')
    plt.ylabel('used example')
    plt.gca().axes.get_xaxis().set_ticks(num_states)
    
    
    plt.errorbar(num_states,avg_learn,fmt='b'+draw_style,yerr=[err_std,err_std])
    plot_file = 'exampleUsedAgainstState.png'
    plt.savefig(plot_file)
    
    
    print 'finished !!!'


# In[ ]:

def LearnRegFromDFA(run_total):
    
    print 'produce for learn against state'
    draw_style = '-o'

    clearLogDir()

    num_states = [2,3,4,5,6,7]

    record = {}
    for st_num in num_states:
        print 'for state number', st_num
        for i in range(run_total):
            auto = Automata(st_num,0,1)
            auto.generate_automata()
            num_edges = len(list( auto.graph.edges()))
            learn_id = st_num*1000+i
#             _, time, _ = learnAutomata_WithTeacher(auto,learn_id,1000,10,learn_once=False)
            _, time_reg, _ = learn_RegAutomata_FromDFA_teacher(auto,st_num+1,3,                                  learn_id,1000,10,learn_once=False)
            
            if(not st_num in record):
                record[st_num] = [time_reg]
            else:
                record[st_num].append(time_reg)

    err_std = []
    avg_learn=[]
    for k in record.keys():
        rec = record[k]
        avg = sum(rec) / len(rec)

        err_std.append(calcSTD(rec))
        avg_learn.append(avg)
    
    plt.clf()
    plt.xlabel('Number of States')
    plt.ylabel('Avg learning time difference in seconds')
    plt.gca().axes.get_xaxis().set_ticks(num_states)
    plt.plot(num_states,avg_learn,'r'+draw_style)
    plot_file = 'LearnRegautofromDFA.png'
    plt.savefig(plot_file)
    
    
    print 'finished !!!'


# In[ ]:

produceMissingErrorGraph()


# In[ ]:



