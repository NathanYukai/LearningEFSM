
# coding: utf-8

# In[66]:

#utility functions
import subprocess,random

def write_to_file(file_name,trace):
    with open(file_name,'w') as f:
        f.write(trace)


#parameters are file names
def execute_clingo(trace, automata):
    script = "./clingo " + trace + " " + automata
    process = subprocess.Popen(script,stdout=subprocess.PIPE,shell=True)
    return process

#only works for integers, take input limit, number of conditions needed
# return list of conditions, guaranteed to be mutually exclusive
def generate_conditions(lower_bound,upper_bound,num):
    if(upper_bound - lower_bound +1 < num):
        print "not so possible to generate so many conditions"
        return []
    results = []
    
    #list of ranges available : [[l,u],[l,u],[l,u]]
    ranges = [[lower_bound,upper_bound]]
    while(len(results)<num):
        #choose a range from list of continuous ranges
        chose_range = ranges[random.randint(0,len(ranges)-1)]
        low = chose_range[0]
        upp = chose_range[1]
        l = random.randint(low,upp)
        u = random.randint(l,upp)
        results.append([l,u])
        
        #update ranges
        left_upper = l-1
        right_lower = u+1
        ranges.remove([low,upp])
        if(left_upper >= low):
            ranges.append([low,left_upper])
        if(right_lower <= upp):
            ranges.append([right_lower,upp])
        
        #reset if requirement can't be met
        possible_ranges = 0
        for r in ranges:
            possible_ranges += r[1]-r[0]
        if(possible_ranges+len(results) < num):
            results = []
            ranges = [[lower_bound,upper_bound]]
    
    return results

#need better definition
def check_all_included(connections,final_state):
    for s in range(final_state):
        s_included = False
        for c in connections:
            if( s in c):
                s_included = True
        if not s_included:
            return False
    return True

#mutual exclusive condition needed for the deltas
# return a list of tuple , (from state, number of MX)
# e.g. [(0,2)] means state0 need MX for two state
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
        


# In[87]:

import random
#ASSUMPTIONS: only one input, represented by number


    
cond_param = 'C'
class Automata:

    #assume first is the initial state and 
    #the last one is the accepting state
    
    states = []
    in_low = 0
    minStates = 2
    
    # to be improved if they works perfectly
    maxCondition = 3
    maxStates = 5
    in_upp = 10
    
    
    def __init__(self, states, in_low, in_upp):
        if (len(states)<self.minStates):
            print "states number not enough, will use default random states"
        self.states = states
        self.in_low = in_low
        self.in_upp = in_upp
        
    def generate_automata(self, file_name):
        output = ""
        
        if((self.states == []) or (len(self.states) < self.minStates)):
            num_states = random.randint(self.minStates,self.maxStates+1)
            for i in range(num_states):
                self.states.append("state"+str(i))
                
        #limiting inputs
        output += "in_limit(" + str(self.in_upp) + "). \n"
        
        #states generation
        for s in self.states:
            output += "state(" + s + ")."
        output += '\n'
        
        output += '\n'
        #condition generation
#         condition_num = random.randint(0,self.maxCondition)
#         for i in range(condition_num):
#             output += "condition"+ str(i+1) + "("+cond_param+"):-"
#             condition_complex = random.randint(1,2)
#             ranges = generate_conditions(self.in_low,self.in_upp,condition_complex)
#             for r in ranges:
#                 output +=  str(r[0])+" <= "+cond_param+"," + " "+cond_param+" <= " + str(r[1]) + ", "
#             output += "input("+cond_param+"). \n"
        
        #delta generation , list of tuples
        deltas = []
        final_state = len(self.states)-1
        #select an state that is not initial state
        init_c = random.randint(1,final_state) 
        deltas.append((0,init_c))
        # state that is not accpt state
        acpt_c = random.randint(0,final_state-1) 
        deltas.append((acpt_c,final_state))
        #add connection between above two states, 
        #unless they are same or, initial state goes to accpt directly
        if(init_c != acpt_c and not(init_c == final_state and acpt_c == 0)):
            deltas.append((init_c,acpt_c))
            
        #After there is a valid path from init to accpt
        #make sure there is no state not connected
        while(not check_all_included(deltas,final_state)):
            from_state = random.randint(0,final_state-2)
            to_state = random.randint(1,final_state-1)
            deltas.append((from_state,to_state))                   
        
        deltas =  list(set(deltas))
        
        #generate conditions along with deltas
        mx_needed = mutual_required(deltas)
        #index corresponding to deltas
        conditions = [[]]*len(deltas)
        for mx in mx_needed:
            #generate that many conditions, can add a random extra number
            # but then need to pick mx[1] out of them, TODO later
            mx_conditions = generate_conditions(self.in_low,self.in_upp,mx[1])
            for d in range(len(deltas)):
                if (deltas[d][0] == mx[0]) :
                    if(len(mx_conditions) == 0):
                        print "something wrong in calculating MX?"
                    conditions[d] = mx_conditions[0]
                    del mx_conditions[0]
               
        #fill up non-mx required conditions
        #TODO add more complex conditions
        for c in range(len(conditions)):
            if (conditions[c]==[]):
                conditions[c] = generate_conditions(self.in_low,self.in_upp,1)[0]
#         print deltas
#         print conditions
        
        #output conditions
        for c in range(len(conditions)):
            output += "condition"+ str(c) + "("+cond_param+"):-"
            output +=  str(conditions[c][0])+" <= "+cond_param+", "
            output += cond_param+" <= " + str(conditions[c][1]) + ", "
            output += "input("+cond_param+"). \n"
            
        output += '\n'
        for d in range(len(deltas)):
            from_st = deltas[d][0]
            to_st = deltas[d][1]
            mx_conditions = []
            tmpstr_ = "delta("+ self.states[from_st]+ "," + cond_param + ","+ self.states[to_st]+ "):-"
            output +=  tmpstr_+ "condition"+ str(d) +"("+ cond_param +").\n"
        
        
        output += '\n'
    
        output += "st(" + str(self.in_low) + "," + self.states[0]+ ").\n"
        
        output += "st(T,Y):- st(T-1,S),state(S),state(Y),delta(S,C,Y).\n"

        return output
                       
    
            
auto = Automata(["state0","state1","state2","state3","state4"],0,10)
print auto.generate_automata('')


# In[164]:

import os
import random
#test if the trace is acceptable or not,
#automata is the file name of automata in ASP
def verify(trace, automata_file):

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
        if ("SATISFIABLE" in a):
            result = True
    
    os.remove(tmp_file)
    print output
    return result

# set of possible inputs, outputs and
# and the max length of trace
def generate_trace(inputs,outputs,length):
    trace = ""
    inputs_example = "time"
    
    for i in range(length):
        trace += inputs_example + "("+str(i)+")." 
    return trace 

automata_file = './asp_automatas/simple_alarm.lp'
print verify(generate_trace(1,0,5),automata_file)


# In[60]:

[[]]*10


# In[ ]:



