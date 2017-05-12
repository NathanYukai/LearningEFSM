condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 

% Time, state
st(0,state0,0).
origin(0).
final(2).

%state, time, input,state, conditionID, REG_condition, REG_Value_NEW
delta(state0,C,state0,0,0,1):- input(_,C).
delta(state0,C,state0,0,1,1):- input(_,C).
delta(state0,C,state0,0,2,1):- input(_,C).
delta(state0,C,state1,0,3,0):- input(_,C).


state(state0).
state(state1).
st(T+1,TO,V-1):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,0),condition(C,ID).
st(T+1,TO,V+1):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,1),condition(C,ID).
st(T+1,TO,R):- st(T,FROM,_),state(FROM),state(TO),delta(FROM,C,TO,ID,_,2),condition(C,ID),final(R).

input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).

:- not st(T,state1,F),trace_length(T),final(F).
:- st(T,ST),st(T,ST2),ST!=ST2.

#show st/3.
