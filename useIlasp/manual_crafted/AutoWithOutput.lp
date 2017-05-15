condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 

state(state0).
state(state1).
state(state2).
st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,C,TO,ID,_),condition(C,ID),input(T,C).


out(T+1,V):- st(T,ST), delta(ST,_,_,_,V),input(T,_),state(ST).

accept :- st(T,state4),trace_length(T).
:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,C,TO,ID), delta(FROM,C,TO,ID2), ID!=ID2.

#show st/2.
#show out/2.

input(0,0).input(1,1).input(2,1).trace_length(3).

delta(state0,C,state1,0,0):- input(_,C).
delta(state1,C,state2,1,1):- input(_,C).
delta(state2,C,state2,1,2):- input(_,C).
