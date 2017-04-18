

condition(C,0):- C >= 7,C <= 10, input(_,C). 
condition(C,1):- C >= 2,C <= 5, input(_,C). 
condition(C,2):- C >= 3,C <= 4, input(_,C). 
condition(C,3):- C >= 5,C <= 7, input(_,C). 
condition(C,4):- C >= 7,C <= 10, input(_,C). 
condition(C,5):- C >= 4,C <= 10, input(_,C). 
condition(C,6):- C >= 0,C <= 3, input(_,C). 

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).

:- not st(T,state4),trace_length(T).

delta(state0,T,C,state3,0):-input(T,C).
delta(state1,T,C,state3,1):-input(T,C).
delta(state2,T,C,state1,2):-input(T,C).
delta(state2,T,C,state3,3):-input(T,C).
delta(state3,T,C,state4,4):-input(T,C).
delta(state4,T,C,state1,5):-input(T,C).
delta(state4,T,C,state2,6):-input(T,C).

input(0,7).input(1,7).input(2,3).input(3,5).input(4,10).input(5,3).input(6,5).input(7,10).trace_length(8).
#show st/2.
