in_limit(10). 
state(state0).state(state1).state(state2).state(state3).state(state4).

condition0(T,C):-6 <= C, C <= 9, input(T,C). 
condition1(T,C):-8 <= C, C <= 8, input(T,C). 
condition2(T,C):-10 <= C, C <= 10, input(T,C). 
condition3(T,C):-8 <= C, C <= 10, input(T,C). 
condition4(T,C):-0 <= C, C <= 1, input(T,C). 

delta(state0,T,C,state2):-condition0(T,C).
delta(state1,T,C,state1):-condition1(T,C).
delta(state1,T,C,state4):-condition2(T,C).
delta(state2,T,C,state1):-condition3(T,C).
delta(state2,T,C,state3):-condition4(T,C).

st(0,state0).
st(T,Y):- st(T-1,S),state(S),state(Y),delta(S,T,C,Y).

state(error).
delta(state0,T,C,error):-input(T,C).
delta(state1,T,C,error):-not condition0(T,C),input(T,C).
delta(state2,T,C,error):-not condition0(T,C),input(T,C).
delta(state3,T,C,error):-input(T,C).
:- not st(T,state4),trace_length(T).
#hide.
#show st/2.
trace_length(3).
 input(1,6).input(2,9).input(3,10).
