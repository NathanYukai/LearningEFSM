in_limit(10).

state(state0).state(state1).state(state2).state(state3).state(state4).

condition0(C):-5 <= C, C <= 5, input(C). 
condition1(C):-3 <= C, C <= 3, input(C). 
condition2(C):-7 <= C, C <= 10, input(C). 
condition3(C):-4 <= C, C <= 10, input(C). 
condition4(C):-1 <= C, C <= 1, input(C). 

delta(state0,C,state1):-condition0(C).
delta(state0,C,state3):-condition1(C).
delta(state3,C,state2):-condition2(C).
delta(state2,C,state4):-condition3(C).
delta(state2,C,state2):-condition4(C).

st(0,state0).
st(T,Y):- st(T-1,S),state(S),state(Y),delta(S,T,Y),input(T).

