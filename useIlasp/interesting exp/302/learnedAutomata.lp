condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).
condition(C,2):- C >= 0,C <= 1, input(_,C).


%0{input(T,0),input(T,1)}1:- time(T).
%time(T-1):- trace_length(T).
%time(T-1):- time(T), T>=1.


st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).


:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,T,C,TO,ID), delta(FROM,T,C,TO,ID2), ID!=ID2.

delta(state2,V0,V1,state1,2) :- input(V0,V1).

delta(state0,V0,V1,state1,2) :- input(V0,V1).

delta(state1,V0,V1,state2,2) :- input(V0,V1).

state(state0).
state(state1).
state(state2).
:- not st(T,state2), trace_length(T).
