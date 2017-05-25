condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).
condition(C,2):- C >= 0,C <= 1, input(_,C).


1{input(T,0);input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.

st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,C,TO,ID),condition(C,ID),input(T,C).


:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,C,TO,ID), delta(FROM,C,TO,ID2), ID!=ID2.

delta(state6,V0,state5,2) :- input(_,V0).

delta(state5,V0,state7,1) :- input(_,V0).

delta(state5,V0,state3,0) :- input(_,V0).

delta(state1,V0,state6,1) :- input(_,V0).

delta(state3,V0,state1,1) :- input(_,V0).

delta(state2,V0,state6,1) :- input(_,V0).

delta(state0,V0,state2,0) :- input(_,V0).

delta(state1,V0,state5,0) :- input(_,V0).

delta(state0,V0,state5,1) :- input(_,V0).

delta(state4,V0,state6,0) :- input(_,V0).

delta(state2,V0,state0,0) :- input(_,V0).

delta(state3,V0,state4,0) :- input(_,V0).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
state(state5).
state(state6).
state(state7).
:- not st(T,state7), trace_length(T).
