
condition(C,2):- C >= 0,C <= 1, input(_,C).
condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).

:- not st(T,state4),trace_length(T).
:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,T,C,TO,ID), delta(FROM,T,C,TO,ID2), ID!=ID2.

delta(state2,V0,V1,state0,0) :- input(V0,V1).
delta(state4,V0,V1,state2,1) :- input(V0,V1).
delta(state1,V0,V1,state4,1) :- input(V0,V1).
delta(state1,V0,V1,state3,0) :- input(V0,V1).
delta(state3,V0,V1,state4,0) :- input(V0,V1).
delta(state0,V0,V1,state1,2) :- input(V0,V1).

%for P6
delta(state2,V0,V1,state1,1) :- input(V0,V1).
%for p3, p5
delta(state4,V0,V1,state0,0) :- input(V0,V1).

%p6
%input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).trace_length(5).

%p3, p5
input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).
#show st/2.
