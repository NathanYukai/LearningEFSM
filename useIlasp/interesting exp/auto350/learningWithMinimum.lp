condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).
condition(C,2):- C >= 0,C <= 1, input(_,C).


1{input(T,0);input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.

%1{error_at(X):X<T}1:- trace_length(T), X<T.
%input(T,X):- pos_input(T,X), not error_at(T).
%input(T,1-X):- pos_input(T,X), error_at(T).

st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).


:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,T,C,TO,ID), delta(FROM,T,C,TO,ID2), ID!=ID2.

#pos(p0,{st(0,state0),st(1,state1)},{},{input(0,0).trace_length(2).}).
#pos(p1,{st(0,state0),st(1,state2)},{},{input(0,1).trace_length(1).}).
#pos(p2,{st(1,state1),st(2,state2)},{},{input(1,1).trace_length(2).}).
#pos(p3,{st(1,state2),st(2,state0)},{},{input(1,0).trace_length(3).}).
2 ~ delta(state0,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state0,1) :- input(V0,V1).
9 ~ delta(state0,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state1,1) :- input(V0,V1).
9 ~ delta(state0,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state2,1) :- input(V0,V1).
9 ~ delta(state0,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state0,1) :- input(V0,V1).
9 ~ delta(state1,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state1,1) :- input(V0,V1).
9 ~ delta(state1,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state2,1) :- input(V0,V1).
9 ~ delta(state1,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state0,1) :- input(V0,V1).
9 ~ delta(state2,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state1,1) :- input(V0,V1).
9 ~ delta(state2,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state2,1) :- input(V0,V1).
9 ~ delta(state2,V0,V1,state2,2) :- input(V0,V1).

state(state0).
state(state1).
state(state2).
:- not st(T,state2),trace_length(T).
#max_penalty(15).
