#max_penalty(300).
condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).
condition(C,2):- C >= 0,C <= 1, input(_,C).


0{input(T,0),input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.


st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).


:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,T,C,TO,ID), delta(FROM,T,C,TO,ID2), ID!=ID2.

#pos(p_extra_0,{st(0,state0),st(1,state1)},{},{input(0,1).trace_length(5).}).
#pos(p0,{st(0,state0),st(1,state1)},{},{input(0,0).trace_length(5).}).
#pos(p1,{st(1,state1),st(2,state4)},{},{input(1,1).trace_length(5).}).
#pos(p2,{st(4,state2),st(5,state5)},{},{input(4,1).trace_length(5).}).
#pos(p3,{st(3,state3),st(4,state2)},{},{input(3,0).trace_length(5).}).
#pos(p4,{st(2,state4),st(3,state3)},{},{input(2,0).trace_length(5).}).
#pos(p5,{st(5,state5),st(6,state1)},{},{input(5,0).trace_length(10).}).
#pos(p6,{st(5,state5),st(6,state2)},{},{input(5,1).trace_length(7).}).
2 ~ delta(state0,V0,V1,state0,1) :- input(V0,V1).
3 ~ delta(state0,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state1,1) :- input(V0,V1).
3 ~ delta(state0,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state2,1) :- input(V0,V1).
3 ~ delta(state0,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state3,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state3,1) :- input(V0,V1).
3 ~ delta(state0,V0,V1,state3,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state4,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state4,1) :- input(V0,V1).
3 ~ delta(state0,V0,V1,state4,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state5,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state5,1) :- input(V0,V1).
3 ~ delta(state0,V0,V1,state5,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state0,1) :- input(V0,V1).
3 ~ delta(state1,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state1,1) :- input(V0,V1).
3 ~ delta(state1,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state2,1) :- input(V0,V1).
3 ~ delta(state1,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state3,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state3,1) :- input(V0,V1).
3 ~ delta(state1,V0,V1,state3,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state4,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state4,1) :- input(V0,V1).
3 ~ delta(state1,V0,V1,state4,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state5,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state5,1) :- input(V0,V1).
3 ~ delta(state1,V0,V1,state5,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state0,1) :- input(V0,V1).
3 ~ delta(state2,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state1,1) :- input(V0,V1).
3 ~ delta(state2,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state2,1) :- input(V0,V1).
3 ~ delta(state2,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state3,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state3,1) :- input(V0,V1).
3 ~ delta(state2,V0,V1,state3,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state4,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state4,1) :- input(V0,V1).
3 ~ delta(state2,V0,V1,state4,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state5,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state5,1) :- input(V0,V1).
3 ~ delta(state2,V0,V1,state5,2) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state0,1) :- input(V0,V1).
3 ~ delta(state3,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state1,1) :- input(V0,V1).
3 ~ delta(state3,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state2,1) :- input(V0,V1).
3 ~ delta(state3,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state3,0) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state3,1) :- input(V0,V1).
3 ~ delta(state3,V0,V1,state3,2) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state4,0) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state4,1) :- input(V0,V1).
3 ~ delta(state3,V0,V1,state4,2) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state5,0) :- input(V0,V1).
2 ~ delta(state3,V0,V1,state5,1) :- input(V0,V1).
3 ~ delta(state3,V0,V1,state5,2) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state0,1) :- input(V0,V1).
3 ~ delta(state4,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state1,1) :- input(V0,V1).
3 ~ delta(state4,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state2,1) :- input(V0,V1).
3 ~ delta(state4,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state3,0) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state3,1) :- input(V0,V1).
3 ~ delta(state4,V0,V1,state3,2) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state4,0) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state4,1) :- input(V0,V1).
3 ~ delta(state4,V0,V1,state4,2) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state5,0) :- input(V0,V1).
2 ~ delta(state4,V0,V1,state5,1) :- input(V0,V1).
3 ~ delta(state4,V0,V1,state5,2) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state0,1) :- input(V0,V1).
3 ~ delta(state5,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state1,1) :- input(V0,V1).
3 ~ delta(state5,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state2,1) :- input(V0,V1).
3 ~ delta(state5,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state3,0) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state3,1) :- input(V0,V1).
3 ~ delta(state5,V0,V1,state3,2) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state4,0) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state4,1) :- input(V0,V1).
3 ~ delta(state5,V0,V1,state4,2) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state5,0) :- input(V0,V1).
2 ~ delta(state5,V0,V1,state5,1) :- input(V0,V1).
3 ~ delta(state5,V0,V1,state5,2) :- input(V0,V1).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
state(state5).
:- not st(T,state5),trace_length(T).
