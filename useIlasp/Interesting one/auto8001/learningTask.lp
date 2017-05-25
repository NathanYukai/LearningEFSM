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

#pos(p0,{accept},{},{input(0,0).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(p1,{accept},{},{input(0,0).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(p2,{accept},{},{input(0,1).input(1,1).trace_length(2).}).
#pos(p3,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(p4,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(p5,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(p6,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(p7,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(p8,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(p9,{accept},{},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(p10,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(p11,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(p12,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n0,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n1,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).trace_length(7).}).
#pos(n2,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n3,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n4,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n5,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n6,{},{accept},{input(0,0).input(1,0).trace_length(2).}).
#pos(n7,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n8,{},{accept},{input(0,0).trace_length(1).}).
#pos(n9,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n10,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n11,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n12,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(n13,{},{accept},{input(0,0).input(1,1).input(2,1).trace_length(3).}).
#pos(n14,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n15,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n16,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n17,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
2 ~ delta(state0,V0,state0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,2) :- input(_,V0).
2 ~ delta(state0,V0,state5,0) :- input(_,V0).
2 ~ delta(state0,V0,state5,1) :- input(_,V0).
2 ~ delta(state0,V0,state5,2) :- input(_,V0).
2 ~ delta(state0,V0,state6,0) :- input(_,V0).
2 ~ delta(state0,V0,state6,1) :- input(_,V0).
2 ~ delta(state0,V0,state6,2) :- input(_,V0).
2 ~ delta(state0,V0,state7,0) :- input(_,V0).
2 ~ delta(state0,V0,state7,1) :- input(_,V0).
2 ~ delta(state0,V0,state7,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,2) :- input(_,V0).
2 ~ delta(state1,V0,state5,0) :- input(_,V0).
2 ~ delta(state1,V0,state5,1) :- input(_,V0).
2 ~ delta(state1,V0,state5,2) :- input(_,V0).
2 ~ delta(state1,V0,state6,0) :- input(_,V0).
2 ~ delta(state1,V0,state6,1) :- input(_,V0).
2 ~ delta(state1,V0,state6,2) :- input(_,V0).
2 ~ delta(state1,V0,state7,0) :- input(_,V0).
2 ~ delta(state1,V0,state7,1) :- input(_,V0).
2 ~ delta(state1,V0,state7,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,2) :- input(_,V0).
2 ~ delta(state2,V0,state5,0) :- input(_,V0).
2 ~ delta(state2,V0,state5,1) :- input(_,V0).
2 ~ delta(state2,V0,state5,2) :- input(_,V0).
2 ~ delta(state2,V0,state6,0) :- input(_,V0).
2 ~ delta(state2,V0,state6,1) :- input(_,V0).
2 ~ delta(state2,V0,state6,2) :- input(_,V0).
2 ~ delta(state2,V0,state7,0) :- input(_,V0).
2 ~ delta(state2,V0,state7,1) :- input(_,V0).
2 ~ delta(state2,V0,state7,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,2) :- input(_,V0).
2 ~ delta(state3,V0,state5,0) :- input(_,V0).
2 ~ delta(state3,V0,state5,1) :- input(_,V0).
2 ~ delta(state3,V0,state5,2) :- input(_,V0).
2 ~ delta(state3,V0,state6,0) :- input(_,V0).
2 ~ delta(state3,V0,state6,1) :- input(_,V0).
2 ~ delta(state3,V0,state6,2) :- input(_,V0).
2 ~ delta(state3,V0,state7,0) :- input(_,V0).
2 ~ delta(state3,V0,state7,1) :- input(_,V0).
2 ~ delta(state3,V0,state7,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,2) :- input(_,V0).
2 ~ delta(state4,V0,state5,0) :- input(_,V0).
2 ~ delta(state4,V0,state5,1) :- input(_,V0).
2 ~ delta(state4,V0,state5,2) :- input(_,V0).
2 ~ delta(state4,V0,state6,0) :- input(_,V0).
2 ~ delta(state4,V0,state6,1) :- input(_,V0).
2 ~ delta(state4,V0,state6,2) :- input(_,V0).
2 ~ delta(state4,V0,state7,0) :- input(_,V0).
2 ~ delta(state4,V0,state7,1) :- input(_,V0).
2 ~ delta(state4,V0,state7,2) :- input(_,V0).
2 ~ delta(state5,V0,state0,0) :- input(_,V0).
2 ~ delta(state5,V0,state0,1) :- input(_,V0).
2 ~ delta(state5,V0,state0,2) :- input(_,V0).
2 ~ delta(state5,V0,state1,0) :- input(_,V0).
2 ~ delta(state5,V0,state1,1) :- input(_,V0).
2 ~ delta(state5,V0,state1,2) :- input(_,V0).
2 ~ delta(state5,V0,state2,0) :- input(_,V0).
2 ~ delta(state5,V0,state2,1) :- input(_,V0).
2 ~ delta(state5,V0,state2,2) :- input(_,V0).
2 ~ delta(state5,V0,state3,0) :- input(_,V0).
2 ~ delta(state5,V0,state3,1) :- input(_,V0).
2 ~ delta(state5,V0,state3,2) :- input(_,V0).
2 ~ delta(state5,V0,state4,0) :- input(_,V0).
2 ~ delta(state5,V0,state4,1) :- input(_,V0).
2 ~ delta(state5,V0,state4,2) :- input(_,V0).
2 ~ delta(state5,V0,state5,0) :- input(_,V0).
2 ~ delta(state5,V0,state5,1) :- input(_,V0).
2 ~ delta(state5,V0,state5,2) :- input(_,V0).
2 ~ delta(state5,V0,state6,0) :- input(_,V0).
2 ~ delta(state5,V0,state6,1) :- input(_,V0).
2 ~ delta(state5,V0,state6,2) :- input(_,V0).
2 ~ delta(state5,V0,state7,0) :- input(_,V0).
2 ~ delta(state5,V0,state7,1) :- input(_,V0).
2 ~ delta(state5,V0,state7,2) :- input(_,V0).
2 ~ delta(state6,V0,state0,0) :- input(_,V0).
2 ~ delta(state6,V0,state0,1) :- input(_,V0).
2 ~ delta(state6,V0,state0,2) :- input(_,V0).
2 ~ delta(state6,V0,state1,0) :- input(_,V0).
2 ~ delta(state6,V0,state1,1) :- input(_,V0).
2 ~ delta(state6,V0,state1,2) :- input(_,V0).
2 ~ delta(state6,V0,state2,0) :- input(_,V0).
2 ~ delta(state6,V0,state2,1) :- input(_,V0).
2 ~ delta(state6,V0,state2,2) :- input(_,V0).
2 ~ delta(state6,V0,state3,0) :- input(_,V0).
2 ~ delta(state6,V0,state3,1) :- input(_,V0).
2 ~ delta(state6,V0,state3,2) :- input(_,V0).
2 ~ delta(state6,V0,state4,0) :- input(_,V0).
2 ~ delta(state6,V0,state4,1) :- input(_,V0).
2 ~ delta(state6,V0,state4,2) :- input(_,V0).
2 ~ delta(state6,V0,state5,0) :- input(_,V0).
2 ~ delta(state6,V0,state5,1) :- input(_,V0).
2 ~ delta(state6,V0,state5,2) :- input(_,V0).
2 ~ delta(state6,V0,state6,0) :- input(_,V0).
2 ~ delta(state6,V0,state6,1) :- input(_,V0).
2 ~ delta(state6,V0,state6,2) :- input(_,V0).
2 ~ delta(state6,V0,state7,0) :- input(_,V0).
2 ~ delta(state6,V0,state7,1) :- input(_,V0).
2 ~ delta(state6,V0,state7,2) :- input(_,V0).
2 ~ delta(state7,V0,state0,0) :- input(_,V0).
2 ~ delta(state7,V0,state0,1) :- input(_,V0).
2 ~ delta(state7,V0,state0,2) :- input(_,V0).
2 ~ delta(state7,V0,state1,0) :- input(_,V0).
2 ~ delta(state7,V0,state1,1) :- input(_,V0).
2 ~ delta(state7,V0,state1,2) :- input(_,V0).
2 ~ delta(state7,V0,state2,0) :- input(_,V0).
2 ~ delta(state7,V0,state2,1) :- input(_,V0).
2 ~ delta(state7,V0,state2,2) :- input(_,V0).
2 ~ delta(state7,V0,state3,0) :- input(_,V0).
2 ~ delta(state7,V0,state3,1) :- input(_,V0).
2 ~ delta(state7,V0,state3,2) :- input(_,V0).
2 ~ delta(state7,V0,state4,0) :- input(_,V0).
2 ~ delta(state7,V0,state4,1) :- input(_,V0).
2 ~ delta(state7,V0,state4,2) :- input(_,V0).
2 ~ delta(state7,V0,state5,0) :- input(_,V0).
2 ~ delta(state7,V0,state5,1) :- input(_,V0).
2 ~ delta(state7,V0,state5,2) :- input(_,V0).
2 ~ delta(state7,V0,state6,0) :- input(_,V0).
2 ~ delta(state7,V0,state6,1) :- input(_,V0).
2 ~ delta(state7,V0,state6,2) :- input(_,V0).
2 ~ delta(state7,V0,state7,0) :- input(_,V0).
2 ~ delta(state7,V0,state7,1) :- input(_,V0).
2 ~ delta(state7,V0,state7,2) :- input(_,V0).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
state(state5).
state(state6).
state(state7).
accept :- st(T,state7),trace_length(T).
#max_penalty(66).
