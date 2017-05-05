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

#pos(p0,{},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(p1,{},{},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(p2,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).trace_length(6).}).
#pos(p3,{},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(p4,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(p5,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(p6,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(p7,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(p8,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(p9,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(p10,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(p11,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(p12,{},{},{input(0,1).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(p13,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(p14,{},{},{input(0,1).input(1,1).input(2,1).input(3,0).trace_length(4).}).
#pos(p15,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(p16,{},{},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(p17,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(p18,{},{},{input(0,1).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(p19,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(p20,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(p21,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(p22,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(p23,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(p24,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(p25,{},{},{input(0,1).input(1,0).trace_length(2).}).
#pos(p26,{},{},{input(0,1).input(1,1).trace_length(2).}).
#pos(p27,{},{},{input(0,1).input(1,0).trace_length(2).}).
#neg(n0,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#neg(n1,{},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#neg(n2,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#neg(n3,{},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#neg(n4,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#neg(n5,{},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).trace_length(5).}).
#neg(n6,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#neg(n7,{},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#neg(n8,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#neg(n9,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#neg(n10,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#neg(n11,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#neg(n12,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#neg(n13,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#neg(n14,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#neg(n15,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#neg(n16,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#neg(n17,{},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#neg(n18,{},{},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#neg(n19,{},{},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#neg(n20,{},{},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#neg(n21,{},{},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#neg(n22,{},{},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#neg(n23,{},{},{input(0,1).trace_length(1).}).
#neg(n24,{st(1,state1)},{},{input(0,0).trace_length(1).}).
2 ~ delta(state0,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state0,1) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state1,1) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state2,1) :- input(V0,V1).
2 ~ delta(state0,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state0,1) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state1,1) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state2,1) :- input(V0,V1).
2 ~ delta(state1,V0,V1,state2,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state0,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state0,1) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state0,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state1,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state1,1) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state1,2) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state2,0) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state2,1) :- input(V0,V1).
2 ~ delta(state2,V0,V1,state2,2) :- input(V0,V1).

state(state0).
state(state1).
state(state2).
:- not st(T,state2),trace_length(T).
#max_penalty(15).
