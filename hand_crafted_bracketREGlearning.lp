condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 


1{input(T,0);input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.


% Time, state
st(0,state0,0).
origin(0).

%decrease
st(T+1,TO,V-1):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,0),condition(C,ID),input(T,C).
%increase
st(T+1,TO,V+1):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,1),condition(C,ID),input(T,C).
%reset
st(T+1,TO,R):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,2),condition(C,ID),origin(R),input(T,C).
%do nothing
st(T+1,TO,V):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,3),condition(C,ID),input(T,C).


:- st(T,ST,_),st(T,ST2,_),ST!=ST2.
:- st(_,_,L), reg_limit(L).
:- st(_,_,V), V < 0.


%add 
%  accept:- 
%reg_limit/1
#pos(p0,{accept},{},{input(0,0).input(1,1).trace_length(2).}).
#pos(p1,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(p2,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(p3,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(p4,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(p5,{accept},{},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).input(14,0).input(15,1).input(16,0).input(17,1).trace_length(18).}).
#pos(p6,{accept},{},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(p7,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n0,{},{accept},{input(0,1).trace_length(1).}).
#pos(n1,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n2,{},{accept},{input(0,1).input(1,1).input(2,1).trace_length(3).}).
#pos(n3,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(n4,{},{accept},{input(0,0).input(1,1).input(2,1).trace_length(3).}).
#pos(n5,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(n6,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n7,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n8,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n9,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n10,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n11,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n12,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n13,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n14,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n15,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).trace_length(7).}).
1 ~ final_reg(0).
1 ~ final_reg(1).
1 ~ final_reg(2).
1 ~ final_reg(3).
2 ~ delta(state0,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,3,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,3,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,3,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,3,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,3,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,3,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,3,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,3,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,3,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,3,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,3,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,3,3) :- input(_,V0).
2 ~ final_reg(0) :- input(_,V0).
2 ~ final_reg(1) :- input(_,V0).
2 ~ final_reg(2) :- input(_,V0).
2 ~ final_reg(3) :- input(_,V0).
1 ~  :- input(_,V0).

state(state0).
reg_limit(4).
accept :- st(T,state0,V), final_reg(V),trace_length(T).
#max_penalty(100).
