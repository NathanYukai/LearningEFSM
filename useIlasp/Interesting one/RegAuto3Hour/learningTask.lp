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
st(T+1,TO,0):- st(T,FROM,0),state(FROM),state(TO),delta(FROM,C,TO,ID,0,0),condition(C,ID),input(T,C).
%increase
st(T+1,TO,V+1):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,1),condition(C,ID),input(T,C).
st(T+1,TO,L-1):- st(T,FROM,L),state(FROM),state(TO),delta(FROM,C,TO,ID,L-1,1),condition(C,ID),input(T,C),reg_limit(L).
%reset
st(T+1,TO,R):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,2),condition(C,ID),origin(R),input(T,C).


:- st(T,ST,_),st(T,ST2,_),ST!=ST2.
:- st(_,_,L), reg_limit(L).
:- st(_,_,V), V < 0.


%add 
%  accept:- 
%reg_limit/1
#pos(p0,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).trace_length(5).}).
#pos(n0,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n1,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n2,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).trace_length(9).}).
#pos(n3,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n4,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).trace_length(10).}).
#pos(n5,{},{accept},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#pos(n6,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).trace_length(10).}).
#pos(n7,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n8,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n9,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n10,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n11,{},{accept},{input(0,1).trace_length(1).}).
#pos(n12,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n13,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n14,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n15,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n16,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n17,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n18,{},{accept},{input(0,0).trace_length(1).}).
#pos(n19,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n20,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).trace_length(11).}).
#pos(n21,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(n22,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n23,{},{accept},{input(0,0).trace_length(1).}).
#pos(n24,{},{accept},{input(0,0).trace_length(1).}).
#pos(n25,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).trace_length(11).}).
#pos(n26,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n27,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n28,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n29,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n30,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n31,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n32,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).trace_length(9).}).
#pos(n33,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).trace_length(10).}).
#pos(n34,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).trace_length(11).}).
#pos(n35,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).trace_length(11).}).
#pos(n36,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n37,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n38,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n39,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n40,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n41,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n42,{},{accept},{input(0,0).trace_length(1).}).
#pos(n43,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n44,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n45,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n46,{},{accept},{input(0,0).trace_length(1).}).
#pos(n47,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n48,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n49,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).trace_length(9).}).
#pos(n50,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n51,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).trace_length(5).}).
#pos(n52,{},{accept},{input(0,0).input(1,0).trace_length(2).}).
#pos(n53,{},{accept},{input(0,0).input(1,0).trace_length(2).}).
#pos(n54,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n55,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n56,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).trace_length(5).}).
#pos(n57,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(n58,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
1 ~ final_reg(0).
1 ~ final_reg(1).
1 ~ final_reg(2).
2 ~ delta(state0,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,2,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,0,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,1,2,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,0,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,0,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,0,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,1,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,1,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,1,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,2,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,2,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,2,2,2) :- input(_,V0).
2 ~ final_reg(0) :- input(_,V0).
2 ~ final_reg(1) :- input(_,V0).
2 ~ final_reg(2) :- input(_,V0).
1 ~  :- input(_,V0).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
reg_limit(3).
accept :- st(T,state4,V), final_reg(V),trace_length(T).
#max_penalty(42).
