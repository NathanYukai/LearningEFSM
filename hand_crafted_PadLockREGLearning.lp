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
#pos(p0,{accept},{},{input(0,1).input(1,0).trace_length(2).}).
#pos(p1,{accept},{},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(p2,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).trace_length(4).}).
#pos(p3,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(p4,{accept},{},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).trace_length(5).}).
#pos(p5,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#pos(p6,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).trace_length(6).}).
#pos(n0,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).trace_length(8).}).
#pos(n1,{},{accept},{input(0,0).trace_length(1).}).
#pos(n2,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n3,{},{accept},{input(0,1).input(1,1).input(2,1).trace_length(3).}).
#pos(n4,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).trace_length(10).}).
#pos(n5,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n6,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n7,{},{accept},{input(0,1).trace_length(1).}).
#pos(n8,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).trace_length(9).}).
#pos(n9,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n10,{},{accept},{input(0,0).input(1,1).input(2,1).trace_length(3).}).
#pos(n11,{},{accept},{input(0,0).trace_length(1).}).
#pos(n12,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).trace_length(11).}).
#pos(n13,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n14,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n15,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n16,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n17,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).trace_length(10).}).
#pos(n18,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n19,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(n20,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n21,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n22,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n23,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n24,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n25,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).trace_length(8).}).
#pos(n26,{},{accept},{input(0,0).input(1,0).trace_length(2).}).
#pos(n27,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n28,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n29,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n30,{},{accept},{input(0,0).trace_length(1).}).
#pos(n31,{},{accept},{input(0,1).trace_length(1).}).
#pos(n32,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n33,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n34,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n35,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n36,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).trace_length(9).}).
#pos(n37,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(n38,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).trace_length(11).}).
#pos(n39,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n40,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n41,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n42,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n43,{},{accept},{input(0,0).input(1,0).trace_length(2).}).
#pos(n44,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n45,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n46,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).trace_length(6).}).
#pos(n47,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n48,{},{accept},{input(0,0).trace_length(1).}).
#pos(n49,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n50,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n51,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n52,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(n53,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n54,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).trace_length(10).}).
#pos(n55,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n56,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n57,{},{accept},{input(0,0).trace_length(1).}).
#pos(n58,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).trace_length(8).}).
#pos(n59,{},{accept},{input(0,1).trace_length(1).}).
#pos(n60,{},{accept},{input(0,0).trace_length(1).}).
#pos(n61,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n62,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).trace_length(10).}).
#pos(n63,{},{accept},{input(0,0).trace_length(1).}).
#pos(n64,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n65,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n66,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n67,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n68,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n69,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n70,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n71,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n72,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n73,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n74,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n75,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n76,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n77,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n78,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n79,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n80,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n81,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n82,{},{accept},{input(0,0).input(1,0).trace_length(2).}).
#pos(n83,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(n84,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n85,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).trace_length(4).}).
#pos(n86,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).trace_length(5).}).
#pos(n87,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).trace_length(9).}).
#pos(n88,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).trace_length(9).}).
#pos(n89,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n90,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n91,{},{accept},{input(0,1).input(1,1).input(2,1).trace_length(3).}).
#pos(n92,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n93,{},{accept},{input(0,1).trace_length(1).}).
#pos(n94,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#pos(n95,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).trace_length(6).}).
1 ~ final_reg(0).
1 ~ final_reg(1).
1 ~ final_reg(2).
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
2 ~ delta(state0,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,1,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,2,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,1,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,2,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,1,2,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,0,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,1,3) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,2,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,1,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,2,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,1,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,2,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,1,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,2,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,1,2,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,0,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,1,3) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,2,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,1,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,2,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,1,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,2,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,1,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,2,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,1,2,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,0,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,1,3) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,2,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,1,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,2,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,1,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,2,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,1,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,2,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,1,2,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,0,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,0,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,0,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,0,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,1,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,1,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,1,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,1,3) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,2,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,2,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,2,2,3) :- input(_,V0).
2 ~ final_reg(0) :- input(_,V0).
2 ~ final_reg(1) :- input(_,V0).
2 ~ final_reg(2) :- input(_,V0).
1 ~  :- input(_,V0).

state(state0).
state(state1).
state(state2).
state(state3).
reg_limit(3).
accept :- st(T,state3,V), final_reg(V),trace_length(T).
#max_penalty(100).
