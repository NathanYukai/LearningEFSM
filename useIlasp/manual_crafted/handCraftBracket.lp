condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 

input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).


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

delta(state0,C,state0,0,0,1):- input(_,C).
delta(state0,C,state0,0,1,1):- input(_,C).
delta(state0,C,state0,0,2,1):- input(_,C).

delta(state0,C,state0,1,3,0):- input(_,C).
delta(state0,C,state0,1,2,0):- input(_,C).
delta(state0,C,state0,1,1,0):- input(_,C).

reg_limit(4).

:- not accept.
accept:- st(T,state0,0), trace_length(T).
%add 
%  accept:- 
%reg_limit/1
