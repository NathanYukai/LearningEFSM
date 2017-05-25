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
delta(state1,V0,state0,0,1,1) :- input(_,V0).

delta(state0,V0,state1,0,0,1) :- input(_,V0).

delta(state0,V0,state2,1,2,0) :- input(_,V0).

delta(state2,V0,state2,1,1,1) :- input(_,V0).

delta(state2,V0,state4,0,2,0) :- input(_,V0).

final_reg(1).


state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
reg_limit(3).
accept :- st(T,state4,V), final_reg(V),trace_length(T).
:- not accept.
