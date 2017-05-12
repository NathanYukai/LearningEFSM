condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 


1{input(T,0);input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.


% Time, state
st(0,state0,0).
origin(0).

st(T+1,TO,V-1):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,0),condition(C,ID).
st(T+1,TO,V+1):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,1),condition(C,ID).
st(T+1,TO,R):- st(T,FROM,V),state(FROM),state(TO),delta(FROM,C,TO,ID,V,2),condition(C,ID),origin(R).


:- st(T,ST),st(T,ST2),ST!=ST2.
:- st(_,_,L+1), reg_limit(L).
:- st(_,_,V), V < 0.

:- not accept.

%add 
%  accept:- 
%reg_limit/1
