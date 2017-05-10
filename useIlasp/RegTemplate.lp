

condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 


1{input(T,0);input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.


% Time, state
st(0,state0).

st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID,REG_O,_),condition(C,ID),reg(T,FROM,REG_O).

%update register value
reg(T+1,ST,REG_N):- reg(T,ST,REG_O), st(T,ST), delta(ST,T,_,_,_,REG_O,REG_N).

%other state keep remembering it’s old value, maybe don’t need this part
% if not, for performance reason, need to find a way of getting the closest TIME
reg(T+1,ST_DIFF,REG_O):- reg(T,ST_DIFF,REG_O), st(T,ST), delta(ST,T,_,_,_,_,_), ST != ST_DIFF.

:- st(T,ST),st(T,ST2),ST!=ST2.


%things to add:

%state
%register value for all state
%deltas

%constraint for at end state,
% for having expected value
