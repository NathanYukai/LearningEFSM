%#modeh(delta(const(st),var(time),var(in),const(st),const(cond_id),const(reg),const(reg))).
%#constant(st,state0).
%#constant(st,state1).
%#constant(cond_id,0).
%#constant(cond_id,1).
%#constant(reg,0).
%#constant(reg,1).
%
%#modeb(1,input(var(time),var(in))).

condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 


% Time, state, register value
reg(0,state1,0).
reg(0,state0,0).

% Time, state
st(0,state0).

%state, time, input,state, conditionID, REG_Value_old, REG_Value_NEW
delta(state0,T,C,state0,0,0,1):- input(T,C).
delta(state0,T,C,state0,1,1,2):- input(T,C).
delta(state0,T,C,state1,2,2,2):- input(T,C).
delta(state1,T,C,state1,0,0,3):- input(T,C).


state(state0).
state(state1).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID,REG_O,_),condition(C,ID),reg(T,FROM,REG_O).

%update register value
reg(T+1,ST,REG_N):- reg(T,ST,REG_O), st(T,ST), delta(ST,T,_,_,_,REG_O,REG_N).

%other state keep remembering it’s old value, maybe don’t need this part
% if not, for performance reason, need to find a way of getting the closest TIME
reg(T+1,ST_DIFF,REG_O):- reg(T,ST_DIFF,REG_O), st(T,ST), delta(ST,T,_,_,_,_,_), ST != ST_DIFF.


input(0,0).input(1,1).input(2,0).input(3,0).trace_length(4).

:- not st(T,state1),trace_length(T).
:- not reg(T,state1,3),trace_length(T).
:- st(T,ST),st(T,ST2),ST!=ST2.

#show st/2.
#show reg/3.
