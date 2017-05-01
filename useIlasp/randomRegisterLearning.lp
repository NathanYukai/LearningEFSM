#modeh(delta(const(st),var(time),var(in),const(st),const(cond_id),const(reg),const(reg))).
#constant(st,state0).
#constant(st,state1).
#constant(cond_id,0).
#constant(cond_id,1).
#constant(reg,0).
#constant(reg,1).

inp_condition(C,0):- C >= 0,C <= 0, input(_,C). 
inp_condition(C,1):- C >= 1,C <= 1, input(_,C). 
inp_condition(C,2):- C >= 0,C <= 1, input(_,C). 


% Time, state, register value
reg(0,state1,0).
reg(0,state0,0).

% Time, state
st(0,state0).


#modeb(1,input(var(time),var(in))).
state(state0).
state(state1).
reg(state0,1).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID,REG_O,_),condition(C,ID),reg(T,FROM,REG_O).
reg(T+1,ST,REG_N):- reg(T,ST,REG_O), st(T,ST), delta(ST,T,_,_,_,REG_O,REG_N).




:- not st(T,state4),trace_length(T).
:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,T,C,TO,ID), delta(FROM,T,C,TO,ID2), ID!=ID2.


