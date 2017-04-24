#modeh(delta(const(st),var(time),var(in),const(st),const(cond_id))).  
#constant(st,state0).
#constant(st,state1).  
#constant(st,state2).  
#constant(st,state3).  
#constant(st,state4).  
#constant(st,state5).

#constant(cond_id,0).
#constant(cond_id,1).
#constant(cond_id,2).


#modeb(1,input(var(time),var(in))).
:~delta(_,_,_,_,2).[1@1,2]

condition(C,2):- C >= 0,C <= 1, input(_,C). 
condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 

%#modeh(condition(var(in),const(cond_id))).
%
%#modeb(1,input(_,var(in))).
%#modeb(1, var(in) >= const(num)).
%#modeb(1, var(in) <= const(num)). 
%#constant(num,0).
%#constant(num,1).
%#constant(num,2).
%#constant(num,3).
%#constant(num,4).
%#constant(num,5).
%#constant(num,6).
%#constant(num,7).
%#constant(num,8).
%#constant(num,9).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
state(state5).
st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).


:- not st(T,state5),trace_length(T).
:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,T,C,TO,ID), delta(FROM,T,C,TO,ID2), ID!=ID2.

