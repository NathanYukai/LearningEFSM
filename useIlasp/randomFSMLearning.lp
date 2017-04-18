#modeh(delta(const(st),var(time),var(in),const(st),const(cond_id))).  
#constant(st,state0).
#constant(st,state1).  
#constant(st,state2).  
#constant(st,state3).  
#constant(st,state4).  
#constant(cond_id,0).
#constant(cond_id,1).
#constant(cond_id,2).
#constant(cond_id,3).
#constant(cond_id,4).
#constant(cond_id,5).
#constant(cond_id,6).
#constant(cond_id,7).


#modeb(1,input(var(time),var(in))).


condition(C,0):- C >= 7,C <= 10, input(_,C). 
condition(C,1):- C >= 2,C <= 5, input(_,C). 
condition(C,2):- C >= 3,C <= 4, input(_,C). 
condition(C,3):- C >= 5,C <= 7, input(_,C). 
condition(C,4):- C >= 7,C <= 10, input(_,C). 
condition(C,5):- C >= 4,C <= 10, input(_,C). 
condition(C,6):- C >= 0,C <= 3, input(_,C). 

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
st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).

%:- not st(T,state4),trace_length(T).
#pos(p0,{st(1,state3),st(2,state4),input(0,10),input(1,10),trace_length(2)},{}).
