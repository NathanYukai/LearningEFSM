#modeh(delta(const(st),const(st),const(cond_id))).

#constant(cond_id,0).
#constant(cond_id,1).
#constant(cond_id,2).

#constant(st,state0).
#constant(st,state1).
#constant(st,state2).

condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).
condition(C,2):- C >= 0,C <= 1, input(_,C).


1{input(T,0);input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.

st(0,state0).
1{st(T+1,TO1);st(T+1,TO2)}1:- st(T,FROM),state(TO1),state(TO2),delta(FROM,TO1,ID),condition(C,ID),input(T,C),delta(FROM,TO2,ID2),condition(C,ID2), TO1 != TO2.

st(T+1,TO):- st(T,FROM),state(TO),delta(FROM,TO,ID),condition(C,ID),input(T,C), determin(FROM).
determin(ST):- not nonDetermin(ST), state(ST).
nonDetermin(ST):- delta(ST,TO1,_), state(TO1),state(TO2), delta(ST,TO2,_),TO1 != TO2.

state(state0).
state(state1).
state(state2).

accept :- st(T,state2),trace_length(T).

#pos(p1,{accept},{},{input(0,1).trace_length(1).}).
#pos(p2,{},{accept},{input(0,1).trace_length(1).}).
%#pos(p4,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#neg(n3,{accept},{},{input(0,0).trace_length(1).}).
