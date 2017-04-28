#modeh(delta(const(st),var(time),var(in),const(st),const(cond_id))).  
#constant(st,state0).
#constant(st,state1).  
#constant(st,state2).  
#constant(st,state3).  
#constant(st,state4).  

#constant(cond_id,0).
#constant(cond_id,1).
#constant(cond_id,2).


#modeb(1,input(var(time),var(in))).

:~delta(_,_,_,_,0).[2@1]
:~delta(_,_,_,_,1).[2@2]
:~delta(_,_,_,_,2).[1@3]

condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 


state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,C,TO,ID),condition(C,ID).


:- not st(T,state4),trace_length(T).
:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,T,C,TO,ID), delta(FROM,T,C,TO,ID2), ID!=ID2.


#pos(p0,{st(0,state0),st(1,state1),st(2,state3),st(3,state4)},{},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(p1,{st(0,state0),st(1,state1),st(2,state4)},{},{input(0,0).input(1,1).trace_length(2).}).
#pos(p2,{st(0,state0),st(1,state1),st(2,state4),st(3,state2),st(4,state0),st(5,state1),st(6,state4),st(7,state2),st(8,state0),st(9,state1),st(10,state4)},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(p3,{st(0,state0),st(1,state1),st(2,state4),st(3,state0),st(4,state1),st(5,state4),st(6,state0),st(7,state1),st(8,state4)},{},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(p4,{st(0,state0),st(1,state1),st(2,state3),st(3,state4),st(4,state2),st(5,state0),st(6,state1),st(7,state3),st(8,state4),st(9,state2),st(10,state0),st(11,state1),st(12,state4)},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(p5,{st(0,state0),st(1,state1),st(2,state3),st(3,state4),st(4,state0),st(5,state1),st(6,state3),st(7,state4),st(8,state0),st(9,state1),st(10,state4)},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(p6,{st(0,state0),st(1,state1),st(2,state4),st(3,state2),st(4,state1),st(5,state4)},{},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).trace_length(5).}).

#max_penalty(300).
%delta(state2,V0,V1,state1,1):- input(V0,V1).
%delta(state4,V0,V1,state0,0):- input(V0,V1).
