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

:~delta(_,_,_,_,0).[1@2]
:~delta(_,_,_,_,1).[1@2]
:~delta(_,_,_,_,2).[2@1]

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


1{input(T,0),input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.

#pos(p0,{},{},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(p1,{},{},{input(0,0).input(1,1).trace_length(2).}).
#pos(p2,{},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(p3,{},{},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(p4,{},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(p5,{},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(p6,{},{},{input(0,0).input(3,1).input(4,1).trace_length(5).}).
#neg(n1,{},{},{input(0,0).input(1,0).trace_length(2).}).

%#pos(p0,{st(0,state0),st(1,state1)},{},{input(0,1).trace_length(2).}).
%#pos(p8,{st(0,state0),st(1,state1)},{},{input(0,0).trace_length(2).}).

%#pos(p1,{st(1,state1),st(2,state4)},{},{input(1,1).trace_length(2).}).
%#pos(p2,{st(1,state1),st(2,state3)},{},{input(1,0).trace_length(3).}).

%#pos(p3,{st(2,state3),st(3,state4)},{},{input(2,0).trace_length(3).}).

%#pos(p4,{st(2,state4),st(3,state0)},{},{input(2,0).trace_length(5).}).
%#pos(p5,{st(2,state4),st(3,state2)},{},{input(2,1).trace_length(6).}).

%#pos(p6,{st(3,state2),st(4,state0)},{},{input(3,0).trace_length(6).}).
%#pos(p7,{st(3,state2),st(4,state1)},{},{input(3,1).trace_length(5).}).

#max_penalty(300).
