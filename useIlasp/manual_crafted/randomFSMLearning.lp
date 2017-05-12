#modeh(delta(const(st),var(in),const(st),const(cond_id))).  
#constant(st,state0).
#constant(st,state1).  
#constant(st,state2).  
#constant(st,state3).  
#constant(st,state4).  

#constant(cond_id,0).
#constant(cond_id,1).
#constant(cond_id,2).


#modeb(1,input(_,var(in))).


condition(C,0):- C >= 0,C <= 0, input(_,C). 
condition(C,1):- C >= 1,C <= 1, input(_,C). 
condition(C,2):- C >= 0,C <= 1, input(_,C). 


state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,C,TO,ID),condition(C,ID),input(T,C).


accept :- st(T,state4),trace_length(T).
:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,C,TO,ID), delta(FROM,C,TO,ID2), ID!=ID2.


#pos(p0,{accept},{},{input(0,0).trace_length(1).}).
#pos(p1,{accept},{},{input(0,1).trace_length(1).}).
#pos(p2,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(p3,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(p4,{accept},{},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(p5,{accept},{},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(p6,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(p7,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(p8,{accept},{},{input(0,0).trace_length(1).}).
#pos(p9,{accept},{},{input(0,0).trace_length(1).}).
#pos(p10,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(p11,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(p12,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(p13,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(p14,{accept},{},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(p15,{accept},{},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(p16,{accept},{},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(p17,{accept},{},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(p18,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(p19,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(p20,{accept},{},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(p21,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).trace_length(7).}).
#pos(p22,{accept},{},{input(0,0).trace_length(1).}).
#pos(p23,{accept},{},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(p24,{accept},{},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n0,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n1,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n2,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).trace_length(9).}).
#pos(n3,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n4,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n5,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n6,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(n7,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n8,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n9,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n10,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n11,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n12,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).trace_length(9).}).
#pos(n13,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n14,{},{accept},{input(0,1).input(1,0).trace_length(2).}).

