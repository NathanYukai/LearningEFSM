%#modeh(delta(var(st),var(st2),const(t2))).
%#modeb(1,state(var(st))).
%#modeb(1,state(var(st2))).

#modeh(condition(var(t1),const(t2))).
#modeb(1,ring_time(var(t1))).
#modeb(1,valid_time(var(t1))).
#constant(t2,1).
#constant(t2,2).
#maxv(1).

0{delta(ST,ST2,1..2)}1 :- state(ST),state(ST2).

state(start).state(ring).state(end).
initil(start).final(end).
limit(8).
ring_time(3).

valid_time(1..8).

st(1,start).
st(T,ST):- st(T-1,ST_p), state(ST_p),state(ST), delta(ST_p,ST,ID),condition(T,ID), valid_time(T).

:- st(T,ST),st(T,ST2),ST!=ST2.
:- not st(8,_).

#pos(p1,{st(3,ring),st(8,end)},{}).
#pos(p2,{st(2,start),st(4,end)},{}).


