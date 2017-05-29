#modeh(delta(const(st),const(st),const(id))).
#constant(st,start).
#constant(st,ring).
#constant(st,end).
#constant(id,1).
#constant(id,2).

1 ~ condition(V0,1) :- not ring_time(V0),valid_time(V0).
1 ~ condition(V0,2) :- ring_time(V0).
1 ~ condition(V0,1) :- ring_time(V0).
1 ~ condition(V0,2) :- not ring_time(V0),valid_time(V0).

state(start).state(ring).state(end).
initil(start).final(end).
ring_time(3).

st(1,start).
st(T+1,TO):- st(T,FROM), state(FROM),state(TO), delta(FROM,TO,ID),condition(T,ID), valid_time(T).

valid_time(L):- last_time(L).
valid_time(T-1):- valid_time(T),T>0.

:- st(T,ST),st(T,ST2),ST!=ST2.
:- not st(T,end), last_time(T).
:- not st(_,ring).

#pos(p1,{},{},{last_time(8).}).
#pos(p2,{},{},{last_time(5).}).
#neg(n1,{},{},{last_time(3).}).
#neg(n2,{},{},{last_time(2).}).

