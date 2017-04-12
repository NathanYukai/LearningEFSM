state(start).state(ring).state(end).
initil(start).final(end).
limit(8).
ring_time(3).

valid_time(1..8).

0{delta(ST,ST2,1..2)}1:- state(ST),state(ST2).
st(1,start).
st(T,ST):- st(T-1,ST_p), state(ST_p),state(ST), delta(ST_p,ST,ID),condition(T,ID), valid_time(T).

:- st(T,ST),st(T,ST2), ST!=ST2.
:- not st(8,_).

%a1 :- not ex(1).
%a2 :- not ex(2).
%1{not a1;not a2}2.
#show st/2.
#show delta/3.

