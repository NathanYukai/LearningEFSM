%define states
state(start).state(ring).state(end).

#const r = 4.
ring_time(r).
later_time(X):- time(Y), Y>X, time(X).
last_time(T):- not later_time(T), time(T).
accept:- st(T,end), last_time(T), time(T).

%Time limit
limit(100).
%all deltas (transitions)
delta(start,T,start) :- not ring_time(T),time(T).
delta(start,T,ring):- ring_time(T),time(T).
delta(ring,T,end):- not ring_time(T),time(T).
delta(end,T,end):- not ring_time(T),time(T).

%define initial state
st(1,start).
%only transfer to a new state if there is a delta from current to next.
st(T,Y):- st(T-1,S),state(S),state(Y),delta(S,T,Y),time(T).
:- not accept.
#hide.

#show st/2.
