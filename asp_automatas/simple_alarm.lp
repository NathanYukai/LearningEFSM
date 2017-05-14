%%%define all states
state(start).state(ring).state(end).

%%% Const can be changed with clingo flags
#const r = 4.
ring_time(r).

%%% Find the end of input
later_time(X):- time(Y), Y>X, time(X).
last_time(T):- not later_time(T), time(T).

%%% Define Accept rule
accept:- st(T+1,end), last_time(T), time(T).

%%% Unsatisfiable if input trace is not accepted
:- not accept.

%%% All deltas (transitions)
delta(start,T,start) :- not ring_time(T),time(T).
delta(start,T,ring):- ring_time(T),time(T).
delta(ring,T,end):- not ring_time(T),time(T).
delta(end,T,end):- not ring_time(T),time(T).

%%% Define initial state
%%% Meaning: at time 0, the automata is at state: start.
st(0,start).

%%% Only transfer to a new state if there is a delta
%%% for the matching states
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,T,TO),time(T).

%%% Example of accepted trace
time(0..5).

#show st/2.



