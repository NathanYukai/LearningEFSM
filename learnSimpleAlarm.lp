%%% State information  
state(start).state(ring).state(end).

%%% Ringing time and limit
limit(8).
ring_time(3).

%%% Available conditions for delta
%%% conditions need to be mutual exclusive 
condition(T,1):- ring_time(T).
condition(T,2):- not ring_time(T),valid_time(T).

%%% Time , seperated by example id
time(E_ID,T) :- pos_exp(E_ID,M), T=0..M.
valid_time(T):- limit(M), T=0..M.

%%% Search space for delta
0{delta(ST,ST2,ID):state(ST2)}1:-state(ST),ID=1..2.

%%% Rule st also need to be seperated by example ID
st(0,start,E_ID):- pos_exp(E_ID,_).
st(T+1,TO,E_ID):- st(T,FROM,E_ID), state(FROM),state(TO), delta(FROM,TO,ID),condition(T,ID), time(E_ID,T).

%%% All positive examples have to be met
%%% None of the negative examples must not be met
:- pos_exp(E_ID,N), not st(N+1,end,E_ID).
:- neg_exp(E_ID,N), st(N+1,end,E_ID).

%%% Positive and negative examples
neg_exp(5,3).
neg_exp(4,2).
pos_exp(1,5).
pos_exp(2,6).
pos_exp(3,5).

#show delta/3.
