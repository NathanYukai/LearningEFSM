%states and initial, final state
state(start).state(ring).state(end).

initial(start).final(end).
limit(8).
ring_time(3).

%rule for delta
%delta(ST1,ST2,ID):- not cons_delta(ST1,T,ST2),state(ST1),state(ST2),condition(ID),time(T).
%cons_delta(end,T,ST2):- state(ST2), not final(ST2), time(T). 
%cons_delta(ST,1,ST1):- state(ST), not initial(ST), state(ST1).

%available conditions
%make sure conditions are mutual exclusive for each delta
condition(T,1):- ring_time(T).
condition(T,2):- not ring_time(T),valid_time(T).

time(E_ID,T) :- pos_exp(E_ID,M), T=1..M.
valid_time(T):- limit(M), T=1..M.


0{delta(ST,ST2,ID):state(ST2)}1:-state(ST),ID=1..2.

#hide.
#show delta/3.
#show st/3.
st(1,start,E_ID):- pos_exp(E_ID,_).
st(T,ST,E_ID):- st(T-1,ST_p,E_ID), state(ST_p),state(ST), delta(ST_p,ST,ID),condition(T,ID), time(E_ID,T).
:- pos_exp(E_ID,N), not st(N,end,E_ID).
:- neg_exp(E_ID,N), st(N,end,E_IE).

neg_exp(5,3).
neg_exp(4,2).
pos_exp(1,5).
pos_exp(2,6).
pos_exp(3,4).
%limit deltas for given ABC
#minimise{delta(A,B,C)}.

