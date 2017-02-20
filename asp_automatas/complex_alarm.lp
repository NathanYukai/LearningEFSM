%add ringing time , canbe changed via constant variables 

%add snoozing by ading vars in delta, not extra states.
%

%define states
state(start).state(active).state(end).

%define time of ringing (register)
#const lim = 10.
#const r = 4.
#const r_dur = 2.
#const s_dur = 2.
ring_time(r).
snooze(r+1).

%Time limit
limit(lim).

%define all the delta ( transitions)
%delta/5:  from state; time; to state; previous snooze value ; after snooze value

delta(start,T,start,0,0) :- not ring_time(T),time(T).
delta(start,T,active,0,0):-  ring_time(T),time(T). 

delta(active,T,active,0,0):- T<R+r_dur,ring_time(R), time(T), not snooze(T).
delta(active,T,active,SN_P,s_dur):- snooze(T), time(T),SN_P=0..s_dur.
%update ring_time after snooze
%try change to just use snooze
ring_time(T+s_dur) :- delta(active,T,active,SN_P,s_dur), SN_P=0..s_dur, time(T).
delta(active,T,active,SN_P,SN_P-1):- time(T),SN_P=1..s_dur.

delta(active,T,end,0,0):- not st(T,active,0),  T>RT+r_dur,ring_time(RT),time(T). 
delta(end,T,end,0,0):- time(T).

%define initial state
st(1,start,0).
%only transfer to a new state if there is a delta from current to next.
st(T,Y,SN):- st(T-1,S,SN_P),state(S),time(T),state(Y),delta(S,T,Y,SN_P,SN),limit(M),time(T). 

%RING at active state if not snooze)
ring(T):- st(T,active,0).

#hide.
%%#show delta/5.
#show st/3.
#show ring/1.

