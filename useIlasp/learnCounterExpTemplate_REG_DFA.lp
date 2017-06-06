condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).
condition(C,2):- C >= 0,C <= 1, input(_,C).


st(0,0,state0).
st(AUTO,T+1,TO):- st(AUTO,T,FROM),state(FROM),state(TO),delta(AUTO,FROM,C,TO,ID),condition(C,ID),input(T,C).


st(1,0,state0,0).
origin(0).

%decrease
st(1,T+1,TO,V-1):- st(1,T,FROM,V),state(FROM),state(TO),delta(1,FROM,C,TO,ID,V,0),condition(C,ID),input(T,C).
%increase
st(1,T+1,TO,V+1):- st(1,T,FROM,V),state(FROM),state(TO),delta(1,FROM,C,TO,ID,V,1),condition(C,ID),input(T,C).
%reset
st(1,T+1,TO,R):- st(1,T,FROM,V),state(FROM),state(TO),delta(1,FROM,C,TO,ID,V,2),condition(C,ID),origin(R),input(T,C).
%do nothing
st(1,T+1,TO,V):- st(1,T,FROM,V),state(FROM),state(TO),delta(1,FROM,C,TO,ID,V,3),condition(C,ID),input(T,C).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
state(state5).
state(state6).

accept(1) :- st(1,T,state6,_), trace_length(T).

:- st(1,T,ST,_),st(1,T,ST2,_),ST!=ST2.
:- st(1,_,_,L), reg_limit(L).
:- st(1,_,_,V), V < 0.



:- st(AUTO,T,ST),st(AUTO,T,ST2),ST!=ST2.

%% accept_0 :- .

same :- accept(0), accept(1).
same :- not accept(0), not accept(1).

#modeh(input(const(step),const(val))).
#modeh(trace_length(const(length))).

#constant(val,0).
#constant(val,1).

#pos(p1,{},{same}).

%%%#constant(step,0).
%%%#constant(length,1).
