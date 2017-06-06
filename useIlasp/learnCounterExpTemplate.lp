condition(0,0).
condition(1,1).

condition(0,2).
condition(1,2).


st(0,0,state0).
st(1,0,state0).
st(AUTO,T+1,TO):- st(AUTO,T,FROM),state(FROM),state(TO),delta(AUTO,FROM,C,TO,ID),condition(C,ID),input(T,C).

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
