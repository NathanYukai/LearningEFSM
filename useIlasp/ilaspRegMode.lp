#modeh(delta(const(st),var(in),const(st),const(cond_id),const(reg_value),const(reg_code))).

#constant(cond_id,0).
#constant(cond_id,1).
#constant(cond_id,2).

#constant(reg_code,0).
#constant(reg_code,1).
#constant(reg_code,2).
#constant(reg_code,3).


#modeb(1,input(_,var(in))).

#modeh(final_reg(const(reg_value))).

%add state constant!
%add reg_value constant 
