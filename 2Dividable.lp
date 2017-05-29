condition(C,0):- C >= 0,C <= 0, input(_,C).
condition(C,1):- C >= 1,C <= 1, input(_,C).
condition(C,2):- C >= 0,C <= 1, input(_,C).


1{input(T,0);input(T,1)}1:- time(T).
time(T-1):- trace_length(T).
time(T-1):- time(T), T>=1.

st(0,state0).
st(T+1,TO):- st(T,FROM),state(FROM),state(TO),delta(FROM,C,TO,ID),condition(C,ID),input(T,C).


:- st(T,ST),st(T,ST2),ST!=ST2.
:- delta(FROM,C,TO,ID), delta(FROM,C,TO,ID2), ID!=ID2.

#pos(p0,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).trace_length(10).}).
#pos(p1,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(p2,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).trace_length(10).}).
#pos(p3,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p4,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p5,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(p6,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p7,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(p8,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p9,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p10,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p11,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(p12,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(p13,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).trace_length(11).}).
#pos(p14,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(p15,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(p16,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p17,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p18,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(p19,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(p20,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p21,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(p22,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p23,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(p24,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p25,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p26,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p27,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(p28,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p29,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p30,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p31,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p32,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(p33,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p34,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p35,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p36,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(p37,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p38,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p39,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p40,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(p41,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(p42,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p43,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(p44,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p45,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p46,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).trace_length(11).}).
#pos(p47,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p48,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p49,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p50,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p51,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(p52,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(p53,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).trace_length(9).}).
#pos(p54,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(p55,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(p56,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(p57,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(p58,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p59,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).trace_length(7).}).
#pos(p60,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(p61,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(p62,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p63,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).trace_length(11).}).
#pos(p64,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(p65,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(p66,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p67,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(p68,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).trace_length(10).}).
#pos(p69,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(p70,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(p71,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).trace_length(9).}).
#pos(p72,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).trace_length(10).}).
#pos(p73,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(p74,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p75,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(p76,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(p77,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(p78,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(p79,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).trace_length(11).}).
#pos(p80,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p81,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p82,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(p83,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(p84,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(p85,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(p86,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(p87,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(p88,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(p89,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(p90,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(p91,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p92,{accept},{},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(p93,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).trace_length(11).}).
#pos(p94,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).trace_length(9).}).
#pos(p95,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(p96,{accept},{},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(p97,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p98,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(p99,{accept},{},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n0,{},{accept},{input(0,1).trace_length(1).}).
#pos(n1,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n2,{},{accept},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#pos(n3,{},{accept},{input(0,1).input(1,1).input(2,1).trace_length(3).}).
#pos(n4,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n5,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n6,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(n7,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(n8,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n9,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n10,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n11,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).trace_length(5).}).
#pos(n12,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n13,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n14,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n15,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).trace_length(5).}).
#pos(n16,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n17,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(n18,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n19,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n20,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n21,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(n22,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n23,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n24,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n25,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(n26,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n27,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n28,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n29,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(n30,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n31,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n32,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n33,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n34,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n35,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n36,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n37,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n38,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n39,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n40,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n41,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n42,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n43,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n44,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n45,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n46,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n47,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n48,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n49,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n50,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n51,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n52,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n53,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n54,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n55,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n56,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n57,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n58,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n59,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n60,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n61,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n62,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n63,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n64,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n65,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n66,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n67,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n68,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n69,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n70,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n71,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n72,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n73,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n74,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n75,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n76,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n77,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n78,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n79,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n80,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n81,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n82,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n83,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n84,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n85,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n86,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n87,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n88,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n89,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n90,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n91,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n92,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n93,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n94,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n95,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n96,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n97,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n98,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n99,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n100,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n101,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n102,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n103,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n104,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n105,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n106,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n107,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n108,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n109,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n110,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n111,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n112,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n113,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n114,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n115,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n116,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n117,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n118,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n119,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n120,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n121,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n122,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n123,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n124,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).trace_length(8).}).
#pos(n125,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n126,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n127,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n128,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n129,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n130,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n131,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n132,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n133,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n134,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n135,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n136,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n137,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n138,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n139,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n140,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n141,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n142,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n143,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n144,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n145,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n146,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n147,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n148,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n149,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n150,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n151,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n152,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n153,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n154,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n155,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n156,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n157,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n158,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n159,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n160,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n161,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n162,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n163,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n164,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n165,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n166,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n167,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n168,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n169,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n170,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n171,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n172,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n173,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n174,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n175,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n176,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n177,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n178,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n179,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n180,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n181,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n182,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n183,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n184,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n185,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n186,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n187,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n188,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n189,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n190,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n191,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n192,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n193,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n194,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n195,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n196,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n197,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n198,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n199,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n200,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n201,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n202,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n203,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n204,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n205,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n206,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n207,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n208,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n209,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n210,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n211,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n212,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n213,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n214,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n215,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n216,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n217,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n218,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n219,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n220,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n221,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n222,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n223,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n224,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n225,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n226,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n227,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n228,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n229,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n230,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n231,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n232,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n233,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n234,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n235,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n236,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n237,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n238,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n239,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n240,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n241,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n242,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n243,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n244,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n245,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n246,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n247,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n248,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n249,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n250,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n251,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n252,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n253,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n254,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n255,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n256,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n257,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n258,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n259,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n260,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n261,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n262,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n263,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n264,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n265,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n266,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n267,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n268,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n269,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n270,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n271,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n272,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n273,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n274,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n275,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n276,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n277,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n278,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n279,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n280,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n281,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n282,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n283,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n284,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n285,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n286,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n287,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n288,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n289,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n290,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n291,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n292,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n293,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n294,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n295,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n296,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n297,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n298,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n299,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n300,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n301,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n302,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n303,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n304,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n305,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n306,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n307,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n308,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n309,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n310,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n311,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n312,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n313,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n314,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n315,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n316,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n317,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n318,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n319,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n320,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n321,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n322,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n323,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n324,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n325,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n326,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n327,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n328,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n329,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n330,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n331,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n332,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n333,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n334,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n335,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n336,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n337,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n338,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n339,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n340,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n341,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n342,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n343,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n344,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n345,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n346,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n347,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n348,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n349,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n350,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n351,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n352,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n353,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n354,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n355,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n356,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n357,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n358,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n359,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n360,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n361,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n362,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n363,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n364,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n365,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n366,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n367,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n368,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n369,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n370,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n371,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n372,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n373,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n374,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n375,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n376,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n377,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n378,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n379,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n380,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n381,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n382,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n383,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n384,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n385,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n386,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n387,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n388,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n389,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n390,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n391,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n392,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n393,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n394,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n395,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n396,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n397,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n398,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n399,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n400,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n401,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n402,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n403,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n404,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n405,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n406,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n407,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n408,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n409,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n410,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n411,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n412,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n413,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n414,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n415,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n416,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n417,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n418,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n419,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n420,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n421,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n422,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n423,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n424,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n425,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n426,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n427,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n428,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n429,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n430,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n431,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n432,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n433,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n434,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n435,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n436,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n437,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n438,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n439,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n440,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n441,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n442,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n443,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n444,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n445,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n446,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n447,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n448,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n449,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n450,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n451,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n452,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n453,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n454,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n455,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n456,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n457,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n458,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n459,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n460,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n461,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n462,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n463,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n464,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n465,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n466,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n467,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n468,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n469,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n470,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n471,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n472,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n473,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n474,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n475,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n476,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n477,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n478,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n479,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n480,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n481,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n482,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n483,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n484,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n485,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n486,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n487,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n488,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n489,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n490,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n491,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n492,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n493,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n494,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n495,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n496,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n497,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n498,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).trace_length(10).}).
#pos(n499,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).trace_length(10).}).
2 ~ delta(state0,V0,state0,0) :- input(_,V0).
2 ~ delta(state0,V0,state0,1) :- input(_,V0).
2 ~ delta(state0,V0,state0,2) :- input(_,V0).
2 ~ delta(state0,V0,state1,0) :- input(_,V0).
2 ~ delta(state0,V0,state1,1) :- input(_,V0).
2 ~ delta(state0,V0,state1,2) :- input(_,V0).
2 ~ delta(state0,V0,state2,0) :- input(_,V0).
2 ~ delta(state0,V0,state2,1) :- input(_,V0).
2 ~ delta(state0,V0,state2,2) :- input(_,V0).
2 ~ delta(state0,V0,state3,0) :- input(_,V0).
2 ~ delta(state0,V0,state3,1) :- input(_,V0).
2 ~ delta(state0,V0,state3,2) :- input(_,V0).
2 ~ delta(state0,V0,state4,0) :- input(_,V0).
2 ~ delta(state0,V0,state4,1) :- input(_,V0).
2 ~ delta(state0,V0,state4,2) :- input(_,V0).
2 ~ delta(state1,V0,state0,0) :- input(_,V0).
2 ~ delta(state1,V0,state0,1) :- input(_,V0).
2 ~ delta(state1,V0,state0,2) :- input(_,V0).
2 ~ delta(state1,V0,state1,0) :- input(_,V0).
2 ~ delta(state1,V0,state1,1) :- input(_,V0).
2 ~ delta(state1,V0,state1,2) :- input(_,V0).
2 ~ delta(state1,V0,state2,0) :- input(_,V0).
2 ~ delta(state1,V0,state2,1) :- input(_,V0).
2 ~ delta(state1,V0,state2,2) :- input(_,V0).
2 ~ delta(state1,V0,state3,0) :- input(_,V0).
2 ~ delta(state1,V0,state3,1) :- input(_,V0).
2 ~ delta(state1,V0,state3,2) :- input(_,V0).
2 ~ delta(state1,V0,state4,0) :- input(_,V0).
2 ~ delta(state1,V0,state4,1) :- input(_,V0).
2 ~ delta(state1,V0,state4,2) :- input(_,V0).
2 ~ delta(state2,V0,state0,0) :- input(_,V0).
2 ~ delta(state2,V0,state0,1) :- input(_,V0).
2 ~ delta(state2,V0,state0,2) :- input(_,V0).
2 ~ delta(state2,V0,state1,0) :- input(_,V0).
2 ~ delta(state2,V0,state1,1) :- input(_,V0).
2 ~ delta(state2,V0,state1,2) :- input(_,V0).
2 ~ delta(state2,V0,state2,0) :- input(_,V0).
2 ~ delta(state2,V0,state2,1) :- input(_,V0).
2 ~ delta(state2,V0,state2,2) :- input(_,V0).
2 ~ delta(state2,V0,state3,0) :- input(_,V0).
2 ~ delta(state2,V0,state3,1) :- input(_,V0).
2 ~ delta(state2,V0,state3,2) :- input(_,V0).
2 ~ delta(state2,V0,state4,0) :- input(_,V0).
2 ~ delta(state2,V0,state4,1) :- input(_,V0).
2 ~ delta(state2,V0,state4,2) :- input(_,V0).
2 ~ delta(state3,V0,state0,0) :- input(_,V0).
2 ~ delta(state3,V0,state0,1) :- input(_,V0).
2 ~ delta(state3,V0,state0,2) :- input(_,V0).
2 ~ delta(state3,V0,state1,0) :- input(_,V0).
2 ~ delta(state3,V0,state1,1) :- input(_,V0).
2 ~ delta(state3,V0,state1,2) :- input(_,V0).
2 ~ delta(state3,V0,state2,0) :- input(_,V0).
2 ~ delta(state3,V0,state2,1) :- input(_,V0).
2 ~ delta(state3,V0,state2,2) :- input(_,V0).
2 ~ delta(state3,V0,state3,0) :- input(_,V0).
2 ~ delta(state3,V0,state3,1) :- input(_,V0).
2 ~ delta(state3,V0,state3,2) :- input(_,V0).
2 ~ delta(state3,V0,state4,0) :- input(_,V0).
2 ~ delta(state3,V0,state4,1) :- input(_,V0).
2 ~ delta(state3,V0,state4,2) :- input(_,V0).
2 ~ delta(state4,V0,state0,0) :- input(_,V0).
2 ~ delta(state4,V0,state0,1) :- input(_,V0).
2 ~ delta(state4,V0,state0,2) :- input(_,V0).
2 ~ delta(state4,V0,state1,0) :- input(_,V0).
2 ~ delta(state4,V0,state1,1) :- input(_,V0).
2 ~ delta(state4,V0,state1,2) :- input(_,V0).
2 ~ delta(state4,V0,state2,0) :- input(_,V0).
2 ~ delta(state4,V0,state2,1) :- input(_,V0).
2 ~ delta(state4,V0,state2,2) :- input(_,V0).
2 ~ delta(state4,V0,state3,0) :- input(_,V0).
2 ~ delta(state4,V0,state3,1) :- input(_,V0).
2 ~ delta(state4,V0,state3,2) :- input(_,V0).
2 ~ delta(state4,V0,state4,0) :- input(_,V0).
2 ~ delta(state4,V0,state4,1) :- input(_,V0).
2 ~ delta(state4,V0,state4,2) :- input(_,V0).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
accept :- st(T,state4),trace_length(T).
#max_penalty(22).
