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

#pos(p0,{accept},{},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).trace_length(8).}).
#pos(p1,{accept},{},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).trace_length(7).}).
#pos(p2,{accept},{},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).trace_length(9).}).
#pos(p3,{accept},{},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(p4,{accept},{},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).trace_length(11).}).
#pos(p5,{accept},{},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n0,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n1,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n2,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n3,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n4,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n5,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n6,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n7,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).input(14,0).input(15,1).trace_length(16).}).
#pos(n8,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n9,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n10,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n11,{},{accept},{input(0,0).trace_length(1).}).
#pos(n12,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n13,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n14,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n15,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n16,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n17,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).input(14,0).input(15,1).trace_length(16).}).
#pos(n18,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n19,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n20,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n21,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).trace_length(4).}).
#pos(n22,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n23,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n24,{},{accept},{input(0,0).trace_length(1).}).
#pos(n25,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n26,{},{accept},{input(0,0).trace_length(1).}).
#pos(n27,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n28,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n29,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n30,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n31,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n32,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n33,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n34,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).trace_length(11).}).
#pos(n35,{},{accept},{input(0,0).trace_length(1).}).
#pos(n36,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n37,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n38,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n39,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n40,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n41,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(n42,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n43,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n44,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n45,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n46,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n47,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n48,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n49,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n50,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n51,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n52,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n53,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).trace_length(11).}).
#pos(n54,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n55,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n56,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n57,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n58,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#pos(n59,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n60,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n61,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n62,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n63,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n64,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n65,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n66,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n67,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).input(14,0).input(15,1).input(16,0).trace_length(17).}).
#pos(n68,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n69,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n70,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n71,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).trace_length(11).}).
#pos(n72,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n73,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n74,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n75,{},{accept},{input(0,0).trace_length(1).}).
#pos(n76,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n77,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n78,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).trace_length(9).}).
#pos(n79,{},{accept},{input(0,0).trace_length(1).}).
#pos(n80,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n81,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n82,{},{accept},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#pos(n83,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).trace_length(6).}).
#pos(n84,{},{accept},{input(0,0).trace_length(1).}).
#pos(n85,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n86,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n87,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n88,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n89,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).trace_length(7).}).
#pos(n90,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).input(14,0).input(15,0).trace_length(16).}).
#pos(n91,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n92,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n93,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n94,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n95,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n96,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n97,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).input(14,0).input(15,1).trace_length(16).}).
#pos(n98,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n99,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n100,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n101,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n102,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n103,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n104,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).input(14,1).input(15,0).trace_length(16).}).
#pos(n105,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n106,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n107,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,0).input(14,1).input(15,1).input(16,0).trace_length(17).}).
#pos(n108,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n109,{},{accept},{input(0,0).trace_length(1).}).
#pos(n110,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n111,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).trace_length(4).}).
#pos(n112,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n113,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).trace_length(7).}).
#pos(n114,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n115,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n116,{},{accept},{input(0,0).trace_length(1).}).
#pos(n117,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n118,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n119,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n120,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n121,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n122,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n123,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n124,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).input(14,0).input(15,1).trace_length(16).}).
#pos(n125,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n126,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n127,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n128,{},{accept},{input(0,0).trace_length(1).}).
#pos(n129,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n130,{},{accept},{input(0,0).trace_length(1).}).
#pos(n131,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).input(14,1).input(15,0).input(16,0).input(17,1).trace_length(18).}).
#pos(n132,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n133,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n134,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n135,{},{accept},{input(0,0).trace_length(1).}).
#pos(n136,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n137,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n138,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n139,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n140,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(n141,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n142,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n143,{},{accept},{input(0,0).trace_length(1).}).
#pos(n144,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n145,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n146,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n147,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n148,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n149,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n150,{},{accept},{input(0,1).input(1,1).input(2,1).trace_length(3).}).
#pos(n151,{},{accept},{input(0,0).trace_length(1).}).
#pos(n152,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n153,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n154,{},{accept},{input(0,0).trace_length(1).}).
#pos(n155,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n156,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n157,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n158,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n159,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n160,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n161,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n162,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n163,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n164,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n165,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,1).input(14,1).input(15,0).trace_length(16).}).
#pos(n166,{},{accept},{input(0,0).trace_length(1).}).
#pos(n167,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n168,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n169,{},{accept},{input(0,0).trace_length(1).}).
#pos(n170,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).trace_length(10).}).
#pos(n171,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n172,{},{accept},{input(0,0).trace_length(1).}).
#pos(n173,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n174,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n175,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n176,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n177,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(n178,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n179,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(n180,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n181,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n182,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n183,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n184,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n185,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n186,{},{accept},{input(0,0).trace_length(1).}).
#pos(n187,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n188,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n189,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n190,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n191,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n192,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n193,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).trace_length(11).}).
#pos(n194,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n195,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(n196,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n197,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n198,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(n199,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).input(14,0).input(15,0).input(16,1).trace_length(17).}).
#pos(n200,{},{accept},{input(0,0).trace_length(1).}).
#pos(n201,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n202,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n203,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n204,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n205,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n206,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n207,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n208,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n209,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n210,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n211,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).trace_length(10).}).
#pos(n212,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n213,{},{accept},{input(0,0).trace_length(1).}).
#pos(n214,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n215,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n216,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n217,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n218,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).trace_length(4).}).
#pos(n219,{},{accept},{input(0,0).trace_length(1).}).
#pos(n220,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n221,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(n222,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n223,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n224,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n225,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n226,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n227,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n228,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n229,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n230,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n231,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n232,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n233,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n234,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,1).input(14,1).input(15,0).trace_length(16).}).
#pos(n235,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n236,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).trace_length(11).}).
#pos(n237,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n238,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n239,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).input(15,1).trace_length(16).}).
#pos(n240,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n241,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n242,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).trace_length(9).}).
#pos(n243,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n244,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n245,{},{accept},{input(0,0).trace_length(1).}).
#pos(n246,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n247,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n248,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n249,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n250,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n251,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n252,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n253,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n254,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n255,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n256,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).input(13,1).input(14,0).input(15,1).input(16,1).trace_length(17).}).
#pos(n257,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n258,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n259,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n260,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n261,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).trace_length(5).}).
#pos(n262,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n263,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n264,{},{accept},{input(0,0).trace_length(1).}).
#pos(n265,{},{accept},{input(0,0).trace_length(1).}).
#pos(n266,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).trace_length(4).}).
#pos(n267,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n268,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n269,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#pos(n270,{},{accept},{input(0,0).trace_length(1).}).
#pos(n271,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n272,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n273,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n274,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).trace_length(4).}).
#pos(n275,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n276,{},{accept},{input(0,0).trace_length(1).}).
#pos(n277,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n278,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n279,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n280,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n281,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(n282,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n283,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n284,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n285,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n286,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n287,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n288,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n289,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n290,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n291,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).input(14,1).input(15,0).input(16,1).trace_length(17).}).
#pos(n292,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n293,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n294,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).trace_length(11).}).
#pos(n295,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n296,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n297,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n298,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n299,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n300,{},{accept},{input(0,0).trace_length(1).}).
#pos(n301,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n302,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n303,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n304,{},{accept},{input(0,0).trace_length(1).}).
#pos(n305,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n306,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n307,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n308,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n309,{},{accept},{input(0,0).trace_length(1).}).
#pos(n310,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n311,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n312,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n313,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).trace_length(11).}).
#pos(n314,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n315,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n316,{},{accept},{input(0,0).trace_length(1).}).
#pos(n317,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n318,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n319,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n320,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n321,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n322,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n323,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n324,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n325,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n326,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).input(15,1).input(16,1).input(17,0).input(18,1).trace_length(19).}).
#pos(n327,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n328,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n329,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n330,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n331,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n332,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n333,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n334,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n335,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n336,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).input(14,0).input(15,0).trace_length(16).}).
#pos(n337,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n338,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n339,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).trace_length(7).}).
#pos(n340,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n341,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n342,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n343,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n344,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n345,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n346,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n347,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).input(15,0).trace_length(16).}).
#pos(n348,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n349,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n350,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n351,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).trace_length(4).}).
#pos(n352,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(n353,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n354,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n355,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n356,{},{accept},{input(0,0).trace_length(1).}).
#pos(n357,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n358,{},{accept},{input(0,0).trace_length(1).}).
#pos(n359,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n360,{},{accept},{input(0,0).trace_length(1).}).
#pos(n361,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n362,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n363,{},{accept},{input(0,0).trace_length(1).}).
#pos(n364,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).input(14,1).input(15,0).input(16,1).trace_length(17).}).
#pos(n365,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n366,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).trace_length(9).}).
#pos(n367,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(n368,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n369,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n370,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n371,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(n372,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n373,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n374,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n375,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#pos(n376,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n377,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n378,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n379,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n380,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n381,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n382,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n383,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).input(14,1).input(15,0).input(16,0).trace_length(17).}).
#pos(n384,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n385,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n386,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n387,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n388,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n389,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n390,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).trace_length(6).}).
#pos(n391,{},{accept},{input(0,0).trace_length(1).}).
#pos(n392,{},{accept},{input(0,0).trace_length(1).}).
#pos(n393,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n394,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n395,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n396,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,0).input(14,0).input(15,1).trace_length(16).}).
#pos(n397,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n398,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n399,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n400,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n401,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n402,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).input(14,1).input(15,0).trace_length(16).}).
#pos(n403,{},{accept},{input(0,0).input(1,0).input(2,1).trace_length(3).}).
#pos(n404,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n405,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).trace_length(10).}).
#pos(n406,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).input(14,0).input(15,1).trace_length(16).}).
#pos(n407,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n408,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n409,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n410,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n411,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n412,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).trace_length(11).}).
#pos(n413,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n414,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).input(14,0).input(15,0).input(16,1).input(17,0).trace_length(18).}).
#pos(n415,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n416,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n417,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n418,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(n419,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n420,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n421,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(n422,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n423,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n424,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n425,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(n426,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n427,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,1).input(14,0).input(15,0).input(16,0).input(17,0).trace_length(18).}).
#pos(n428,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n429,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n430,{},{accept},{input(0,0).trace_length(1).}).
#pos(n431,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n432,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n433,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).input(14,1).input(15,0).input(16,1).trace_length(17).}).
#pos(n434,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n435,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n436,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n437,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n438,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).trace_length(6).}).
#pos(n439,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n440,{},{accept},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#pos(n441,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n442,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n443,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,0).input(14,1).input(15,1).input(16,1).input(17,0).trace_length(18).}).
#pos(n444,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(n445,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n446,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n447,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,1).input(14,0).trace_length(15).}).
#pos(n448,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n449,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n450,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n451,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n452,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).trace_length(11).}).
#pos(n453,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n454,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n455,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(n456,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n457,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n458,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).trace_length(11).}).
#pos(n459,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n460,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n461,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n462,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n463,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(n464,{},{accept},{input(0,0).trace_length(1).}).
#pos(n465,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n466,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n467,{},{accept},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#pos(n468,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n469,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n470,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n471,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(n472,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n473,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n474,{},{accept},{input(0,0).trace_length(1).}).
#pos(n475,{},{accept},{input(0,1).input(1,1).input(2,1).trace_length(3).}).
#pos(n476,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n477,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n478,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n479,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).input(15,1).input(16,1).trace_length(17).}).
#pos(n480,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n481,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n482,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).trace_length(11).}).
#pos(n483,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n484,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n485,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).trace_length(11).}).
#pos(n486,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(n487,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n488,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n489,{},{accept},{input(0,0).trace_length(1).}).
#pos(n490,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n491,{},{accept},{input(0,0).trace_length(1).}).
#pos(n492,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).input(14,0).input(15,0).trace_length(16).}).
#pos(n493,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).trace_length(9).}).
#pos(n494,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n495,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n496,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).trace_length(5).}).
#pos(n497,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n498,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n499,{},{accept},{input(0,0).trace_length(1).}).
#pos(n500,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n501,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n502,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n503,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n504,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n505,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).input(14,1).input(15,0).input(16,0).trace_length(17).}).
#pos(n506,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n507,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n508,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n509,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).trace_length(7).}).
#pos(n510,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n511,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n512,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n513,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n514,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(n515,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n516,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n517,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).input(14,1).input(15,0).trace_length(16).}).
#pos(n518,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n519,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n520,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n521,{},{accept},{input(0,0).trace_length(1).}).
#pos(n522,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n523,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).input(15,0).trace_length(16).}).
#pos(n524,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(n525,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n526,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n527,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n528,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n529,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n530,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n531,{},{accept},{input(0,0).trace_length(1).}).
#pos(n532,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n533,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).input(15,1).trace_length(16).}).
#pos(n534,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n535,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n536,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n537,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n538,{},{accept},{input(0,0).trace_length(1).}).
#pos(n539,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n540,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n541,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).input(14,0).input(15,0).trace_length(16).}).
#pos(n542,{},{accept},{input(0,0).trace_length(1).}).
#pos(n543,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).input(14,0).input(15,1).trace_length(16).}).
#pos(n544,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n545,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n546,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n547,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n548,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n549,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n550,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n551,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n552,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n553,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n554,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).input(14,1).input(15,1).trace_length(16).}).
#pos(n555,{},{accept},{input(0,0).trace_length(1).}).
#pos(n556,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n557,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n558,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).input(14,1).input(15,0).input(16,0).input(17,1).trace_length(18).}).
#pos(n559,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n560,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n561,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).input(14,0).input(15,1).input(16,0).trace_length(17).}).
#pos(n562,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n563,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n564,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n565,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n566,{},{accept},{input(0,0).trace_length(1).}).
#pos(n567,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n568,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n569,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n570,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n571,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n572,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n573,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n574,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n575,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n576,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n577,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n578,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n579,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,1).input(14,0).trace_length(15).}).
#pos(n580,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n581,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n582,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n583,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n584,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,0).input(14,1).input(15,0).trace_length(16).}).
#pos(n585,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n586,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).trace_length(12).}).
#pos(n587,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n588,{},{accept},{input(0,0).trace_length(1).}).
#pos(n589,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(n590,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n591,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,1).input(14,0).input(15,0).input(16,0).input(17,1).trace_length(18).}).
#pos(n592,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n593,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n594,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(n595,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).input(15,0).input(16,0).input(17,0).input(18,1).trace_length(19).}).
#pos(n596,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n597,{},{accept},{input(0,0).trace_length(1).}).
#pos(n598,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n599,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n600,{},{accept},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#pos(n601,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).input(14,0).trace_length(15).}).
#pos(n602,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n603,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).input(14,0).trace_length(15).}).
#pos(n604,{},{accept},{input(0,1).input(1,1).input(2,1).trace_length(3).}).
#pos(n605,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n606,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n607,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,1).input(14,0).input(15,0).trace_length(16).}).
#pos(n608,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).trace_length(10).}).
#pos(n609,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).input(13,1).input(14,1).input(15,1).trace_length(16).}).
#pos(n610,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n611,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n612,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n613,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n614,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n615,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n616,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n617,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n618,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n619,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).input(14,1).input(15,0).input(16,0).trace_length(17).}).
#pos(n620,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).input(14,0).input(15,0).input(16,0).trace_length(17).}).
#pos(n621,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n622,{},{accept},{input(0,0).trace_length(1).}).
#pos(n623,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).trace_length(12).}).
#pos(n624,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).input(14,1).trace_length(15).}).
#pos(n625,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n626,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n627,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n628,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n629,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,1).input(14,1).trace_length(15).}).
#pos(n630,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n631,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).trace_length(7).}).
#pos(n632,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n633,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n634,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).input(14,0).input(15,1).input(16,1).input(17,0).trace_length(18).}).
#pos(n635,{},{accept},{input(0,0).trace_length(1).}).
#pos(n636,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n637,{},{accept},{input(0,0).trace_length(1).}).
#pos(n638,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).trace_length(7).}).
#pos(n639,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n640,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).input(14,1).input(15,0).input(16,0).input(17,0).trace_length(18).}).
#pos(n641,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).trace_length(11).}).
#pos(n642,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n643,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n644,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).input(14,1).input(15,1).input(16,1).trace_length(17).}).
#pos(n645,{},{accept},{input(0,0).trace_length(1).}).
#pos(n646,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n647,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n648,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n649,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).trace_length(9).}).
#pos(n650,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).input(15,1).trace_length(16).}).
#pos(n651,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n652,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n653,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,1).input(14,1).input(15,0).input(16,1).input(17,0).trace_length(18).}).
#pos(n654,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(n655,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).trace_length(6).}).
#pos(n656,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n657,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n658,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).input(15,1).trace_length(16).}).
#pos(n659,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).trace_length(5).}).
#pos(n660,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n661,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n662,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n663,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).trace_length(6).}).
#pos(n664,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n665,{},{accept},{input(0,0).trace_length(1).}).
#pos(n666,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n667,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n668,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).input(14,1).input(15,0).trace_length(16).}).
#pos(n669,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n670,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n671,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n672,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n673,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).trace_length(8).}).
#pos(n674,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n675,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n676,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).input(14,0).input(15,1).trace_length(16).}).
#pos(n677,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n678,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n679,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n680,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n681,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n682,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n683,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n684,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n685,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n686,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).input(14,0).input(15,0).input(16,1).input(17,1).trace_length(18).}).
#pos(n687,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n688,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).trace_length(4).}).
#pos(n689,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n690,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n691,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).input(13,0).input(14,1).input(15,1).input(16,1).trace_length(17).}).
#pos(n692,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).trace_length(4).}).
#pos(n693,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n694,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n695,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n696,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n697,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n698,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n699,{},{accept},{input(0,0).trace_length(1).}).
#pos(n700,{},{accept},{input(0,0).trace_length(1).}).
#pos(n701,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n702,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n703,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n704,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).trace_length(11).}).
#pos(n705,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).input(14,0).input(15,1).trace_length(16).}).
#pos(n706,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n707,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n708,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).trace_length(8).}).
#pos(n709,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).trace_length(11).}).
#pos(n710,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n711,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n712,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n713,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).trace_length(6).}).
#pos(n714,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n715,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n716,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n717,{},{accept},{input(0,0).trace_length(1).}).
#pos(n718,{},{accept},{input(0,0).trace_length(1).}).
#pos(n719,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n720,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n721,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n722,{},{accept},{input(0,0).trace_length(1).}).
#pos(n723,{},{accept},{input(0,0).trace_length(1).}).
#pos(n724,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(n725,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n726,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).trace_length(4).}).
#pos(n727,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).trace_length(8).}).
#pos(n728,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n729,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n730,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n731,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n732,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n733,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).trace_length(7).}).
#pos(n734,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n735,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n736,{},{accept},{input(0,0).trace_length(1).}).
#pos(n737,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).input(13,1).input(14,1).trace_length(15).}).
#pos(n738,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).trace_length(6).}).
#pos(n739,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n740,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n741,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n742,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).input(14,1).trace_length(15).}).
#pos(n743,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).trace_length(8).}).
#pos(n744,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n745,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).trace_length(9).}).
#pos(n746,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n747,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n748,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n749,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n750,{},{accept},{input(0,0).trace_length(1).}).
#pos(n751,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).trace_length(4).}).
#pos(n752,{},{accept},{input(0,0).trace_length(1).}).
#pos(n753,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n754,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).trace_length(5).}).
#pos(n755,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n756,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n757,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n758,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n759,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n760,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).trace_length(5).}).
#pos(n761,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n762,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n763,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n764,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n765,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,1).trace_length(14).}).
#pos(n766,{},{accept},{input(0,0).trace_length(1).}).
#pos(n767,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,1).input(14,1).trace_length(15).}).
#pos(n768,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n769,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n770,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n771,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n772,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n773,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).trace_length(8).}).
#pos(n774,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n775,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n776,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).trace_length(11).}).
#pos(n777,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,0).input(11,1).input(12,0).input(13,0).input(14,0).input(15,0).input(16,0).input(17,0).input(18,1).trace_length(19).}).
#pos(n778,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n779,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).trace_length(10).}).
#pos(n780,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n781,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,0).input(14,1).input(15,0).trace_length(16).}).
#pos(n782,{},{accept},{input(0,0).trace_length(1).}).
#pos(n783,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n784,{},{accept},{input(0,0).trace_length(1).}).
#pos(n785,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n786,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).trace_length(5).}).
#pos(n787,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n788,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n789,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n790,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).trace_length(10).}).
#pos(n791,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n792,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).trace_length(11).}).
#pos(n793,{},{accept},{input(0,0).trace_length(1).}).
#pos(n794,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n795,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n796,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n797,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n798,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,0).input(11,0).input(12,1).input(13,0).input(14,1).input(15,0).trace_length(16).}).
#pos(n799,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n800,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).input(13,1).input(14,1).trace_length(15).}).
#pos(n801,{},{accept},{input(0,1).input(1,0).input(2,1).trace_length(3).}).
#pos(n802,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n803,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,0).trace_length(12).}).
#pos(n804,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n805,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n806,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n807,{},{accept},{input(0,0).trace_length(1).}).
#pos(n808,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n809,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n810,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n811,{},{accept},{input(0,0).trace_length(1).}).
#pos(n812,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).trace_length(11).}).
#pos(n813,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n814,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n815,{},{accept},{input(0,0).input(1,0).input(2,0).trace_length(3).}).
#pos(n816,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).trace_length(5).}).
#pos(n817,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n818,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n819,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n820,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).trace_length(4).}).
#pos(n821,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).trace_length(9).}).
#pos(n822,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n823,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n824,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n825,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n826,{},{accept},{input(0,0).trace_length(1).}).
#pos(n827,{},{accept},{input(0,1).input(1,1).trace_length(2).}).
#pos(n828,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n829,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#pos(n830,{},{accept},{input(0,0).trace_length(1).}).
#pos(n831,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,1).input(15,1).input(16,0).trace_length(17).}).
#pos(n832,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).input(13,1).input(14,1).trace_length(15).}).
#pos(n833,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n834,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(n835,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n836,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).input(12,1).input(13,1).input(14,1).input(15,1).trace_length(16).}).
#pos(n837,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
#pos(n838,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).trace_length(4).}).
#pos(n839,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n840,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n841,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n842,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).trace_length(9).}).
#pos(n843,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n844,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n845,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).trace_length(14).}).
#pos(n846,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n847,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,0).input(11,0).trace_length(12).}).
#pos(n848,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).trace_length(6).}).
#pos(n849,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).trace_length(5).}).
#pos(n850,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n851,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n852,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).trace_length(9).}).
#pos(n853,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n854,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,0).input(11,1).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n855,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).input(14,0).trace_length(15).}).
#pos(n856,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,1).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n857,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).trace_length(4).}).
#pos(n858,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n859,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n860,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n861,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).trace_length(7).}).
#pos(n862,{},{accept},{input(0,1).input(1,0).input(2,0).trace_length(3).}).
#pos(n863,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n864,{},{accept},{input(0,0).trace_length(1).}).
#pos(n865,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).trace_length(14).}).
#pos(n866,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n867,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).trace_length(6).}).
#pos(n868,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).trace_length(13).}).
#pos(n869,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n870,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).trace_length(7).}).
#pos(n871,{},{accept},{input(0,0).trace_length(1).}).
#pos(n872,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,1).input(15,1).trace_length(16).}).
#pos(n873,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).input(14,1).input(15,0).input(16,0).input(17,1).trace_length(18).}).
#pos(n874,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).trace_length(4).}).
#pos(n875,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,1).input(12,1).trace_length(13).}).
#pos(n876,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).trace_length(4).}).
#pos(n877,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,1).input(5,0).input(6,1).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,1).input(13,0).trace_length(14).}).
#pos(n878,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,0).input(10,0).input(11,1).trace_length(12).}).
#pos(n879,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,0).trace_length(11).}).
#pos(n880,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,0).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n881,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,1).input(12,1).input(13,1).trace_length(14).}).
#pos(n882,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).input(13,0).input(14,0).trace_length(15).}).
#pos(n883,{},{accept},{input(0,0).trace_length(1).}).
#pos(n884,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).input(14,0).trace_length(15).}).
#pos(n885,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).input(13,0).trace_length(14).}).
#pos(n886,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).input(11,1).trace_length(12).}).
#pos(n887,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,1).input(14,0).input(15,1).input(16,0).input(17,0).trace_length(18).}).
#pos(n888,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,0).input(11,0).trace_length(12).}).
#pos(n889,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,1).input(10,1).input(11,0).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n890,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,0).input(12,0).input(13,0).input(14,0).trace_length(15).}).
#pos(n891,{},{accept},{input(0,0).input(1,1).input(2,1).input(3,0).input(4,0).trace_length(5).}).
#pos(n892,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,1).input(8,0).input(9,0).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n893,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,1).input(6,0).trace_length(7).}).
#pos(n894,{},{accept},{input(0,0).trace_length(1).}).
#pos(n895,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n896,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).trace_length(12).}).
#pos(n897,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,0).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).input(15,0).trace_length(16).}).
#pos(n898,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n899,{},{accept},{input(0,0).trace_length(1).}).
#pos(n900,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n901,{},{accept},{input(0,1).input(1,0).input(2,1).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,0).input(10,1).input(11,1).input(12,1).trace_length(13).}).
#pos(n902,{},{accept},{input(0,1).input(1,0).trace_length(2).}).
#pos(n903,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).trace_length(4).}).
#pos(n904,{},{accept},{input(0,0).input(1,1).trace_length(2).}).
#pos(n905,{},{accept},{input(0,0).input(1,1).input(2,0).trace_length(3).}).
#pos(n906,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,0).input(4,0).input(5,0).input(6,0).input(7,0).input(8,0).input(9,1).input(10,1).input(11,1).input(12,0).input(13,1).input(14,0).trace_length(15).}).
#pos(n907,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,0).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,1).input(10,0).input(11,0).input(12,0).trace_length(13).}).
#pos(n908,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).input(13,0).input(14,0).input(15,0).trace_length(16).}).
#pos(n909,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,1).input(4,1).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,0).input(11,1).input(12,0).trace_length(13).}).
#pos(n910,{},{accept},{input(0,0).input(1,0).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).input(13,1).trace_length(14).}).
#pos(n911,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,0).input(5,0).input(6,1).input(7,1).input(8,0).input(9,1).input(10,1).input(11,1).input(12,1).input(13,0).input(14,0).input(15,0).trace_length(16).}).
#pos(n912,{},{accept},{input(0,0).input(1,1).input(2,0).input(3,1).input(4,0).trace_length(5).}).
#pos(n913,{},{accept},{input(0,1).input(1,0).input(2,0).input(3,1).input(4,1).input(5,0).input(6,1).input(7,0).input(8,0).input(9,0).input(10,1).input(11,0).input(12,0).input(13,1).trace_length(14).}).
#pos(n914,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,1).input(4,1).input(5,0).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,0).input(12,1).trace_length(13).}).
#pos(n915,{},{accept},{input(0,1).input(1,1).input(2,0).trace_length(3).}).
#pos(n916,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,1).input(4,0).input(5,1).input(6,0).input(7,0).input(8,1).input(9,0).input(10,1).trace_length(11).}).
#pos(n917,{},{accept},{input(0,0).input(1,0).input(2,1).input(3,0).input(4,1).input(5,0).input(6,0).input(7,1).input(8,1).input(9,0).input(10,0).input(11,0).input(12,1).trace_length(13).}).
#pos(n918,{},{accept},{input(0,1).input(1,1).input(2,1).input(3,0).input(4,1).input(5,1).input(6,1).input(7,0).input(8,1).input(9,0).input(10,0).trace_length(11).}).
#pos(n919,{},{accept},{input(0,0).trace_length(1).}).
#pos(n920,{},{accept},{input(0,1).input(1,1).input(2,0).input(3,0).input(4,0).input(5,1).input(6,1).input(7,1).input(8,1).input(9,1).input(10,1).input(11,1).input(12,0).trace_length(13).}).
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
2 ~ delta(state0,V0,state5,0) :- input(_,V0).
2 ~ delta(state0,V0,state5,1) :- input(_,V0).
2 ~ delta(state0,V0,state5,2) :- input(_,V0).
2 ~ delta(state0,V0,state6,0) :- input(_,V0).
2 ~ delta(state0,V0,state6,1) :- input(_,V0).
2 ~ delta(state0,V0,state6,2) :- input(_,V0).
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
2 ~ delta(state1,V0,state5,0) :- input(_,V0).
2 ~ delta(state1,V0,state5,1) :- input(_,V0).
2 ~ delta(state1,V0,state5,2) :- input(_,V0).
2 ~ delta(state1,V0,state6,0) :- input(_,V0).
2 ~ delta(state1,V0,state6,1) :- input(_,V0).
2 ~ delta(state1,V0,state6,2) :- input(_,V0).
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
2 ~ delta(state2,V0,state5,0) :- input(_,V0).
2 ~ delta(state2,V0,state5,1) :- input(_,V0).
2 ~ delta(state2,V0,state5,2) :- input(_,V0).
2 ~ delta(state2,V0,state6,0) :- input(_,V0).
2 ~ delta(state2,V0,state6,1) :- input(_,V0).
2 ~ delta(state2,V0,state6,2) :- input(_,V0).
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
2 ~ delta(state3,V0,state5,0) :- input(_,V0).
2 ~ delta(state3,V0,state5,1) :- input(_,V0).
2 ~ delta(state3,V0,state5,2) :- input(_,V0).
2 ~ delta(state3,V0,state6,0) :- input(_,V0).
2 ~ delta(state3,V0,state6,1) :- input(_,V0).
2 ~ delta(state3,V0,state6,2) :- input(_,V0).
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
2 ~ delta(state4,V0,state5,0) :- input(_,V0).
2 ~ delta(state4,V0,state5,1) :- input(_,V0).
2 ~ delta(state4,V0,state5,2) :- input(_,V0).
2 ~ delta(state4,V0,state6,0) :- input(_,V0).
2 ~ delta(state4,V0,state6,1) :- input(_,V0).
2 ~ delta(state4,V0,state6,2) :- input(_,V0).
2 ~ delta(state5,V0,state0,0) :- input(_,V0).
2 ~ delta(state5,V0,state0,1) :- input(_,V0).
2 ~ delta(state5,V0,state0,2) :- input(_,V0).
2 ~ delta(state5,V0,state1,0) :- input(_,V0).
2 ~ delta(state5,V0,state1,1) :- input(_,V0).
2 ~ delta(state5,V0,state1,2) :- input(_,V0).
2 ~ delta(state5,V0,state2,0) :- input(_,V0).
2 ~ delta(state5,V0,state2,1) :- input(_,V0).
2 ~ delta(state5,V0,state2,2) :- input(_,V0).
2 ~ delta(state5,V0,state3,0) :- input(_,V0).
2 ~ delta(state5,V0,state3,1) :- input(_,V0).
2 ~ delta(state5,V0,state3,2) :- input(_,V0).
2 ~ delta(state5,V0,state4,0) :- input(_,V0).
2 ~ delta(state5,V0,state4,1) :- input(_,V0).
2 ~ delta(state5,V0,state4,2) :- input(_,V0).
2 ~ delta(state5,V0,state5,0) :- input(_,V0).
2 ~ delta(state5,V0,state5,1) :- input(_,V0).
2 ~ delta(state5,V0,state5,2) :- input(_,V0).
2 ~ delta(state5,V0,state6,0) :- input(_,V0).
2 ~ delta(state5,V0,state6,1) :- input(_,V0).
2 ~ delta(state5,V0,state6,2) :- input(_,V0).
2 ~ delta(state6,V0,state0,0) :- input(_,V0).
2 ~ delta(state6,V0,state0,1) :- input(_,V0).
2 ~ delta(state6,V0,state0,2) :- input(_,V0).
2 ~ delta(state6,V0,state1,0) :- input(_,V0).
2 ~ delta(state6,V0,state1,1) :- input(_,V0).
2 ~ delta(state6,V0,state1,2) :- input(_,V0).
2 ~ delta(state6,V0,state2,0) :- input(_,V0).
2 ~ delta(state6,V0,state2,1) :- input(_,V0).
2 ~ delta(state6,V0,state2,2) :- input(_,V0).
2 ~ delta(state6,V0,state3,0) :- input(_,V0).
2 ~ delta(state6,V0,state3,1) :- input(_,V0).
2 ~ delta(state6,V0,state3,2) :- input(_,V0).
2 ~ delta(state6,V0,state4,0) :- input(_,V0).
2 ~ delta(state6,V0,state4,1) :- input(_,V0).
2 ~ delta(state6,V0,state4,2) :- input(_,V0).
2 ~ delta(state6,V0,state5,0) :- input(_,V0).
2 ~ delta(state6,V0,state5,1) :- input(_,V0).
2 ~ delta(state6,V0,state5,2) :- input(_,V0).
2 ~ delta(state6,V0,state6,0) :- input(_,V0).
2 ~ delta(state6,V0,state6,1) :- input(_,V0).
2 ~ delta(state6,V0,state6,2) :- input(_,V0).

state(state0).
state(state1).
state(state2).
state(state3).
state(state4).
state(state5).
state(state6).
accept :- st(T,state6),trace_length(T).
#max_penalty(30).
