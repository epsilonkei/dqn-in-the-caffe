#!/usr/bin/env bash

if [ ! "$XRANGE" ]; then XRANGE="[*:*]"; fi
if [ ! "$YRANGE" ]; then YRANGE="[*:*]"; fi

echo "XRANGE=$XRANGE";
echo "YRANGE=$YRANGE";

gnuplot <<EOF
set terminal postscript eps color enhanced
set tics font "Times New Roman,25"
set xlabel font "Times New Roman,25"
set ylabel font "Times New Roman,25"
set zlabel font "Times New Roman,25"
set key font "Times New Roman,25"
set key right top
set key width 8
set output "train_eval.eps"
set grid
set size ratio 0.5
set xlabel "Epoch"
set ylabel "Reward"
set xrange ${XRANGE}
set title "_"
plot "train_log.csv" using 1:2 with line linewidth 3 title "Train", "eval_log.csv" using 1:2 with line linewidth 3 title "Eval"
EOF

gnuplot <<EOF
set terminal postscript eps color enhanced
set tics font "Times New Roman,25"
set xlabel font "Times New Roman,25"
set ylabel font "Times New Roman,25"
set zlabel font "Times New Roman,25"
set key font "Times New Roman,25"
set key right top
set key width 8
set output "q_log.eps"
set grid
set size ratio 0.5
set xlabel "Epoch"
set ylabel "Average maximum predicted q-value"
set xrange ${XRANGE}
set title "_"
plot "q_log.csv" using 1:2 with line linewidth 3 title "q-value"
EOF
