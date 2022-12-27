res = readtable('r0776947.csv');
x = transpose(table2array(res(:,1)));
avg = transpose(table2array(res(:,3)));
best = transpose(table2array(res(:,4)));
subplot(3,1,1);
plot(x,avg);
subplot(3,1,2); 
plot(x,best);
subplot(3,1,3);
plot(x,best,x,avg);
 
