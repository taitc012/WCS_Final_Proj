function getAoA( dirName );

fileName = sprintf( '%s/result/tmp_rx_ant.mat', dirName );
load(fileName);
[n_samples, ANT_CNT] = size(tmp_rx_ant);
f = 2.49*1e9;
c = 3*1e8;
n_signal = 1;
lambda = c/f*100;

[ AoA_degree, AoA_prob ] = music( tmp_rx_ant(:,1:7)', n_signal, ANT_CNT-1, lambda, lambda/2, n_samples, 0 );
AoA_expectations = AoA_degree*AoA_prob'/sum(AoA_prob);
save( [dirName '/AoA_expectations.mat'], 'AoA_expectations' );
end